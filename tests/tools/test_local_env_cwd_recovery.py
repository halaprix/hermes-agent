"""Tests for LocalEnvironment recovery when ``self.cwd`` is deleted.

When a tool call inside the persistent terminal session ``rm -rf``'s its own
working directory, the next ``subprocess.Popen(..., cwd=self.cwd)`` would
otherwise raise ``FileNotFoundError`` before bash starts, wedging every
subsequent terminal/file-tool call until the gateway restarts.

Regression coverage for https://github.com/NousResearch/hermes-agent/issues/17558.

Also covers #65583: a working directory that *exists* but the runtime user
cannot enter (e.g. a ``/root`` cwd captured while running as root and replayed
under a non-root gateway) makes ``Popen(cwd=...)`` raise ``PermissionError``
just as fatally, so ``_resolve_safe_cwd`` must reject it too.
"""

import os
import shutil
import tempfile
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools.environments.local import (
    LocalEnvironment,
    _resolve_safe_cwd,
)

_skip_as_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses directory permission checks, so os.access can't "
    "simulate an unenterable dir",
)
_posix_only = pytest.mark.skipif(
    os.name != "posix",
    reason="chmod-based unenterable-dir simulation is POSIX-only",
)


class TestResolveSafeCwd:
    """Pure-function unit tests for the recovery helper."""

    def test_returns_cwd_when_directory_exists(self, tmp_path):
        path = str(tmp_path)
        assert _resolve_safe_cwd(path) == path

    def test_walks_up_to_first_existing_ancestor(self, tmp_path):
        nested = tmp_path / "child" / "grandchild"
        nested.mkdir(parents=True)
        deleted = str(nested)
        shutil.rmtree(tmp_path / "child")

        # The deepest existing ancestor on the path is tmp_path itself.
        assert _resolve_safe_cwd(deleted) == str(tmp_path)

    def test_falls_back_when_path_is_empty(self):
        assert _resolve_safe_cwd("") == tempfile.gettempdir()

    def test_returns_tempdir_when_nothing_on_path_exists(self, monkeypatch):
        monkeypatch.setattr(os.path, "isdir", lambda p: False)
        assert _resolve_safe_cwd("/no/such/dir") == tempfile.gettempdir()

    def test_returns_root_when_only_root_exists(self, monkeypatch):
        """If every ancestor except the filesystem root is gone, the root
        itself is still a valid recovery target — don't skip it just because
        ``os.path.dirname('/') == '/'`` is the loop's exit condition."""
        sep = os.path.sep
        monkeypatch.setattr(os.path, "isdir", lambda p: p == sep)
        assert _resolve_safe_cwd("/no/such/deep/dir") == sep


class TestResolveSafeCwdUnenterable:
    """#65583: a cwd that exists but the user can't enter must be rejected.

    ``Popen(cwd=path)`` chdir()s into *path* in the child, which needs
    execute/search permission on the directory. A ``/root`` working dir
    captured while running as root and replayed under a non-root gateway
    ``exists`` (so ``os.path.isdir`` is True) but is unenterable, and Popen
    raises ``PermissionError`` before bash starts — wedging every terminal and
    file-tool call. ``_resolve_safe_cwd`` must treat it like a missing cwd.
    """

    @_posix_only
    @_skip_as_root
    def test_unenterable_dir_falls_back_to_home(self, tmp_path, monkeypatch):
        locked = tmp_path / "root-like"
        locked.mkdir(mode=0o000)
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        try:
            assert os.path.isdir(str(locked))  # it exists…
            assert not os.access(str(locked), os.X_OK)  # …but is unenterable
            assert _resolve_safe_cwd(str(locked)) == str(home)
        finally:
            os.chmod(str(locked), 0o700)  # let tmp_path cleanup remove it

    @_posix_only
    @_skip_as_root
    def test_unenterable_dir_falls_back_to_tempdir_when_home_unusable(
        self, tmp_path, monkeypatch
    ):
        locked = tmp_path / "root-like"
        locked.mkdir(mode=0o000)
        # HOME points at another unenterable dir → must reach the tempdir floor.
        bad_home = tmp_path / "bad-home"
        bad_home.mkdir(mode=0o000)
        monkeypatch.setenv("HOME", str(bad_home))
        try:
            assert _resolve_safe_cwd(str(locked)) == tempfile.gettempdir()
        finally:
            os.chmod(str(locked), 0o700)
            os.chmod(str(bad_home), 0o700)

    @_posix_only
    @_skip_as_root
    def test_run_bash_recovers_from_unenterable_cwd(self, tmp_path, monkeypatch, caplog):
        locked = tmp_path / "root-like"
        locked.mkdir(mode=0o000)
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))

        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=str(locked), timeout=10)

        captured: dict = {}
        fds: list = []
        try:
            with patch("tools.environments.local._find_bash", return_value="/bin/bash"), \
                 patch("subprocess.Popen", side_effect=_make_fake_popen(captured, fds)), \
                 patch("tools.terminal_tool._interrupt_event", _fake_interrupt()), \
                 caplog.at_level("WARNING", logger="tools.environments.local"):
                env.execute("echo hello")
        finally:
            _close_fds(fds)
            os.chmod(str(locked), 0o700)

        # Popen must have been handed an enterable directory, NOT the locked one.
        assert captured["cwd"] == str(home)
        assert os.access(captured["cwd"], os.X_OK)
        assert env.cwd == str(home)
        # The warning must name the real reason (accessibility), not "missing".
        assert any("not accessible" in rec.message for rec in caplog.records)
        assert not any("missing on disk" in rec.message for rec in caplog.records)


def _fake_interrupt():
    return threading.Event()


def _make_fake_popen(captured: dict, fds: list):
    """Build a fake ``Popen`` whose ``stdout`` exposes a real OS file
    descriptor so ``BaseEnvironment._wait_for_process`` can call
    ``select.select([fd], ...)`` and ``os.read(fd, ...)`` against it without
    tripping ``TypeError: fileno() returned a non-integer`` from a MagicMock
    ``fileno()`` (or worse, accidentally reading from the test runner's own
    stdout).

    The pipe's write end is closed immediately so the drain loop sees EOF on
    the first iteration.  Every fd handed out is appended to ``fds`` so the
    caller can clean up after the test.
    """
    def fake_popen(cmd, **kwargs):
        captured["cwd"] = kwargs.get("cwd")
        captured["env"] = kwargs.get("env", {})
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        stdout = os.fdopen(read_fd, "rb", buffering=0)
        fds.append(stdout)
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.returncode = 0
        proc.stdout = stdout
        proc.stdin = MagicMock()
        return proc
    return fake_popen


def _close_fds(fds):
    for f in fds:
        try:
            f.close()
        except Exception:
            pass


class TestRunBashCwdRecovery:
    """End-to-end recovery: deleted ``self.cwd`` must not crash Popen."""

    def test_recovers_when_cwd_deleted_after_init(self, tmp_path, caplog):
        """Reproduces the wedge from #17558: cwd was valid when the
        snapshot was taken, but a subsequent command deleted it before the
        next ``Popen``."""
        wedged = tmp_path / "wedge-repro"
        wedged.mkdir()

        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=str(wedged), timeout=10)

        # The previous tool call deleted the working directory.
        shutil.rmtree(wedged)
        assert env.cwd == str(wedged) and not os.path.isdir(env.cwd)

        captured = {}
        fds: list = []
        try:
            with patch("tools.environments.local._find_bash", return_value="/bin/bash"), \
                 patch("subprocess.Popen", side_effect=_make_fake_popen(captured, fds)), \
                 patch("tools.terminal_tool._interrupt_event", _fake_interrupt()), \
                 caplog.at_level("WARNING", logger="tools.environments.local"):
                env.execute("echo hello")
        finally:
            _close_fds(fds)

        # Popen must have been handed a real, existing directory.
        assert captured["cwd"] == str(tmp_path)
        assert os.path.isdir(captured["cwd"])

        # ``self.cwd`` is updated so the next call doesn't re-warn.
        assert env.cwd == str(tmp_path)

        # The warning surfaces the wedge so it isn't silently masked.
        assert any("missing on disk" in rec.message for rec in caplog.records)

    def test_no_warning_when_cwd_still_exists(self, tmp_path, caplog):
        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=str(tmp_path), timeout=10)

        captured = {}
        fds: list = []
        try:
            with patch("tools.environments.local._find_bash", return_value="/bin/bash"), \
                 patch("subprocess.Popen", side_effect=_make_fake_popen(captured, fds)), \
                 patch("tools.terminal_tool._interrupt_event", _fake_interrupt()), \
                 caplog.at_level("WARNING", logger="tools.environments.local"):
                env.execute("echo hello")
        finally:
            _close_fds(fds)

        assert captured["cwd"] == str(tmp_path)
        assert env.cwd == str(tmp_path)
        assert not any("missing on disk" in rec.message for rec in caplog.records)


class TestUpdateCwdRejectsMissingPaths:
    """``_update_cwd`` must not propagate a deleted path back into ``self.cwd``."""

    def test_skips_assignment_when_marker_path_missing(self, tmp_path):
        original = tmp_path / "starting"
        original.mkdir()

        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=str(original), timeout=10)

        # Simulate the stale-marker case: the prior command's ``pwd -P`` left
        # a path in the cwd file, but that path has since been deleted.
        deleted = tmp_path / "wedge-repro"
        with open(env._cwd_file, "w") as f:
            f.write(str(deleted))

        env._update_cwd({"output": "", "returncode": 0})

        assert env.cwd == str(original)

    def test_accepts_assignment_when_marker_path_exists(self, tmp_path):
        original = tmp_path / "starting"
        original.mkdir()
        new_dir = tmp_path / "next"
        new_dir.mkdir()

        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=str(original), timeout=10)

        with open(env._cwd_file, "w") as f:
            f.write(str(new_dir))

        env._update_cwd({"output": "", "returncode": 0})

        assert env.cwd == str(new_dir)
