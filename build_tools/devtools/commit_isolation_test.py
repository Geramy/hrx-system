# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import io
import subprocess
import tempfile
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from build_tools.devtools.command_plan import CommandPlan
from build_tools.devtools.commit_isolation import CommitIsolationStep


@dataclass(frozen=True)
class CallbackStep:
    callback: Callable[[], int]

    def describe(self) -> str:
        return "test callback"

    def run(self, verbose: bool = False) -> int:
        del verbose
        return self.callback()


class CommitIsolationTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temporary_directory.name)
        self.hooks_directory = self.repo_root / "empty-hooks"
        self.hooks_directory.mkdir()
        self.git("init", "--quiet")
        self.git("config", "user.name", "Commit Isolation Test")
        self.git("config", "user.email", "commit-isolation@example.com")
        self.git("config", "core.hooksPath", str(self.hooks_directory))

    def tearDown(self):
        self.temporary_directory.cleanup()

    def git(self, *arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", *arguments],
            cwd=self.repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def git_text(self, *arguments: str) -> str:
        return self.git(*arguments).stdout.decode("utf-8")

    def commit_base(self, files: dict[str, str]) -> None:
        for relative_path, contents in files.items():
            (self.repo_root / relative_path).write_text(contents, encoding="utf-8")
        self.git("add", "--all")
        self.git("commit", "--quiet", "-m", "base")

    def run_isolated(self, callback: Callable[[], int]) -> tuple[int, str]:
        step = CommitIsolationStep(
            CommandPlan([CallbackStep(callback)]), self.repo_root
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = step.run()
        return result, output.getvalue()

    def test_restores_non_index_changes_after_staged_fixups(self):
        self.commit_base(
            {
                "shared.txt": "first base\nmiddle base\nlast base\n",
                "head-path.txt": "head base\n",
            }
        )
        shared_path = self.repo_root / "shared.txt"
        head_path = self.repo_root / "head-path.txt"
        untracked_path = self.repo_root / "untracked.txt"

        shared_path.write_text(
            "first staged\nmiddle base\nlast base\n", encoding="utf-8"
        )
        self.git("add", "shared.txt")
        shared_path.write_text(
            "first staged\nmiddle base\nlast unstaged\n", encoding="utf-8"
        )
        head_path.write_text("head unstaged\n", encoding="utf-8")
        untracked_path.write_text("untracked\n", encoding="utf-8")

        def apply_fixup() -> int:
            self.assertEqual(
                shared_path.read_text(encoding="utf-8"),
                "first staged\nmiddle base\nlast base\n",
            )
            self.assertEqual(head_path.read_text(encoding="utf-8"), "head base\n")
            self.assertFalse(untracked_path.exists())
            shared_path.write_text(
                "first staged\nmiddle formatted\nlast base\n", encoding="utf-8"
            )
            self.git("add", "shared.txt")
            return 0

        result, output = self.run_isolated(apply_fixup)

        self.assertEqual(result, 0)
        self.assertIn("isolated non-index changes", output)
        self.assertIn("restored non-index changes", output)
        self.assertEqual(
            shared_path.read_text(encoding="utf-8"),
            "first staged\nmiddle formatted\nlast unstaged\n",
        )
        self.assertEqual(
            self.git_text("show", ":shared.txt"),
            "first staged\nmiddle formatted\nlast base\n",
        )
        self.assertEqual(head_path.read_text(encoding="utf-8"), "head unstaged\n")
        self.assertEqual(untracked_path.read_text(encoding="utf-8"), "untracked\n")
        self.assertEqual(self.git_text("stash", "list"), "")

    def test_restores_non_index_changes_after_failed_plan(self):
        self.commit_base({"staged.txt": "base\n", "unstaged.txt": "base\n"})
        staged_path = self.repo_root / "staged.txt"
        unstaged_path = self.repo_root / "unstaged.txt"
        untracked_path = self.repo_root / "untracked.txt"
        staged_path.write_text("staged\n", encoding="utf-8")
        self.git("add", "staged.txt")
        unstaged_path.write_text("unstaged\n", encoding="utf-8")
        untracked_path.write_text("untracked\n", encoding="utf-8")

        def fail() -> int:
            self.assertEqual(staged_path.read_text(encoding="utf-8"), "staged\n")
            self.assertEqual(unstaged_path.read_text(encoding="utf-8"), "base\n")
            self.assertFalse(untracked_path.exists())
            return 7

        result, _ = self.run_isolated(fail)

        self.assertEqual(result, 7)
        self.assertEqual(staged_path.read_text(encoding="utf-8"), "staged\n")
        self.assertEqual(self.git_text("show", ":staged.txt"), "staged\n")
        self.assertEqual(unstaged_path.read_text(encoding="utf-8"), "unstaged\n")
        self.assertEqual(untracked_path.read_text(encoding="utf-8"), "untracked\n")
        self.assertEqual(self.git_text("stash", "list"), "")

    def test_restores_non_index_changes_after_plan_exception(self):
        self.commit_base({"unstaged.txt": "base\n"})
        unstaged_path = self.repo_root / "unstaged.txt"
        unstaged_path.write_text("unstaged\n", encoding="utf-8")

        def raise_error() -> int:
            self.assertEqual(unstaged_path.read_text(encoding="utf-8"), "base\n")
            raise RuntimeError("test failure")

        with self.assertRaisesRegex(RuntimeError, "test failure"):
            self.run_isolated(raise_error)

        self.assertEqual(unstaged_path.read_text(encoding="utf-8"), "unstaged\n")
        self.assertEqual(self.git_text("stash", "list"), "")

    def test_skips_stash_when_worktree_already_matches_index(self):
        self.commit_base({"staged.txt": "base\n"})
        staged_path = self.repo_root / "staged.txt"
        staged_path.write_text("staged\n", encoding="utf-8")
        self.git("add", "staged.txt")
        callback_called = False

        def succeed() -> int:
            nonlocal callback_called
            callback_called = True
            return 0

        result, output = self.run_isolated(succeed)

        self.assertEqual(result, 0)
        self.assertTrue(callback_called)
        self.assertNotIn("isolated non-index changes", output)
        self.assertEqual(self.git_text("stash", "list"), "")

    def test_retains_exact_stash_when_fixup_conflicts_with_unstaged_hunk(self):
        self.commit_base({"shared.txt": "first base\nlast base\n"})
        shared_path = self.repo_root / "shared.txt"
        shared_path.write_text("first staged\nlast base\n", encoding="utf-8")
        self.git("add", "shared.txt")
        shared_path.write_text("first staged\nlast unstaged\n", encoding="utf-8")

        def conflicting_fixup() -> int:
            shared_path.write_text("first staged\nlast formatted\n", encoding="utf-8")
            self.git("add", "shared.txt")
            return 0

        result, output = self.run_isolated(conflicting_fixup)

        self.assertEqual(result, 1)
        stash_object_id = self.git_text("stash", "list", "--format=%H").strip()
        self.assertTrue(stash_object_id)
        self.assertIn(stash_object_id, output)
        self.assertIn("conflict", output)
        self.assertEqual(
            self.git_text("show", ":shared.txt"),
            "first staged\nlast formatted\n",
        )


if __name__ == "__main__":
    unittest.main()
