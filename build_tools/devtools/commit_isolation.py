# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs a command plan against the prospective Git commit snapshot."""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from build_tools.devtools.command_plan import CommandPlan


def _git(
    repo_root: Path,
    arguments: list[str],
    *,
    capture_output: bool = False,
    input_data: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        input=input_data,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
    )


def _git_output(repo_root: Path, arguments: list[str]) -> str | None:
    result = _git(repo_root, arguments, capture_output=True)
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8").strip()


def _has_non_index_changes(repo_root: Path) -> bool:
    tracked_result = _git(repo_root, ["diff", "--quiet", "--"])
    if tracked_result.returncode == 1:
        return True
    if tracked_result.returncode != 0:
        raise RuntimeError("could not inspect unstaged tracked changes")
    untracked_result = _git(
        repo_root,
        ["ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True,
    )
    if untracked_result.returncode != 0:
        raise RuntimeError("could not inspect untracked files")
    return bool(untracked_result.stdout)


def _stash_ref_for_object(repo_root: Path, object_id: str) -> str | None:
    result = _git(
        repo_root,
        ["stash", "list", "--format=%H%x09%gd"],
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    matching_refs = []
    for line in result.stdout.decode("utf-8").splitlines():
        entry_object_id, separator, entry_ref = line.partition("\t")
        if separator and entry_object_id == object_id:
            matching_refs.append(entry_ref)
    if len(matching_refs) != 1:
        return None
    return matching_refs[0]


def _stash_unstaged_patch(repo_root: Path, object_id: str) -> bytes | None:
    result = _git(
        repo_root,
        [
            "diff",
            "--binary",
            "--full-index",
            "--unified=0",
            f"{object_id}^2",
            object_id,
            "--",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _stash_untracked_paths(repo_root: Path, object_id: str) -> list[str] | None:
    parents = _git_output(repo_root, ["rev-list", "--parents", "-n", "1", object_id])
    if parents is None:
        return None
    if len(parents.split()) < 4:
        return []
    result = _git(
        repo_root,
        ["ls-tree", "-r", "-z", "--name-only", f"{object_id}^3"],
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return [path.decode("utf-8") for path in result.stdout.split(b"\0") if path]


@dataclass
class _CommitIsolation:
    repo_root: Path
    stash_object_id: str | None = None
    stash_message: str | None = None

    def begin(self) -> bool:
        try:
            if not _has_non_index_changes(self.repo_root):
                return True
        except RuntimeError as exc:
            print(f"precommit: {exc}")
            return False

        index_tree = _git_output(self.repo_root, ["write-tree"])
        if index_tree is None:
            print("precommit: could not capture the prospective index tree")
            return False
        previous_stash = _git_output(
            self.repo_root, ["rev-parse", "--verify", "refs/stash"]
        )
        token = uuid.uuid4().hex
        self.stash_message = (
            f"iree precommit index isolation: "
            f"worktree={self.repo_root.name} pid={os.getpid()} token={token}"
        )
        stash_result = _git(
            self.repo_root,
            [
                "stash",
                "push",
                "--quiet",
                "--keep-index",
                "--include-untracked",
                "--message",
                self.stash_message,
                "--",
            ],
        )
        if stash_result.returncode != 0:
            print("precommit: could not isolate non-index changes")
            return False

        self.stash_object_id = _git_output(
            self.repo_root, ["rev-parse", "--verify", "refs/stash"]
        )
        if self.stash_object_id is None or self.stash_object_id == previous_stash:
            print("precommit: Git did not create the expected isolation stash")
            return False
        if _git_output(self.repo_root, ["write-tree"]) != index_tree:
            print("precommit: isolation changed the prospective index tree")
            return False
        try:
            if _has_non_index_changes(self.repo_root):
                print("precommit: isolation left non-index source changes visible")
                return False
        except RuntimeError as exc:
            print(f"precommit: {exc}")
            return False

        print(
            "precommit: isolated non-index changes in "
            f"{self.stash_object_id} ({self.stash_message})"
        )
        return True

    def restore(self) -> bool:
        if self.stash_object_id is None:
            return True
        stash_ref = _stash_ref_for_object(self.repo_root, self.stash_object_id)
        if stash_ref is None:
            print(
                "precommit: could not uniquely locate isolation stash "
                f"{self.stash_object_id}; restore it manually"
            )
            return False
        if (
            _git_output(self.repo_root, ["rev-parse", stash_ref])
            != self.stash_object_id
        ):
            print(
                f"precommit: isolation stash {stash_ref} moved; expected "
                f"{self.stash_object_id}; restore that object manually"
            )
            return False
        unstaged_patch = _stash_unstaged_patch(self.repo_root, self.stash_object_id)
        untracked_paths = _stash_untracked_paths(self.repo_root, self.stash_object_id)
        if unstaged_patch is None or untracked_paths is None:
            print(
                "precommit: could not read isolated non-index changes; the exact "
                f"stash remains at {self.stash_object_id} ({stash_ref})"
            )
            return False
        conflicting_untracked_paths = [
            path for path in untracked_paths if os.path.lexists(self.repo_root / path)
        ]
        if conflicting_untracked_paths:
            print(
                "precommit: generated paths conflict with isolated untracked "
                f"files: {', '.join(conflicting_untracked_paths)}"
            )
            print(
                "precommit: the exact isolation stash remains at "
                f"{self.stash_object_id} ({stash_ref})"
            )
            return False
        if unstaged_patch:
            check_result = _git(
                self.repo_root,
                ["apply", "--check", "--recount", "-"],
                input_data=unstaged_patch,
            )
            if check_result.returncode != 0:
                print(
                    "precommit: staged fixups conflict with isolated unstaged changes"
                )
                print(
                    "precommit: the exact isolation stash remains at "
                    f"{self.stash_object_id} ({stash_ref})"
                )
                return False
            apply_result = _git(
                self.repo_root,
                ["apply", "--recount", "-"],
                input_data=unstaged_patch,
            )
            if apply_result.returncode != 0:
                print(
                    "precommit: could not restore isolated unstaged changes; "
                    f"the exact stash remains at {self.stash_object_id} "
                    f"({stash_ref})"
                )
                return False
        for start in range(0, len(untracked_paths), 128):
            restore_result = _git(
                self.repo_root,
                [
                    "restore",
                    f"--source={self.stash_object_id}^3",
                    "--worktree",
                    "--",
                    *untracked_paths[start : start + 128],
                ],
            )
            if restore_result.returncode != 0:
                print(
                    "precommit: could not restore isolated untracked files; "
                    f"the exact stash remains at {self.stash_object_id} "
                    f"({stash_ref})"
                )
                return False
        drop_result = _git(self.repo_root, ["stash", "drop", "--quiet", stash_ref])
        if drop_result.returncode != 0:
            print(
                "precommit: non-index changes were restored but the exact "
                f"stash could not be dropped: {self.stash_object_id} ({stash_ref})"
            )
            return False
        print(f"precommit: restored non-index changes from {self.stash_object_id}")
        return True


@dataclass(frozen=True)
class CommitIsolationStep:
    """Runs all nested steps with non-index source changes stashed."""

    plan: CommandPlan
    repo_root: Path

    def describe(self) -> str:
        nested_description = self.plan.describe()
        if not nested_description:
            return "# isolate non-index changes for the prospective commit"
        return (
            "# isolate non-index changes for the prospective commit\n"
            f"{nested_description}"
        )

    def run(self, verbose: bool = False) -> int:
        isolation = _CommitIsolation(self.repo_root)
        plan_result = 1
        restore_result = True
        try:
            if isolation.begin():
                plan_result = self.plan.run(verbose=verbose)
        except KeyboardInterrupt:
            plan_result = 130
        finally:
            restore_result = isolation.restore()
        if not restore_result:
            return 1
        return plan_result
