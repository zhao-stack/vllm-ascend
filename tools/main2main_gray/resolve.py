# SPDX-License-Identifier: Apache-2.0
"""Resolve the workflow's source and upgrade range before invoking the analyzer."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from run import clean_head, git, json_write, require_sha


def resolve_range(
    up: Path, down: Path, target: str = "", baseline_url: str = "", baseline_ref: str = "refs/heads/main2main_baseline"
) -> dict:
    new = require_sha(git(up, "rev-parse", "HEAD"))
    source = require_sha(git(down, "rev-parse", "HEAD"))
    clean_head(up, new)
    clean_head(down, source)
    if target and require_sha(target) != new:
        raise ValueError("Target does not match the frozen vLLM checkout")
    mode = "selected_source"
    baseline = None
    fallback = None
    if baseline_url:
        # Read only the personal fork's state; no pushes and no production runner.
        ref = baseline_ref
        if not ref.startswith("refs/heads/"):
            raise ValueError("Baseline must be a branch ref")
        remote = git(down, "ls-remote", baseline_url, ref)
        if remote:
            baseline = require_sha(remote.split()[0])
            git(down, "fetch", "--no-tags", baseline_url, baseline)
            marker = git(down, "show", f"{baseline}:.github/vllm-main-verified.commit")
            require_sha(marker)
            behind = subprocess.run(
                ["git", "-C", str(up), "merge-base", "--is-ancestor", new, marker],
                capture_output=True,
            ).returncode
            if behind not in (0, 1):
                raise ValueError("Cannot verify the accumulated baseline in vLLM history")
            if target and behind == 0:
                fallback = "explicit_target_at_or_before_baseline"
            else:
                git(down, "checkout", "--detach", baseline)
                git(down, "config", "user.name", "main2main-gray")
                git(down, "config", "user.email", "main2main-gray@users.noreply.github.com")
                try:
                    # Stable commit identities allow reuse for identical inputs.
                    git(down, "rebase", "--committer-date-is-author-date", source)
                    mode = "incremental_rebased"
                except subprocess.CalledProcessError:
                    git(down, "rebase", "--abort")
                    git(down, "checkout", "--detach", source)
                    fallback = "baseline_rebase_conflict"
        else:
            fallback = "no_baseline_ref"
        if mode != "incremental_rebased":
            mode = "fresh"
    ascend = require_sha(git(down, "rev-parse", "HEAD"))
    clean_head(down, ascend)
    # Current main2main repositories use this marker. Missing markers fail closed.
    old = require_sha(git(down, "show", f"{ascend}:.github/vllm-main-verified.commit"))
    git(up, "merge-base", "--is-ancestor", old, new)
    return {
        "vllm_old_sha": old,
        "vllm_new_sha": new,
        "vllm_ascend_sha": ascend,
        "source_sha_before_resolution": source,
        "baseline_sha": baseline,
        "source_mode": mode,
        "fallback_reason": fallback,
        "range_source": "selected_ascend_marker_to_frozen_vllm_head",
        "has_changes": old != new,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--ascend-root", type=Path, required=True)
    parser.add_argument("--target", default="")
    parser.add_argument("--baseline-url", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args()
    result = resolve_range(args.vllm_root, args.ascend_root, args.target, args.baseline_url)
    json_write(args.output, result)
    with args.github_output.open("a", encoding="utf-8") as stream:
        for key in ("vllm_old_sha", "vllm_new_sha", "vllm_ascend_sha"):
            stream.write(f"{key}={result[key]}\n")
        stream.write(f"has_changes={str(result['has_changes']).lower()}\n")
    print(result)


if __name__ == "__main__":
    main()
