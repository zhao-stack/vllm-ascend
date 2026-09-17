# SPDX-License-Identifier: Apache-2.0
"""Network-free tests of QA budgets, complete coverage and failure reporting."""

import json
import tempfile
import unittest
from pathlib import Path

from qa import MAX_INPUT_BYTES, MAX_OUTPUT_TOKENS, review

ROOT = "0123456789abcdef"
URL = "https://github.com/vllm-project/vllm/blob/" + "a" * 40 + "/vllm/api.py#L1"
MARKDOWN = f"# QA\n\n## {ROOT} — 候选\n\n[source]({URL})\n"


class QATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.source = self.path / "qa-review.md"
        self.source.write_bytes(MARKDOWN.encode("utf-8"))
        self.output = self.path / "result"
        self.calls = []

    def response(self, body, key):
        self.calls.append(body)
        self.assertNotIn("tools", body)
        self.assertEqual(body["max_tokens"], MAX_OUTPUT_TOKENS)
        self.assertEqual(body["messages"][1]["content"], MARKDOWN)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "reviews": [
                                    {
                                        "root_cause_id": ROOT,
                                        "verdict": "insufficient_evidence",
                                        "reason": "未提供新版删除或迁移证据。",
                                        "evidence": [URL],
                                    }
                                ]
                            }
                        )
                    },
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
        }

    def status(self):
        return json.loads((self.output / "qa-status.json").read_text(encoding="utf-8"))

    def test_complete_review_preserves_input_and_usage(self):
        before = self.source.read_bytes()
        self.assertEqual(review(self.source, self.output, "test-only", self.response), 0)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.status()["counts"]["insufficient_evidence"], 1)
        self.assertEqual(self.status()["usage"]["total_tokens"], 140)
        self.assertIn(ROOT, (self.output / "qa-verdict.md").read_text(encoding="utf-8"))
        self.assertNotIn("test-only", (self.output / "qa-status.json").read_text())

    def test_missing_key_and_oversize_never_call_provider(self):
        self.assertEqual(review(self.source, self.output, "", self.response), 1)
        self.assertEqual(self.status()["model_calls"], 0)
        self.source.write_text("x" * (MAX_INPUT_BYTES + 1))
        self.assertEqual(review(self.source, self.output, "test-only", self.response), 1)
        self.assertEqual(self.status()["model_calls"], 0)
        self.assertFalse(self.calls)

    def test_no_roots_skip_model(self):
        self.source.write_text("# No upgrade\n")
        self.assertEqual(review(self.source, self.output, "test-only", self.response), 0)
        self.assertEqual(self.status()["qa"], "skipped_no_candidates")
        self.assertFalse(self.calls)

    def test_incomplete_invalid_and_unknown_evidence_fail(self):
        def broken(kind):
            def transport(body, key):
                result = self.response(body, key)
                choice = result["choices"][0]
                data = json.loads(choice["message"]["content"])
                if kind == "truncated":
                    choice["finish_reason"] = "length"
                elif kind == "missing":
                    data["reviews"] = []
                elif kind == "duplicate":
                    data["reviews"] *= 2
                elif kind == "unknown":
                    data["reviews"][0]["evidence"] = ["https://github.com/invented/source"]
                elif kind == "no_evidence":
                    data["reviews"][0].update(verdict="confirm", evidence=[])
                choice["message"]["content"] = json.dumps(data)
                return result

            return transport

        for kind in ("truncated", "missing", "duplicate", "unknown", "no_evidence"):
            with self.subTest(kind=kind):
                self.assertEqual(review(self.source, self.output, "test-only", broken(kind)), 1)
                self.assertEqual(self.status()["qa"], "failed")
                self.assertEqual(self.status()["model_calls"], 1)
                self.assertFalse((self.output / "qa-verdict.md").exists())

    def test_timeout_is_one_attempt_without_retry(self):
        def timeout(body, key):
            self.calls.append(body)
            raise TimeoutError("sensitive provider transport detail")

        self.assertEqual(review(self.source, self.output, "test-only", timeout), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.status()["model_calls"], 1)
        self.assertNotIn("sensitive", (self.output / "qa-status.json").read_text())


if __name__ == "__main__":
    unittest.main()
