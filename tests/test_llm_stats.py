"""LLM Token 用量统计单元测试"""

import unittest

from minimate import llm


class TokenStatsTest(unittest.TestCase):
    def setUp(self):
        llm.reset_stats()

    def test_record_usage(self):
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 20})()
        resp = type("R", (), {"usage": usage})()
        llm._record_usage(resp)
        stats = llm.get_stats()
        self.assertEqual(stats["calls"], 1)
        self.assertEqual(stats["prompt_tokens"], 10)
        self.assertEqual(stats["completion_tokens"], 20)

    def test_record_without_usage(self):
        resp = type("R", (), {"usage": None})()
        llm._record_usage(resp)
        self.assertEqual(llm.get_stats()["calls"], 1)
        self.assertEqual(llm.get_stats()["prompt_tokens"], 0)

    def test_reset(self):
        usage = type("U", (), {"prompt_tokens": 5, "completion_tokens": 5})()
        llm._record_usage(type("R", (), {"usage": usage})())
        llm.reset_stats()
        self.assertEqual(llm.get_stats(), {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})


if __name__ == "__main__":
    unittest.main()
