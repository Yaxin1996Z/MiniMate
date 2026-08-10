"""配置模块单元测试"""

import json
import os
import tempfile
import unittest

from minimate.config import get_mcp_servers, load_config


class ConfigTest(unittest.TestCase):
    def test_default_when_file_missing(self):
        cfg = load_config("/nonexistent/config.json")
        self.assertEqual(cfg["mcp"]["servers"], [])

    def test_load_and_merge(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"mcp": {"servers": [{"name": "demo", "command": "python"}]}}, f)
            cfg = load_config(path)
            self.assertEqual(len(cfg["mcp"]["servers"]), 1)
            self.assertEqual(cfg["mcp"]["servers"][0]["name"], "demo")

    def test_broken_json_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{broken")
            cfg = load_config(path)
            self.assertEqual(cfg["mcp"]["servers"], [])

    def test_get_mcp_servers(self):
        cfg = {"mcp": {"servers": [{"name": "a"}]}}
        self.assertEqual(len(get_mcp_servers(cfg)), 1)
        self.assertEqual(get_mcp_servers({"mcp": {"servers": []}}), [])


if __name__ == "__main__":
    unittest.main()
