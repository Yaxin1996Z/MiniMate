"""日志与审计单元测试"""

import logging
import os
import unittest

from minimate import logging as mlog


class CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class LoggingTest(unittest.TestCase):
    def test_log_path_under_logs(self):
        self.assertTrue(mlog.log_path().endswith("minimate.log"))
        self.assertIn("logs", mlog.log_path())

    def test_audit_dangerous_tool(self):
        handler = CaptureHandler()
        mlog.logger.addHandler(handler)
        try:
            mlog.audit_tool_call("run_shell", "echo x", "ok", 0.1)
            mlog.audit_tool_call("read_file", "a.txt", "内容", 0.01)
        finally:
            mlog.logger.removeHandler(handler)

        messages = [r.getMessage() for r in handler.records]
        self.assertTrue(any("DANGEROUS_TOOL" in m and "run_shell" in m for m in messages))
        self.assertTrue(any("TOOL_CALL" in m and "read_file" in m for m in messages))

    def test_dangerous_tool_set(self):
        self.assertIn("write_file", mlog.DANGEROUS_TOOLS)
        self.assertIn("run_shell", mlog.DANGEROUS_TOOLS)


if __name__ == "__main__":
    unittest.main()
