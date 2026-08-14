"""评测用例数据模型"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalCase:
    """一条评测用例

    checker 与 expected 组合成确定性判定：
      file_exists     expected=沙箱内相对路径
      file_contains   expected="路径|关键词"
      dir_contains    expected="目录|文件名"
      output_contains expected=最终输出必须包含的关键词
      math_answer     expected=期望数值（容差 0.01）
      command_exit0   expected=需在沙箱内执行且退出码为 0 的命令
      grep_finds      expected="目录|关键词"（目录内文件内容命中关键词）
    """

    id: str
    name: str
    mode: str                       # chat / react / plan / multi
    prompt: str
    checker: str
    expected: str
    setup: Optional[list[dict | str]] = None  # 准备操作：{"write": {path, content}} /
    #                                        # {"mkdir": path} 或 shell 命令字符串（沙箱内执行）
    timeout: int = 240
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalCaseResult:
    """单条用例执行结果"""

    case: EvalCase
    passed: bool
    output: str = ""                # Agent 最终输出（截断）
    trace: str = ""                 # 完整交互过程（分节文本）
    checker_reason: str = ""        # 判定依据
    duration: float = 0.0           # 秒
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""                 # 执行异常信息

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class EvalSummary:
    """评测汇总统计"""

    total: int = 0
    passed: int = 0
    by_mode: dict[str, dict] = field(default_factory=dict)
    total_duration: float = 0.0
    total_tokens: int = 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.total if self.total else 0.0

    @property
    def avg_duration(self) -> float:
        return self.total_duration / self.total if self.total else 0.0
