"""Task / PlanTask 与计划解析 —— 多 Agent 编排基础

- parse_plan : LLM 计划 JSON 解析
- PlanTask   : 计划步骤建模（id / goal / 类型 / 依赖 / 工具）
- topo_sort  : Kahn 拓扑排序（分层批次，支持并行）
"""

from __future__ import annotations

import json
import re


def parse_plan(text: str) -> list[dict]:
    """解析 LLM 输出的计划 JSON 数组，兼容 ```json 代码块包裹和前后杂文本"""
    if not text:
        return []

    # 去掉 ```json ... ``` 代码块包裹
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)

    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []

    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
        return data
    return []


# ============================================================
# 计划任务建模 + DAG 拓扑排序
# ============================================================

class PlanTask:
    """计划步骤（Task 建模）：id + goal + 类型 + 依赖 + 工具

    step_type：
      reason - 需 LLM 推理/生成（走 ReAct）
      action - 确定性动作（直连工具，LLM 不参与）
      verify - 校验（执行命令检查，失败才升级 LLM）
    """

    def __init__(
        self,
        task_id: str,
        goal: str,
        step_type: str = "reason",
        detail: str = "",
        tool: str = "",
        args: dict | None = None,
        depends_on: list[str] | None = None,
    ):
        self.task_id = str(task_id)
        self.goal = goal
        self.step_type = step_type
        self.detail = detail
        self.tool = tool
        self.args = args or {}
        self.depends_on = [str(d) for d in (depends_on or [])]
        self.result = ""

    @classmethod
    def from_dict(cls, data: dict, default_depends_on: list[str] | None = None) -> "PlanTask":
        return cls(
            task_id=data.get("id") or data.get("step") or "step",
            goal=data.get("goal") or data.get("step") or "步骤",
            step_type=data.get("type", "reason"),
            detail=data.get("detail", ""),
            tool=data.get("tool", ""),
            args=data.get("args") or {},
            depends_on=data.get("depends_on") or default_depends_on,
        )


def topo_sort(tasks: list[PlanTask]) -> list[list[PlanTask]]:
    """Kahn 拓扑排序：返回分层批次（同批内可并行执行），检测循环依赖"""
    by_id = {t.task_id: t for t in tasks}
    for t in tasks:
        for dep in t.depends_on:
            if dep not in by_id:
                raise ValueError(f"计划依赖不存在：{dep}")

    indegree = {t.task_id: len(t.depends_on) for t in tasks}
    children: dict[str, list[str]] = {t.task_id: [] for t in tasks}
    for t in tasks:
        for dep in t.depends_on:
            children[dep].append(t.task_id)

    batches: list[list[PlanTask]] = []
    remaining = set(t.task_id for t in tasks)
    while remaining:
        ready = [tid for tid in remaining if indegree[tid] == 0]
        if not ready:
            raise ValueError("计划存在循环依赖")
        batches.append([by_id[tid] for tid in sorted(ready)])
        for tid in ready:
            remaining.remove(tid)
            for child in children[tid]:
                indegree[child] -= 1
    return batches
