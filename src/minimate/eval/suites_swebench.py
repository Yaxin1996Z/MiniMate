"""SWE-bench Multilingual 评测 —— 真实 GitHub Issue 修复基准

判定协议与官方 SWE-bench 一致：
  1. 克隆仓库并 checkout 到 issue 对应的 base_commit
  2. Agent 阅读 issue 描述（problem_statement），修改代码
  3. 应用官方 test_patch（PR 新增/修改的测试文件）
  4. 运行 FAIL_TO_PASS 测试：全部通过才判定修复成功
  5. 可选运行 PASS_TO_PASS：验证既有功能不回归

说明：SWE-bench Multilingual 是 9 种非 Python 语言（Java / Go / Rust / JS/TS
/ Ruby / PHP / C / C++），因此判定命令按语言适配（Java 用 Maven，JS/TS 用 npm
等），不依赖 pytest。
"""

from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

from .. import llm
from ..agent import Agent
from ..tools import ToolExecutor, register_all_tools


SWEBENCH_DATA = os.path.join(
    os.path.expanduser("~"), ".minimate", "eval", "swebench_multilingual.jsonl"
)


def _long_path(path: str) -> str:
    """Windows 下把 8.3 短路径转成长路径（Maven -pl 解析需要一致路径）"""
    if os.name != "nt":
        return path
    import ctypes

    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetLongPathNameW(str(path), buf, len(buf))
    if n and n < len(buf):
        return buf.value
    return path


# ============================================================
# 数据模型
# ============================================================

@dataclass
class SWEBenchCase:
    """一条真实 GitHub Issue 评测实例"""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str                       # gold patch（仅用于判定器自检）
    test_patch: str
    fail_to_pass: list[str]          # 修复后必须通过的测试
    pass_to_pass: list[str]          # 不能回归的既有测试
    eval_type: str = ""

    @classmethod
    def from_row(cls, raw: dict) -> "SWEBenchCase":
        return cls(
            instance_id=raw["instance_id"],
            repo=raw["repo"],
            base_commit=raw["base_commit"],
            problem_statement=raw.get("problem_statement") or "",
            patch=raw.get("patch") or "",
            test_patch=raw.get("test_patch") or "",
            fail_to_pass=list(raw.get("FAIL_TO_PASS") or []),
            pass_to_pass=list(raw.get("PASS_TO_PASS") or []),
            eval_type=raw.get("eval_type") or "",
        )


@dataclass
class SWEBenchResult:
    """单条实例评测结果"""

    case: SWEBenchCase
    passed: bool
    output: str = ""
    trace: str = ""
    reason: str = ""
    duration: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def load_swebench(path: str = SWEBENCH_DATA) -> list[SWEBenchCase]:
    """从本地缓存 JSONL 加载全部实例"""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"SWE-bench Multilingual 数据未找到：{path}\n"
            "请先下载：minimate --swebench-download"
        )
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(SWEBenchCase.from_row(json.loads(line)))
    return cases


def download_swebench_data(path: str = SWEBENCH_DATA) -> int:
    """从 HuggingFace datasets-server 下载 300 条实例清单到本地缓存"""
    import requests

    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows: list[dict] = []
    for offset in (0, 100, 200):
        resp = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": "SWE-bench/SWE-bench_Multilingual",
                "config": "default",
                "split": "test",
                "offset": offset,
                "length": 100,
            },
            timeout=120,
        )
        resp.raise_for_status()
        rows.extend(row["row"] for row in resp.json()["rows"])
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


# ============================================================
# 语言适配器（clone + 测试执行）
# ============================================================

class JavaMavenAdapter:
    """Java + Maven 仓库适配器（gson 等纯 Java 仓库）"""

    def __init__(self, case: SWEBenchCase, maven_test_timeout: int = 900):
        self.case = case
        self.maven_test_timeout = maven_test_timeout

    def setup(self, sandbox: str) -> str:
        """准备仓库：优先用本地缓存（~/.minimate/eval/repos），避免网络不稳定"""
        cache_repo = self._ensure_cache()
        repo_dir = os.path.join(sandbox, "repo")
        # 从缓存本地克隆（无需网络），再 checkout 到目标 commit
        subprocess.run(
            ["git", "clone", "--quiet", "--local", cache_repo, repo_dir],
            check=True,
            timeout=300,
        )
        subprocess.run(
            ["git", "-C", repo_dir, "checkout", "--quiet", self.case.base_commit],
            check=True,
            timeout=120,
        )
        return repo_dir

    def _ensure_cache(self) -> str:
        """确保仓库缓存存在：缺失时从 GitHub 克隆（带重试）"""
        cache_root = os.path.join(
            os.path.expanduser("~"), ".minimate", "eval", "repos"
        )
        os.makedirs(cache_root, exist_ok=True)
        cache_repo = os.path.join(cache_root, self.case.repo.replace("/", "__"))
        if os.path.isdir(cache_repo):
            return cache_repo

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                subprocess.run(
                    [
                        "git", "clone", "--quiet",
                        f"https://github.com/{self.case.repo}.git",
                        cache_repo,
                    ],
                    check=True,
                    timeout=600,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                shutil.rmtree(cache_repo, ignore_errors=True)
                time.sleep(3)
        if last_err is not None:
            raise last_err
        return cache_repo

    def apply_test_patch(self, repo_dir: str) -> None:
        """应用官方 test_patch（新增/修改测试文件）"""
        patch_file = os.path.join(repo_dir, "_test.patch")
        with open(patch_file, "w", encoding="utf-8") as f:
            f.write(self.case.test_patch)
        subprocess.run(
            ["git", "-C", repo_dir, "apply", "--whitespace=fix", "_test.patch"],
            check=True,
            timeout=60,
        )
        self._patch_build(repo_dir)

    def _patch_build(self, repo_dir: str) -> None:
        """构建环境适配：移除老仓库与新版 JDK 不兼容的 OSGi 打包插件（bnd）

        bnd-maven-plugin 3.x 依赖已移除的 JDK 内部 API，在 JDK 17+ 上会
        报 "Null query"；OSGi bundle 打包与单元测试无关，直接移除。
        """
        # 删除 module-info.java：老测试代码可能引用未声明的模块（如 java.awt），
        # 回退 classpath 编译模式避免 JPMS 隔离报错（单元测试判定不需要模块化）
        for mod_info in glob.glob(
            os.path.join(repo_dir, "**", "module-info.java"), recursive=True
        ):
            try:
                os.remove(mod_info)
            except OSError:
                pass
        for pom in glob.glob(os.path.join(repo_dir, "**", "pom.xml"), recursive=True):
            try:
                with open(pom, encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                continue
            # 移除 javac -Werror：老代码在 JDK 24 上常有 deprecation 等警告，
            # -Werror 会把警告误判为编译失败（与测试判定无关）
            modified = re.sub(r"<arg>\s*-Werror\s*</arg>", "", text)
            modified = re.sub(
                r"<compilerArgument>\s*-Werror\s*</compilerArgument>", "", modified
            )
            # maven-compiler-plugin 的 failOnWarning 同样会把警告当失败，关闭
            modified = re.sub(
                r"<failOnWarning>\s*true\s*</failOnWarning>",
                "<failOnWarning>false</failOnWarning>",
                modified,
            )
            modified = re.sub(
                r"<plugin>\s*<groupId>biz\.aQute\.bnd</groupId>.*?</plugin>",
                "",
                modified,
                flags=re.S,
            )
            if modified != text:
                with open(pom, "w", encoding="utf-8") as f:
                    f.write(modified)

    def run_tests(self, repo_dir: str, tests: list[str]) -> tuple[bool, str]:
        """运行指定测试：从 test_patch 动态推断测试模块，mvn -pl <模块> -am test"""
        if not tests:
            return True, "无测试需要运行"
        classes: dict[str, list[str]] = {}
        for t in tests:
            cls, sep, method = t.rpartition("#")
            if sep:
                classes.setdefault(cls, []).append(method)
            else:
                # 纯类名（F2P 只给类名时，如 NodeTest）
                classes.setdefault(t, [])
        # 纯类名（无方法）不加尾 #，避免 surefire 匹配失败
        spec = ",".join(
            f"{c}#{'+'.join(ms)}" if ms else c for c, ms in classes.items()
        )
        module = self._test_module()
        # 用 java 直接运行 Maven launcher，规避 Windows 下 .CMD 批处理参数传递问题
        java, maven_home = self._maven_java()
        boot = sorted(glob.glob(os.path.join(maven_home, "boot", "plexus-classworlds-*.jar")))
        if not boot:
            return False, f"未找到 plexus-classworlds.jar（Maven home: {maven_home}）"
        cmd = [
            java,
            # Windows 上 JVM 默认 file.encoding=GBK，导致内嵌 javac 按 GBK 读
            # UTF-8 源码产生编码警告（-Werror 误判失败），强制 UTF-8
            "-Dfile.encoding=UTF-8",
            "-classpath", boot[-1],
            f"-Dclassworlds.conf={os.path.join(maven_home, 'bin', 'm2.conf')}",
            f"-Dmaven.home={maven_home}",
            f"-Dmaven.multiModuleProjectDirectory={repo_dir}",
            "org.codehaus.plexus.classworlds.launcher.Launcher",
            "-q", "-pl", module, "-am",
            "test",
            f"-Dtest={spec}",
            "-DfailIfNoTests=false",
            # 多模块仓库：-am 构建的依赖模块可能没有指定测试，忽略而非报错
            "-Dsurefire.failIfNoSpecifiedTests=false",
            # 老版本 JaCoCo 解析不了新版 JDK 编译的 class，覆盖率与判定无关，跳过
            "-Djacoco.skip=true",
            # 老版本 ProGuard 不支持新版 JDK 的 class 版本，混淆与测试判定无关，跳过
            "-Dproguard.skip=true",
            # patch 应用后偶发编码警告，-Werror 会误判失败，关闭 warning 即失败
            "-Dmaven.compiler.failOnWarning=false",
            # Windows 上 javac 默认按平台编码（GBK）读 UTF-8 源码会报编码警告，
            # 显式指定 UTF-8，避免 -Werror 误判
            "-Dmaven.compiler.encoding=UTF-8",
            # release=11 同时约束语法与 API 版本，避免老代码与 JDK 21+ 的
            # SequencedCollection（List.addFirst/getFirst 等）方法冲突
            "-Dmaven.compiler.release=11",
            "-Djava.version=11",
            # 部分仓库的编译属性用驼峰 javaVersion（如 gson 2.9.1-SNAPSHOT）
            "-DjavaVersion=11",
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=self.maven_test_timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return False, f"测试超时（>{self.maven_test_timeout}s）"
        tail = self._extract_maven_result(proc.stdout or "") + "\n" + (proc.stderr or "")[-1000:]
        return proc.returncode == 0, f"退出码 {proc.returncode}\n{tail}"

    def _test_module(self) -> str:
        """从 test_patch 的 diff 路径推断测试所在的 Maven 模块（如 gson / javaparser-core-testing）"""
        m = re.search(r"diff --git a/([^/]+)/src/test", self.case.test_patch or "")
        if m:
            return m.group(1)
        # 兜底：gson 等单测试模块仓库
        m2 = re.search(r"diff --git a/([^/]+)/test", self.case.test_patch or "")
        return m2.group(1) if m2 else "."

    @staticmethod
    def _extract_maven_result(out: str) -> str:
        """提取 Maven 输出中的测试统计与失败详情（避免长输出淹没关键信息）"""
        lines = out.splitlines()
        keep: list[str] = []
        keywords = (
            "Tests run:", "Failures:", "Errors:", "Skipped:",
            "[ERROR]", "[WARNING]",
            "[ERROR] test", "FAILURE!", "BUILD SUCCESS", "BUILD FAILURE",
            "COMPILATION ERROR", "cannot find symbol", "expected:",
            "but was:", "No tests matching",
        )
        for line in lines:
            if any(k in line for k in keywords):
                keep.append(line)
        if not keep:
            return out[-1500:]
        return "\n".join(keep[-40:])

    @staticmethod
    def _maven_java() -> tuple[str, str]:
        """返回 (java 可执行路径, Maven home)，从 mvn 命令路径推导"""
        mvn = shutil.which("mvn")
        if not mvn:
            raise RuntimeError("未找到 mvn，请安装 Maven 并加入 PATH")
        maven_home = os.path.dirname(os.path.dirname(os.path.abspath(mvn)))
        java = shutil.which("java") or "java"
        return java, maven_home

    def apply_gold_patch(self, repo_dir: str) -> None:
        """应用 gold patch（判定器自检：验证 harness 能判定正确修复）"""
        patch_file = os.path.join(repo_dir, "_gold.patch")
        with open(patch_file, "w", encoding="utf-8") as f:
            f.write(self.case.patch)
        subprocess.run(
            ["git", "-C", repo_dir, "apply", "--whitespace=fix", "_gold.patch"],
            check=True,
            timeout=60,
        )


def make_adapter(case: SWEBenchCase) -> JavaMavenAdapter:
    """按仓库/语言选择适配器（当前支持 Java + Maven，可扩展）"""
    return JavaMavenAdapter(case)


# ============================================================
# 运行器
# ============================================================

class SWEBenchRunner:
    """SWE-bench Multilingual 运行器

    流程：clone → checkout base_commit → Agent 修复 → 应用 test_patch →
    运行 FAIL_TO_PASS 测试 → 判定。
    """

    def __init__(
        self,
        agent_mode: str = "react",
        max_steps: int = 8,
        results_dir: str = "",
        gold_check: bool = False,
    ):
        self.agent_mode = agent_mode
        self.max_steps = max_steps
        self.results_dir = os.path.abspath(
            results_dir
            or os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "eval", "results"
            )
        )
        # gold_check=True 时不调用 LLM，直接应用 gold patch 自检判定器
        self.gold_check = gold_check
        self._tools = ToolExecutor()
        register_all_tools(self._tools)

    def run_suite(
        self,
        case_ids: Optional[list[str]] = None,
        max_cases: Optional[int] = None,
    ) -> tuple[list[SWEBenchResult], dict, dict]:
        cases = load_swebench()
        if case_ids:
            cases = [c for c in cases if c.instance_id in case_ids]
        if max_cases is not None and max_cases > 0:
            cases = cases[:max_cases]

        results = [self.run_case(case) for case in cases]
        summary = {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "total_duration": round(sum(r.duration for r in results), 2),
            "total_tokens": sum(r.total_tokens for r in results),
        }
        paths = self._save_report(results, summary)
        return results, summary, paths

    def run_case(self, case: SWEBenchCase) -> SWEBenchResult:
        sandbox = _long_path(tempfile.mkdtemp(prefix="swebench_"))
        old_cwd = os.getcwd()
        try:
            adapter = make_adapter(case)
            repo_dir = adapter.setup(sandbox)

            if self.gold_check:
                # 自检模式：直接应用 gold patch，验证判定器能识别正确修复
                adapter.apply_gold_patch(repo_dir)
                output = "[gold_check] 已应用官方修复补丁"
                trace = output
                prompt_tokens = completion_tokens = 0
            else:
                # 真实模式：Agent 根据 issue 描述修复代码
                os.chdir(repo_dir)
                start = time.time()
                stats_before = llm.get_stats()
                output, trace = self._agent_fix(case)
                stats_after = llm.get_stats()
                self.duration_agent = time.time() - start
                prompt_tokens = max(
                    0, stats_after["prompt_tokens"] - stats_before["prompt_tokens"]
                )
                completion_tokens = max(
                    0, stats_after["completion_tokens"] - stats_before["completion_tokens"]
                )

            # 应用官方测试补丁并运行 FAIL_TO_PASS
            adapter.apply_test_patch(repo_dir)
            os.chdir(repo_dir)
            test_start = time.time()
            passed, test_detail = adapter.run_tests(repo_dir, case.fail_to_pass)
            test_duration = time.time() - test_start

            reason = (
                f"FAIL_TO_PASS {len(case.fail_to_pass)} 个测试全部通过（{test_duration:.0f}s）"
                if passed
                else f"FAIL_TO_PASS 测试未全部通过：{test_detail[-600:]}"
            )
            return SWEBenchResult(
                case=case,
                passed=passed,
                output=(output or "")[:800],
                trace=trace,
                reason=reason,
                duration=test_duration + getattr(self, "duration_agent", 0.0),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as e:
            return SWEBenchResult(
                case=case,
                passed=False,
                error=str(e),
                reason=f"执行异常：{e}",
            )
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(sandbox, ignore_errors=True)

    def _agent_fix(self, case: SWEBenchCase) -> tuple[str, str]:
        """Agent 根据 issue 描述修复代码"""
        prompt = (
            f"这是一个真实 GitHub Issue，你需要修复仓库中的代码。\n\n"
            f"【Issue 描述】\n{case.problem_statement}\n\n"
            "【要求】\n"
            "- 当前目录就是仓库根目录，请先阅读相关源码定位 bug\n"
            "- 只修改必要的源码文件，不要修改测试文件\n"
            "- 不要运行测试命令（评测系统会自行验证）\n"
            "- 修改完成后，用 run_shell 执行 git diff --stat 确认改动，然后给出修复说明"
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            agent = Agent(
                role="代码修复专家",
                goal="定位并修复 GitHub Issue 描述的代码缺陷",
                tools=self._tools,
                max_steps=self.max_steps,
            )
            output = agent.run(prompt, mode=self.agent_mode)
        return (output or "").strip(), buf.getvalue()

    def _save_report(
        self, results: list[SWEBenchResult], summary: dict
    ) -> dict:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self.results_dir, f"swebench_{stamp}")
        os.makedirs(out_dir, exist_ok=True)
        md_path = os.path.join(out_dir, "report.md")
        json_path = os.path.join(out_dir, "report.json")

        lines = [
            "# SWE-bench Multilingual 评测报告",
            "",
            f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- LLM 模型：{llm.get_model()}",
            "",
            "## 汇总",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
            f"| 实例总数 | {summary['total']} |",
            f"| 修复成功 | {summary['passed']} |",
            f"| 修复成功率 | {summary['passed'] / summary['total'] * 100 if summary['total'] else 0:.1f}% |",
            f"| 总耗时 | {summary['total_duration']:.0f}s |",
            f"| Token 总消耗 | {summary['total_tokens']} |",
            "",
            "## 逐条明细",
            "",
        ]
        for r in results:
            mark = "✅ 通过" if r.passed else "❌ 失败"
            lines += [
                f"### {r.case.instance_id}（{r.case.repo}）{mark}",
                "",
                f"- **Issue**：{(r.case.problem_statement or '')[:200]}",
                f"- **判定**：{r.reason or r.error}",
                f"- **耗时**：{r.duration:.0f}s ｜ **Token**：{r.total_tokens}",
            ]
            if r.output:
                lines += ["", "Agent 输出摘要：", "", "```text", r.output[:500], "```"]
            lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        payload = {
            "summary": summary,
            "cases": [
                {
                    "instance_id": r.case.instance_id,
                    "repo": r.case.repo,
                    "passed": r.passed,
                    "reason": r.reason,
                    "error": r.error,
                    "duration": round(r.duration, 2),
                    "total_tokens": r.total_tokens,
                    "output": (r.output or "")[:300],
                }
                for r in results
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return {"markdown": md_path, "json": json_path}
