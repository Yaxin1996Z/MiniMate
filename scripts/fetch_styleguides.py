"""下载权威编程规范到知识库（rag/repo/）

来源全部为开源权威规范：
  - Google Java / C++ / Python Style Guide
  - PEP 8（Python 官方）
  - C++ Core Guidelines（ISO C++ 委员会）
  - Linux Kernel Coding Style（C）

用法：python scripts/fetch_styleguides.py
"""

from __future__ import annotations

import os
import re
import sys
from html.parser import HTMLParser

import requests


SOURCES = [
    ("java_google_style", "https://google.github.io/styleguide/javaguide.html"),
    ("cpp_google_style", "https://google.github.io/styleguide/cppguide.html"),
    ("python_google_style", "https://google.github.io/styleguide/pyguide.html"),
    ("python_pep8", "https://peps.python.org/pep-0008/"),
    ("cpp_core_guidelines", "https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines"),
    ("c_kernel_style", "https://www.kernel.org/doc/html/latest/process/coding-style.html"),
]


class MarkdownConverter(HTMLParser):
    """HTML → Markdown 轻量转换（标题/段落/列表/代码块/表格）"""

    _SKIP_TAGS = {"script", "style", "nav", "footer", "aside"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._skip_depth = 0
        self._in_pre = False
        self._in_li = False
        self._in_code = False
        self._table_row: list[str] = []
        self._in_table = False
        self._pending_space = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._ensure_blank()
            level = int(tag[1])
            self.out.append("\n" + "#" * level + " ")
        elif tag == "p":
            self._ensure_blank()
        elif tag == "pre":
            self._ensure_blank()
            self.out.append("\n```\n")
            self._in_pre = True
        elif tag == "code" and not self._in_pre:
            self.out.append("`")
            self._in_code = True
        elif tag == "li":
            self.out.append("\n- ")
            self._in_li = True
        elif tag == "br":
            self.out.append("\n")
        elif tag == "table":
            self._ensure_blank()
            self._in_table = True
        elif tag == "tr":
            self._table_row = []
        elif tag in ("td", "th") and self._in_table:
            self._table_row.append("")
            self._pending_space = True

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li"):
            self.out.append("\n")
            self._in_li = False
        elif tag == "pre":
            self.out.append("\n```\n")
            self._in_pre = False
        elif tag == "code" and self._in_code:
            self.out.append("`")
            self._in_code = False
        elif tag == "tr" and self._in_table and self._table_row:
            self.out.append("\n| " + " | ".join(self._table_row) + " |\n")
            self._table_row = []
        elif tag == "table":
            self._in_table = False
            self._ensure_blank()

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_pre:
            self.out.append(data)
            return
        if self._in_table and self._table_row:
            self._table_row[-1] += data.strip()
            return
        text = data.strip()
        if not text:
            if self._pending_space:
                self.out.append(" ")
            return
        if self._pending_space:
            self.out.append(" ")
        self.out.append(text)
        self._pending_space = True

    def _ensure_blank(self):
        if self.out and self.out[-1].strip():
            self.out.append("\n\n")

    def render(self) -> str:
        text = "".join(self.out)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        return text.strip() + "\n"


def fetch(name: str, url: str) -> str:
    """下载并转换 HTML 规范为 Markdown"""
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    conv = MarkdownConverter()
    conv.feed(resp.text)
    md = conv.render()
    return f"# {name}\n\n> 来源：{url}\n\n" + md


def main() -> None:
    repo_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "minimate", "rag", "repo",
    )
    os.makedirs(repo_dir, exist_ok=True)
    for name, url in SOURCES:
        path = os.path.join(repo_dir, f"{name}.md")
        try:
            md = fetch(name, url)
            with open(path, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"OK {name}: {len(md)} chars -> {path}")
        except Exception as e:
            print(f"FAIL {name}: {e}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
