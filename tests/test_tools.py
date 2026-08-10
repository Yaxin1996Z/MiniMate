"""文件工具单元测试：read_file / list_files"""

import os
import tempfile
import unittest

from minimate.tools import (
    Tool,
    ToolExecutor,
    find_files,
    grep_files,
    list_files,
    read_file,
    run_shell,
    write_file,
)


class FileToolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.fp = os.path.join(self.dir, "hello.txt")
        with open(self.fp, "w", encoding="utf-8") as f:
            f.write("你好，MiniMate")
        os.makedirs(os.path.join(self.dir, "sub"), exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_file_ok(self):
        result = read_file.run_text(self.fp)
        self.assertIn("你好，MiniMate", result)

    def test_read_file_not_found(self):
        result = read_file.run_text(os.path.join(self.dir, "nope.txt"))
        self.assertIn("文件不存在", result)

    def test_read_file_is_dir(self):
        result = read_file.run_text(self.dir)
        self.assertIn("是目录", result)

    def test_read_file_empty_path(self):
        self.assertIn("请提供文件路径", read_file.run_text(""))

    def test_read_file_trailing_text(self):
        """模型在参数后追加散文（如'请稍候...'）时，路径只取第一行"""
        result = read_file.run_text(f"{self.fp}\n请稍候，正在读取文件内容...")
        self.assertIn("你好，MiniMate", result)

    def test_list_files_ok(self):
        result = list_files.run_text(self.dir)
        self.assertIn("hello.txt", result)
        self.assertIn("[目录] sub", result)

    def test_list_files_not_found(self):
        result = list_files.run_text(os.path.join(self.dir, "nope"))
        self.assertIn("目录不存在", result)

    def test_list_files_is_file(self):
        result = list_files.run_text(self.fp)
        self.assertIn("不是目录", result)

    def test_write_file_ok(self):
        target = os.path.join(self.dir, "new", "note.txt")
        result = write_file.run_text(f"{target} | 测试内容")
        self.assertIn("文件已写入", result)
        with open(target, encoding="utf-8") as f:
            self.assertEqual(f.read(), "测试内容")

    def test_write_file_overwrites(self):
        result = write_file.run_text(f"{self.fp} | 新内容")
        self.assertIn("文件已写入", result)
        with open(self.fp, encoding="utf-8") as f:
            self.assertEqual(f.read(), "新内容")

    def test_write_file_bad_format(self):
        result = write_file.run_text("只有路径没有内容")
        self.assertIn("需要 2 个参数", result)

    def test_write_file_empty_path(self):
        result = write_file.run_text(" | 内容")
        self.assertIn("请提供文件路径", result)

    def test_run_shell_echo(self):
        result = run_shell.run_text("echo hello")
        self.assertIn("退出码：0", result)
        self.assertIn("hello", result)

    def test_run_shell_error_code(self):
        result = run_shell.run_text("exit 3")
        self.assertIn("退出码：3", result)

    def test_run_shell_empty(self):
        self.assertIn("请提供", run_shell.run_text(""))

    def test_run_shell_failure_hints_system(self):
        """命令失败时返回系统适配提示"""
        result = run_shell.run_text("this_command_does_not_exist_xyz")
        self.assertIn("退出码", result)
        self.assertIn("提示", result)

    def test_tools_prompt_injects_system_hint(self):
        """run_shell 工具描述包含当前系统类型提示"""
        tools = ToolExecutor()
        tools.register(Tool(name="run_shell", description="执行命令", func=lambda a: ""))
        prompt = tools.get_tools_prompt()
        if os.name == "nt":
            self.assertIn("Windows", prompt)
        else:
            self.assertIn("Linux", prompt)

    def test_find_files_by_pattern(self):
        result = find_files.run_text(f"{self.dir} | *.txt")
        self.assertIn("hello.txt", result)

    def test_find_files_nested(self):
        nested = os.path.join(self.dir, "sub", "app.py")
        with open(nested, "w", encoding="utf-8") as f:
            f.write("print('hi')")
        result = find_files.run_text(f"{self.dir} | *.py")
        self.assertIn("sub", result)
        self.assertIn("app.py", result)

    def test_find_files_no_match(self):
        result = find_files.run_text(f"{self.dir} | *.xyz")
        self.assertIn("未找到", result)

    def test_find_files_bad_format(self):
        self.assertIn("需要 2 个参数", find_files.run_text("只给目录"))

    def test_grep_files_hit(self):
        result = grep_files.run_text(f"{self.dir} | 你好")
        self.assertIn("hello.txt:1", result)
        self.assertIn("你好", result)

    def test_grep_files_no_hit(self):
        result = grep_files.run_text(f"{self.dir} | 不存在的词")
        self.assertIn("未找到", result)

    def test_grep_files_bad_format(self):
        self.assertIn("需要 2 个参数", grep_files.run_text(""))


class SchemaTest(unittest.TestCase):
    """工具 Schema 推断与 Function Calling 执行"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_file_schema(self):
        schema = read_file.schema
        fn = schema["function"]
        self.assertEqual(fn["name"], "read_file")
        self.assertEqual(fn["parameters"]["required"], ["path"])
        self.assertEqual(fn["parameters"]["properties"]["path"]["type"], "string")

    def test_write_file_schema(self):
        schema = write_file.schema
        params = schema["function"]["parameters"]
        self.assertEqual(sorted(params["required"]), ["content", "path"])
        self.assertEqual(params["properties"]["path"]["type"], "string")

    def test_executor_execute_kwargs(self):
        """Function Calling 通道：按命名参数执行"""
        tools = ToolExecutor()
        tools.register(write_file)
        target = os.path.join(self.tmp.name, "fc.txt")
        result = tools.execute("write_file", path=target, content="FC 内容")
        self.assertIn("文件已写入", result)
        with open(target, encoding="utf-8") as f:
            self.assertEqual(f.read(), "FC 内容")

    def test_executor_schemas(self):
        tools = ToolExecutor()
        tools.register(read_file)
        tools.register(run_shell)
        self.assertEqual(len(tools.schemas), 2)
        names = {s["function"]["name"] for s in tools.schemas}
        self.assertEqual(names, {"read_file", "run_shell"})


if __name__ == "__main__":
    unittest.main()
