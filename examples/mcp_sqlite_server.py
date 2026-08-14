"""SQLite MCP Server（FastMCP）—— 提供创建、查询、删除 SQLite 表的工具

运行：uv run python examples/mcp_sqlite_server.py
通过 config.json 的 mcp.servers 配置后，MiniMate 启动时自动加载其工具。

工具列表：
- create_table: 在 SQLite 数据库中创建表
- query_table:  查询 SQLite 数据库中的表数据
- drop_table:   删除 SQLite 数据库中的表
"""

import os
import sqlite3

from fastmcp import FastMCP

# 默认数据库文件路径（位于 examples 目录下）
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sqlite_mcp.db")

mcp = FastMCP("mini-sqlite")


def _get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """获取 SQLite 数据库连接（自动创建数据库文件）"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


@mcp.tool()
def create_table(
    table_name: str,
    columns: str,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """在 SQLite 数据库中创建一张表。

    参数：
        table_name: 要创建的表名
        columns: 列定义，例如 "id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER"
        db_path: 数据库文件路径（默认使用 examples/sqlite_mcp.db）
    """
    if not table_name or not table_name.strip():
        return "错误：表名不能为空"
    if not columns or not columns.strip():
        return "错误：列定义不能为空"

    sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns})'
    try:
        conn = _get_connection(db_path)
        try:
            conn.execute(sql)
            conn.commit()
        finally:
            conn.close()
        return f"✅ 表 '{table_name}' 创建成功。"
    except sqlite3.Error as e:
        return f"❌ 创建表失败：{e}"


@mcp.tool()
def query_table(
    table_name: str,
    columns: str = "*",
    where: str = "",
    limit: int = 100,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """查询 SQLite 数据库中的表数据。

    参数：
        table_name: 要查询的表名
        columns: 要查询的列，逗号分隔，默认 "*"
        where: 可选的 WHERE 条件，例如 "age > 18"
        limit: 返回的最大行数，默认 100
        db_path: 数据库文件路径（默认使用 examples/sqlite_mcp.db）
    """
    if not table_name or not table_name.strip():
        return "错误：表名不能为空"

    sql = f'SELECT {columns} FROM "{table_name}"'
    if where and where.strip():
        sql += f" WHERE {where}"
    sql += f" LIMIT {int(limit)}"

    try:
        conn = _get_connection(db_path)
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description] if cursor.description else []
        finally:
            conn.close()

        if not rows:
            return f"查询完成：表 '{table_name}' 中没有匹配的数据。"

        # 格式化输出结果
        lines = []
        lines.append(" | ".join(col_names))
        lines.append("-" * len(lines[0]))
        for row in rows:
            lines.append(" | ".join(str(row[col]) for col in col_names))
        return f"共 {len(rows)} 行结果：\n" + "\n".join(lines)
    except sqlite3.Error as e:
        return f"❌ 查询失败：{e}"


@mcp.tool()
def drop_table(
    table_name: str,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """删除 SQLite 数据库中的一张表。

    参数：
        table_name: 要删除的表名
        db_path: 数据库文件路径（默认使用 examples/sqlite_mcp.db）
    """
    if not table_name or not table_name.strip():
        return "错误：表名不能为空"

    sql = f'DROP TABLE IF EXISTS "{table_name}"'
    try:
        conn = _get_connection(db_path)
        try:
            conn.execute(sql)
            conn.commit()
        finally:
            conn.close()
        return f"✅ 表 '{table_name}' 删除成功。"
    except sqlite3.Error as e:
        return f"❌ 删除表失败：{e}"


if __name__ == "__main__":
    mcp.run()
