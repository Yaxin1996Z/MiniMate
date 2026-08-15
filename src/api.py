"""
MiniMate FastAPI 服务 —— 工作/代码助手 REST API

支持三种 Agent 模式：chat（纯问答）/ react（ReAct 循环）/ plan（Plan & Execute）
"""

import sys
from typing import Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

from minimate import __version__
from cli import run_query


app = FastAPI(
    title="MiniMate API",
    description="工作/代码助手 Agent 服务 - 支持 chat / react / plan / multi 四种模式",
    version=__version__,
)


# ============================================================
# Pydantic 模型
# ============================================================

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="任务/问题")
    mode: str = Field("react", description="执行模式：chat / react / plan / multi")
    kb_path: Optional[str] = Field("", description="知识库文档目录路径（可选）")
    max_steps: int = Field(8, ge=1, le=30, description="ReAct 最大循环步数")


# ============================================================
# 健康检查
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "MiniMate",
        "version": __version__,
    }


# ============================================================
# 执行任务
# ============================================================

@app.post("/api/chat")
def chat(req: ChatRequest):
    """以指定模式执行任务，返回最终答案"""
    try:
        answer = run_query(
            req.question,
            mode=req.mode,
            kb_path=req.kb_path,
            max_steps=req.max_steps,
        )
        return {
            "status": "ok",
            "question": req.question,
            "mode": req.mode,
            "answer": answer,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"执行出错：{str(e)}")


# ============================================================
# 服务启动入口
# ============================================================

def main():
    """启动 API 服务：python src/api.py [--port 8080] / minimate-server"""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="MiniMate API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    print(f"  MiniMate API 启动")
    print(f"  ===========================")
    print(f"  地址: http://localhost:{args.port}")
    print(f"  文档: http://localhost:{args.port}/docs")
    print(f"  健康: http://localhost:{args.port}/health")
    print()
    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
