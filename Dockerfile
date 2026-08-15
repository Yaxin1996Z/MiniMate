# ============================================================
# MiniMate — 工作/代码助手 Agent Docker 镜像
# 单阶段构建，国内镜像加速
# ============================================================

FROM python:3.12-slim

WORKDIR /app

# 设置 pip 国内镜像（加速依赖下载）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com

# 先拷贝构建所需文件（pyproject + README + src），再安装，
# 确保 minimate 包与 cli/api entry point 正确装进 site-packages
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY run.py .
RUN pip install --no-cache-dir .

# 创建输出目录（知识库/模型目录由 compose 挂载，见 docker-compose.yml）
RUN mkdir -p /app/output

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import minimate; print('ok')" || exit 1

# 默认启动：FastAPI 服务
CMD ["minimate-server"]
