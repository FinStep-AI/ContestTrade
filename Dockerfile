# 从 Docker Hub 拉取官方的 python 镜像，并指定标签为 3.13-slim 作为我们自定义镜像的基础。
FROM python:3.13-slim

# 设置容器内的环境变量。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录    
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 复制应用程序代码
COPY . .

# 创建日志目录
RUN mkdir -p /app/contest_trade/agents_workspace/logs

# 创建并切换非 root 用户
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

