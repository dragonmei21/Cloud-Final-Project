# Forge Line Streamlit app container
FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files and source package for build
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

# Install dependencies (includes local package)
ENV UV_SYSTEM_PYTHON=1
RUN uv sync --frozen --no-dev

# Copy only required app code and assets
COPY app/ app/
COPY data/gold/ data/gold/

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
