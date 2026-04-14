FROM python:3.13-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY reference_times.json ./reference_times.json

# Install dependencies
RUN uv sync --frozen

# src/ contains app.py and diagnose.py — add to Python path
ENV PYTHONPATH=/app/src

EXPOSE 80

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]

