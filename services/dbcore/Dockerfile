FROM python:3.10-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock /app/
RUN uv sync --no-dev --compile-bytecode

COPY schema.py dbconfig.py main.py /app/

CMD ["uv", "run", "--no-dev", "fastapi", "run", "main.py", "--port", "80"]