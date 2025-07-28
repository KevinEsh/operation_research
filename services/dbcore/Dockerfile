FROM python:3.10-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

ENV DATABASE_SERVER="postgresql"
ENV POSTGRES_HOST="database-server"
ENV POSTGRES_USER="admin"
ENV POSTGRES_PASSWORD="password"

COPY pyproject.toml uv.lock /app/
RUN uv sync --no-dev --compile-bytecode

COPY schema.py dbconfig.py main.py /app/

CMD ["uv", "run", "--no-dev", "fastapi", "run", "main.py", "--port", "80"]