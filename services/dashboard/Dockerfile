FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml uv.lock /app/
RUN uv sync --no-dev --compile-bytecode

COPY /pages /pages
COPY main.py main.py

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]