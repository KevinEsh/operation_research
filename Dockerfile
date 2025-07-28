# Then, use a final image without uv
FROM apache/airflow:3.0.3

# Cambia al usuario root para poder instalar paquetes
USER root

# Instala la librería que falta
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Vuelve al usuario airflow
USER airflow

COPY pyproject.toml uv.lock ./

# Install Python before the project for caching
# RUN uv python install 3.12
RUN uv pip install --no-cache -r pyproject.toml
# RUN uv sync --no-dev --compile-bytecode 

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"


