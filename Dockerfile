FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY atlas atlas
COPY web web
ENV ATLAS_DATA=/data FASTEMBED_CACHE_PATH=/data/models
VOLUME /data
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "atlas.server:app", "--host", "0.0.0.0", "--port", "8000"]
