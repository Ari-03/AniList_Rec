# The export container (SPEC §6, issue #20): the scoring service with a model
# bundle baked in. Build the bundle first, then the image:
#
#   uv run export-bundle --arch bm25 --out bundle
#   docker build -t anirec-scoring .
#   docker run --rm -p 8000:8000 anirec-scoring
#
# The bundle directory is the only model-specific input — shipping a different
# architecture (#21) rebuilds with a different bundle/, nothing else changes.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# dependency layer first so bundle/model swaps don't re-resolve the world
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra service

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra service

COPY bundle ./bundle
ENV ANIREC_BUNDLE_DIR=/app/bundle

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "--factory", "anilist_rec.service:create_app", \
     "--host", "0.0.0.0", "--port", "8000"]
