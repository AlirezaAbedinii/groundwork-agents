# RAG hybrid-search service images.
#
# Target `api` serves the API; it is the last stage, so a plain `docker build`
# builds it. Hybrid retrieval's cross-encoder reranker needs sentence-transformers
# (rerank extra); it runs on the CPU, so torch is the CPU-only build, and the
# libraries that decide its scores are pinned in docker/constraints.txt. The model
# is downloaded at build time, so a container never fetches it. Embeddings and
# generation stay API-based.
#
# Target `ui` is the Streamlit UI alone: it talks to the API over HTTP only.

FROM python:3.12-slim AS ui

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# The `ui` extra from pyproject.toml, without the service and its stack.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['optional-dependencies']['ui']))" \
        > /tmp/ui-requirements.txt \
    && pip install -r /tmp/ui-requirements.txt
COPY ui ./ui

RUN useradd --uid 10001 --create-home rag
USER 10001

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]


FROM python:3.12-slim AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# torch first, from the PyTorch CPU index, so nothing installs the CUDA build. Its
# C++ headers (63 MB) only serve compiling extensions, which nothing here does.
COPY docker/constraints.txt /tmp/constraints.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu -c /tmp/constraints.txt torch \
    && rm -rf /usr/local/lib/python3.12/site-packages/torch/include

# Dependencies before the rest of the project, so code edits don't bust this layer.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install -c /tmp/constraints.txt -e ".[api,ingestion,indexing,llm,rerank,telemetry]"

# The reranker the settings name, into HF_HOME.
RUN python -c "from sentence_transformers import CrossEncoder; \
from rag.config import Settings; CrossEncoder(Settings(_env_file=None).reranker_model)"

# Then the rest of the project (scripts, ui, sample corpus).
COPY . .

# Non-root. The user owns data/ (the trace store) and HF_HOME (models a keyless
# setup downloads at run time, such as the MiniLM embeddings).
RUN useradd --uid 10001 --create-home rag && mkdir -p data && chown -R rag /app/data /opt/hf
USER 10001

EXPOSE 8000

CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
