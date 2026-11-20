# SEM Bidding Simulator: Streamlit dashboard image.
#   docker build -t sem-bidding-simulator .
#   docker run --rm -p 127.0.0.1:8501:8501 sem-bidding-simulator
# The synthetic DuckDB database is generated at build time, so the first visit is instant.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Runtime dependencies only (pytest is for development).
COPY requirements.txt .
RUN grep -v '^pytest' requirements.txt > /tmp/requirements.runtime.txt \
 && pip install -r /tmp/requirements.runtime.txt \
 && rm /tmp/requirements.runtime.txt

RUN useradd --create-home --uid 10001 app \
 && mkdir -p /app/data && chown app:app /app/data

COPY --chown=app:app app.py ./
COPY --chown=app:app src ./src
COPY --chown=app:app sql ./sql

USER app
RUN python -c "from src import data_generator as dg; dg.write_duckdb(dg.generate(seed=42), 'data/sem.duckdb')"

EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=6 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).status == 200 else 1)"

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--server.fileWatcherType=none", "--browser.gatherUsageStats=false", \
     "--client.toolbarMode=viewer"]
