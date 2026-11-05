FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY src ./src
WORKDIR /app/src

CMD ["/app/.venv/bin/marimo", "run", \
     "main.py", "data_nb.py", "factors_nb.py", "ols_nb.py", \
     "estimation_nb.py", "validation_nb.py", "hypotheses_nb.py", \
     "--headless", "--host", "0.0.0.0", "--port", "2718", \
     "--token-password-file", "/run/secrets/marimo_token"]