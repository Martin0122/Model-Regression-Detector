FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY prompts ./prompts
COPY datasets ./datasets
RUN mkdir -p reports src/runs

# Secrets are deliberately NOT declared with ENV - values set that way are baked into image
# metadata and readable via `docker history` by anyone who pulls the image. Pass them at run
# time instead:
#   -e OPENAI_API_KEY=...        (required)
#   -e SLACK_WEBHOOK_URL=...     (optional; alerts are skipped when unset)
# With OPENAI_API_KEY unset the pipeline fails fast with a clear message rather than silently.

# Non-secret tuning defaults, safe to bake in and override with -e.
ENV REGRESSION_WARNING_THRESHOLD=0.03
ENV REGRESSION_CRITICAL_THRESHOLD=0.08
ENV DRIFT_WINDOW_SIZE=7
ENV DRIFT_WARNING_THRESHOLD=0.03
ENV DRIFT_CRITICAL_THRESHOLD=0.08
ENV MIN_COMPLETION_RATE=0.95
ENV BLOCK_ON_CRITICAL_DRIFT=false

# src/runs and reports hold run history and generated reports - mount volumes over them
# (e.g. -v $(pwd)/src/runs:/app/src/runs -v $(pwd)/reports:/app/reports) so history
# persists across container invocations; without a volume every run starts from zero history.
CMD ["python", "-m", "src.pipeline"]
