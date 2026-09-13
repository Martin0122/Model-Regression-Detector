FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY prompts ./prompts
COPY datasets ./datasets
RUN mkdir -p reports src/runs

# Required
ENV OPENAI_API_KEY=""
# Optional
ENV SLACK_WEBHOOK_URL=""
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
