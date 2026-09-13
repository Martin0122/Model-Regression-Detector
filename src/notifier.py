import json
import os
import urllib.request
from .config import ComparisonResult, DriftResult

STATUS_EMOJI = {"pass": ":white_check_mark:", "warning": ":warning:", "critical": ":rotating_light:"}


def _post_to_slack(payload: dict, webhook_url: str | None) -> bool:
    """Returns True if an alert was sent, False if skipped or if delivery failed.
    Alerting is a side effect of a pipeline run that has already succeeded and been saved -
    a bad webhook URL, a Slack outage, or a slow endpoint must never crash the run or block
    it indefinitely just because the notification couldn't go out."""
    webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("SLACK_WEBHOOK_URL not set; skipping Slack alert.")
        return False

    try:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(webhook_url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status >= 300:
                raise RuntimeError(f"Slack webhook returned status {response.status}")
    except Exception as e:
        print(f"WARNING: Slack alert failed to send: {type(e).__name__}: {e}")
        return False

    print("Slack alert sent.")
    return True


def build_slack_payload(comparison: ComparisonResult, report_url: str | None = None) -> dict:
    emoji = STATUS_EMOJI[comparison.status]
    lines = [
        f"{emoji} *Model regression check: {comparison.status.upper()}*",
        f"Pass rate: {comparison.previous_pass_rate:.1%} -> {comparison.current_pass_rate:.1%} ({comparison.pass_rate_delta:+.1%})",
    ]
    if comparison.regressions:
        lines.append(f"{len(comparison.regressions)} regression(s): {', '.join(f.test_case_id for f in comparison.regressions[:10])}")
    if comparison.improvements:
        lines.append(f"{len(comparison.improvements)} improvement(s)")
    if report_url:
        lines.append(f"<{report_url}|View full diff report>")

    return {"text": "\n".join(lines)}


def send_slack_alert(comparison: ComparisonResult, report_url: str | None = None, webhook_url: str | None = None) -> bool:
    return _post_to_slack(build_slack_payload(comparison, report_url), webhook_url)


def build_drift_slack_payload(drift: DriftResult) -> dict:
    emoji = STATUS_EMOJI[drift.status]
    return {"text": "\n".join([
        f"{emoji} *Slow drift check: {drift.status.upper()}*",
        f"{drift.window_size}-run moving average: {drift.baseline_moving_average:.1%} ({drift.baseline_run_id}) -> {drift.current_moving_average:.1%} ({drift.current_run_id})",
        f"Drift: {drift.drift_delta:+.1%}",
    ])}


def send_drift_alert(drift: DriftResult, webhook_url: str | None = None) -> bool:
    """Only alerts when drift crosses a threshold - a 'pass' status means no slow drift detected."""
    if drift.status == "pass":
        return False
    return _post_to_slack(build_drift_slack_payload(drift), webhook_url)
