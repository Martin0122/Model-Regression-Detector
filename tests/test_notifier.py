"""Slack alerting, focused on delivery failures.

Alerting is a side effect of a run that already succeeded and was saved. A bad webhook, a
Slack outage, or a hung endpoint must never crash the pipeline or block it indefinitely.
"""
import http.server
import json
import threading

import pytest

from src.notifier import build_drift_slack_payload, build_slack_payload, send_slack_alert
from src.config import CaseFlip, DriftResult


@pytest.fixture
def http_server():
    """Starts throwaway webhook endpoints and tears them down fully.

    shutdown() only stops the serve_forever loop - without server_close() the listening socket
    stays open, and without joining the thread the teardown races the loop's exit.
    """
    started = []

    def _start(status_code, capture=None):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                if capture is not None:
                    capture.append((dict(self.headers), body))
                self.send_response(status_code)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started.append((server, thread))
        return f"http://127.0.0.1:{server.server_address[1]}/webhook"

    yield _start

    for server, thread in started:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive(), "webhook server thread did not stop"


def test_successful_post_sends_json_payload(http_server, make_comparison):
    captured = []
    url = http_server(200, capture=captured)

    assert send_slack_alert(make_comparison(status="critical"), "http://report", webhook_url=url) is True
    headers, body = captured[0]
    assert headers["Content-Type"] == "application/json"
    assert "text" in json.loads(body)


def test_http_error_does_not_raise(http_server, make_comparison):
    url = http_server(404)
    assert send_slack_alert(make_comparison(), "http://report", webhook_url=url) is False


def test_unreachable_host_does_not_raise(make_comparison):
    assert send_slack_alert(make_comparison(), "http://report", webhook_url="http://127.0.0.1:1/nope") is False


def test_malformed_url_does_not_raise(make_comparison):
    """The Request() constructor validates the URL and raises before urlopen is reached,
    so the guarded region has to cover construction too, not just the network call."""
    assert send_slack_alert(make_comparison(), "http://report", webhook_url="not-a-valid-url") is False


def test_missing_webhook_is_skipped_not_an_error(monkeypatch, make_comparison):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    assert send_slack_alert(make_comparison(), "http://report", webhook_url=None) is False


def test_payload_includes_status_and_rates(make_comparison):
    comparison = make_comparison(
        status="critical",
        previous=0.94,
        current=0.89,
        regressions=[CaseFlip(test_case_id="tc_7", category="billing")],
    )
    text = build_slack_payload(comparison, "http://report/x.html")["text"]
    assert "CRITICAL" in text
    assert "94.0%" in text and "89.0%" in text
    assert "tc_7" in text
    assert "http://report/x.html" in text


def test_drift_payload_includes_window_and_delta():
    drift = DriftResult(
        window_size=7,
        baseline_run_id="r1",
        baseline_moving_average=1.0,
        current_run_id="r8",
        current_moving_average=0.9,
        drift_delta=-0.1,
        status="critical",
    )
    text = build_drift_slack_payload(drift)["text"]
    assert "CRITICAL" in text
    assert "7-run moving average" in text
