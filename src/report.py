import os
from html import escape
from pathlib import Path

from .comparer import build_category_map, compare_runs, list_runs, load_runs_safe, pass_rate
from .drift import detect_drift
from .errors import InsufficientRunHistory
from .models import ComparisonResult, DriftResult, EvalRun, ScoredResult
from .notifier import send_drift_alert, send_slack_alert
from .settings import ThresholdConfig

STATUS_COLORS = {"pass": "#1a7f37", "warning": "#9a6700", "critical": "#cf222e"}


def _case_lookup(run: EvalRun) -> dict[str, ScoredResult]:
    return {r.test_case_id: r for r in run.results}


def _cell(text: str | None) -> str:
    return escape(text) if text else "<em>n/a</em>"


def _render_case_rows(
    flips, baseline_lookup: dict[str, ScoredResult], current_lookup: dict[str, ScoredResult]
) -> str:
    rows = []
    for flip in flips:
        baseline_result = baseline_lookup.get(flip.test_case_id)
        current_result = current_lookup.get(flip.test_case_id)
        rows.append(f"""
        <tr>
          <td>{escape(flip.test_case_id)}</td>
          <td>{escape(flip.category or "")}</td>
          <td>{_cell(baseline_result.generated_category if baseline_result else None)} / {_cell(baseline_result.generated_summary if baseline_result else None)}</td>
          <td>{_cell(current_result.generated_category if current_result else None)} / {_cell(current_result.generated_summary if current_result else None)}</td>
        </tr>""")
    return "".join(rows)


def _render_trend_svg(trend: list[tuple[str, float]], width: int = 640, height: int = 140) -> str:
    if len(trend) < 2:
        return "<p>Not enough runs yet for a trend chart.</p>"

    padding = 24
    plot_width = width - 2 * padding
    plot_height = height - 2 * padding
    step = plot_width / (len(trend) - 1)

    points = []
    for i, (_, rate) in enumerate(trend):
        x = padding + i * step
        y = padding + (1 - rate) * plot_height
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)

    circles = "".join(
        f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="3" fill="#0969da" />'
        for p in points
    )

    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Pass rate trend">
      <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#d0d7de" />
      <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#d0d7de" />
      <polyline points="{polyline}" fill="none" stroke="#0969da" stroke-width="2" />
      {circles}
    </svg>"""


def generate_html_report(
    comparison: ComparisonResult,
    baseline_run: EvalRun,
    current_run: EvalRun,
    trend: list[tuple[str, float]],
    output_dir: str = "./reports",
) -> Path:
    baseline_lookup = _case_lookup(baseline_run)
    current_lookup = _case_lookup(current_run)
    status_color = STATUS_COLORS[comparison.status]

    category_rows = "".join(
        f"""
        <tr>
          <td>{escape(delta.category)}</td>
          <td>{delta.previous_accuracy:.1%}</td>
          <td>{delta.current_accuracy:.1%}</td>
          <td>{delta.delta:+.1%}</td>
        </tr>"""
        for delta in comparison.category_deltas
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eval Diff Report — {escape(current_run.run_metadata.run_id)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #1f2328; }}
  h1, h2 {{ border-bottom: 1px solid #d0d7de; padding-bottom: 0.3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
  th, td {{ border: 1px solid #d0d7de; padding: 0.5rem; text-align: left; font-size: 0.9rem; }}
  th {{ background: #f6f8fa; }}
  .status-badge {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; color: white; font-weight: 600; background: {status_color}; }}
  .meta {{ color: #57606a; font-size: 0.9rem; }}
  .scorecard {{ display: flex; gap: 2rem; margin: 1rem 0 1.5rem; }}
  .scorecard div {{ font-size: 1.4rem; font-weight: 600; }}
  .scorecard span {{ display: block; font-size: 0.8rem; font-weight: 400; color: #57606a; }}
</style>
</head>
<body>
<h1>Eval Diff Report</h1>
<p class="meta">
  Prompt version: <strong>{escape(current_run.run_metadata.prompt_version)}</strong> &middot;
  Model: <strong>{escape(current_run.run_metadata.model)}</strong> &middot;
  Timestamp: <strong>{escape(current_run.run_metadata.timestamp)}</strong>
</p>
<p><span class="status-badge">{comparison.status.upper()}</span></p>

<div class="scorecard">
  <div>{comparison.previous_pass_rate:.1%}<span>Baseline pass rate</span></div>
  <div>{comparison.current_pass_rate:.1%}<span>Current pass rate</span></div>
  <div>{comparison.pass_rate_delta:+.1%}<span>Delta</span></div>
  <div>{len(comparison.regressions)}<span>Regressions</span></div>
  <div>{len(comparison.improvements)}<span>Improvements</span></div>
</div>

<h2>Per-category accuracy</h2>
<table>
  <tr><th>Category</th><th>Baseline</th><th>Current</th><th>Delta</th></tr>
  {category_rows}
</table>

<h2>Regressions ({len(comparison.regressions)})</h2>
<table>
  <tr><th>Case</th><th>Category</th><th>Baseline output (category / summary)</th><th>Current output (category / summary)</th></tr>
  {_render_case_rows(comparison.regressions, baseline_lookup, current_lookup) or '<tr><td colspan="4">None</td></tr>'}
</table>

<h2>Improvements ({len(comparison.improvements)})</h2>
<table>
  <tr><th>Case</th><th>Category</th><th>Baseline output (category / summary)</th><th>Current output (category / summary)</th></tr>
  {_render_case_rows(comparison.improvements, baseline_lookup, current_lookup) or '<tr><td colspan="4">None</td></tr>'}
</table>

<h2>Pass rate trend (last {len(trend)} runs)</h2>
{_render_trend_svg(trend)}
</body>
</html>
"""

    output_path = (
        Path(output_dir) / f"report_{current_run.run_metadata.run_id.replace(':', '-')}.html"
    )
    output_path.write_text(html)
    return output_path


def main(trend_size: int = 10) -> tuple[ComparisonResult, Path, DriftResult | None]:
    runs = load_runs_safe(list_runs())
    if len(runs) < 2:
        raise InsufficientRunHistory(
            f"Need at least 2 valid runs to build a report, found {len(runs)}."
        )

    baseline_run = runs[-2]
    current_run = runs[-1]
    category_map = build_category_map()
    thresholds = ThresholdConfig.from_env()
    comparison = compare_runs(baseline_run, current_run, category_map, thresholds)

    trend_runs = runs[-trend_size:]
    trend = [(r.run_metadata.run_id, pass_rate(r.results)) for r in trend_runs]

    report_path = generate_html_report(comparison, baseline_run, current_run, trend)
    print(f"Report saved to {report_path}")

    report_url = os.getenv("REPORT_PUBLIC_URL", str(report_path))
    send_slack_alert(comparison, report_url)

    drift = detect_drift()
    if drift is None:
        print("Drift check skipped: not enough run history yet.")
    else:
        print(
            f"Drift check: {drift.status.upper()} ({drift.drift_delta:+.1%} over {drift.window_size} runs)"
        )
        send_drift_alert(drift)

    return comparison, report_path, drift


if __name__ == "__main__":
    try:
        main()
    except InsufficientRunHistory as e:
        print(f"{e} Run `python -m src.pipeline` to record runs first.")
        raise SystemExit(1) from None
