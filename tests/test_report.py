"""HTML report rendering, including escaping of model-generated content."""
from src.config import CaseFlip, CategoryDelta
from src.report import _render_trend_svg, generate_html_report


def test_report_escapes_model_generated_content(tmp_path, make_run, make_scored_result, make_comparison):
    """Summaries come back from an LLM and land in an HTML page a human opens - they must be
    escaped, not interpolated raw."""
    payload = '<script>alert("xss")</script>'
    baseline = make_run("r1", [make_scored_result("tc_1", True, generated_summary="fine")])
    current = make_run("r2", [make_scored_result("tc_1", False, generated_summary=payload)])
    comparison = make_comparison(
        status="critical",
        previous=1.0,
        current=0.0,
        regressions=[CaseFlip(test_case_id="tc_1", category="billing")],
        category_deltas=[CategoryDelta(category="billing", previous_accuracy=1.0, current_accuracy=0.0, delta=-1.0)],
    )

    path = generate_html_report(comparison, baseline, current, [("r1", 1.0), ("r2", 0.0)], output_dir=str(tmp_path))
    html = path.read_text()

    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_report_is_well_formed(tmp_path, make_run, make_scored_result, make_comparison):
    baseline = make_run("r1", [make_scored_result("tc_1", True)])
    current = make_run("r2", [make_scored_result("tc_1", True)])
    path = generate_html_report(make_comparison(), baseline, current, [("r1", 1.0), ("r2", 1.0)], output_dir=str(tmp_path))
    html = path.read_text()
    assert html.startswith("<!doctype html>")
    assert "</html>" in html
    assert "<title>" in html


def test_report_renders_run_metadata(tmp_path, make_run, make_scored_result, make_comparison):
    baseline = make_run("r1", [make_scored_result("tc_1", True)])
    current = make_run("r2", [make_scored_result("tc_1", True)], prompt_version="v9", model="gpt-4o")
    path = generate_html_report(make_comparison(), baseline, current, [("r1", 1.0)], output_dir=str(tmp_path))
    html = path.read_text()
    assert "v9" in html
    assert "gpt-4o" in html


def test_trend_chart_handles_too_few_points():
    assert "Not enough" in _render_trend_svg([])
    assert "Not enough" in _render_trend_svg([("r1", 0.5)])


def test_trend_chart_renders_svg_with_two_points():
    svg = _render_trend_svg([("r1", 0.5), ("r2", 0.7)])
    assert "<svg" in svg
    assert "<polyline" in svg
