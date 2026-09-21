from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_frontend_file(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_template_contains_teacher_dashboard_views_and_controls():
    template = read_frontend_file("templates/index.html")

    for view in ("analyzer", "history", "settings", "documentation", "about"):
        assert f'id="view-{view}"' in template
    for control in ("clearText", "historyList", "detailsToggle", "animationsToggle", "infoModal"):
        assert f'id="{control}"' in template
    assert 'name="density" value="comfortable"' in template
    assert 'name="density" value="compact"' in template
    assert "shield-check" in template
    assert "mailto:" in template


def test_javascript_contains_local_history_and_input_workflows():
    script = read_frontend_file("static/js/script.js")

    for feature in (
        "localStorage",
        "saveHistory",
        "renderHistory",
        "FileReader",
        "dataTransfer",
        "animationsToggle",
        "density",
        "infoModal",
        "/api/feedback",
    ):
        assert feature in script
    assert "modelVersion" in script
    assert "prediction" in script


def test_styles_define_responsive_dashboard_and_accessibility_states():
    styles = read_frontend_file("static/css/style.css")

    for selector in (".sidebar", ".sidebar-collapsed", ".sidebar-open", ".modal-backdrop", ".editor-shell.is-dragging"):
        assert selector in styles
    assert "@media (max-width: 899px)" in styles
    assert "@media (max-width: 700px)" in styles
    assert ".sr-only" in styles
