from pathlib import Path
import tempfile

from pattern_analyzer.drawing_templates import DrawingTemplateRepository


def test_template_repository_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        repo = DrawingTemplateRepository(Path(tmp) / "drawing template")
        repo.save("rectangle", "多單", {
            "style": {
                "border_color": "#4caf50",
                "fill_color": "#4caf50",
                "opacity": 20,
                "width": 2,
            }
        })
        assert repo.exists("rectangle", "多單")
        items = repo.list("rectangle")
        assert len(items) == 1
        assert items[0]["name"] == "多單"
        assert items[0]["style"]["opacity"] == 20
        assert not repo.list("line")


def test_line_category_shared_by_hline_and_trend():
    repo = DrawingTemplateRepository.category_for_drawing_type
    assert repo("horizontal_line") == "line"
    assert repo("trend_line") == "line"
    assert repo("fibonacci") == "fibonacci"
