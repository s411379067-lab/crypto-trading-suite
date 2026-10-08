import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtCore, QtWidgets

from pattern_analyzer.research_panel import IntradayNoteDelegate, ResearchPanel


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_intraday_note_rows_show_timestamp_and_text_as_two_tone_fields():
    app = _app()
    panel = ResearchPanel()
    case = SimpleNamespace(
        patterns=[],
        intraday_notes=[{
            "id": "note-1",
            "replay_time": "2026-03-02 09:49:00 EST",
            "text": "Retest < 1R",
        }],
        display={},
        reference_levels={},
        calendar={},
    )

    panel.set_case(case, lambda: "")

    item = panel.note_list.item(0)
    assert item.text() == "2026-03-02 09:49:00 EST\nRetest < 1R"
    assert item.data(QtCore.Qt.UserRole) == "note-1"
    assert isinstance(panel.note_list.itemDelegate(), IntradayNoteDelegate)

    option = QtWidgets.QStyleOptionViewItem()
    option.rect = QtCore.QRect(0, 0, 500, 30)
    option.font = panel.note_list.font()
    document = panel.note_list.itemDelegate()._document(option, panel.note_list.model().index(0, 0))
    rendered_html = document.toHtml()
    assert document.toPlainText() == "2026-03-02 09:49:00 EST\nRetest < 1R"
    assert "#62c9ff" in rendered_html
    assert "#e6edf7" in rendered_html
    assert "Retest &lt; 1R" in rendered_html

    panel.resize(640, 480)
    panel.show()
    app.processEvents()
    assert not panel.note_list.viewport().grab().isNull()
    panel.close()


def test_note_editor_gives_text_area_full_width_above_action_buttons():
    app = _app()
    panel = ResearchPanel()
    case = SimpleNamespace(
        patterns=[],
        intraday_notes=[{
            "id": "note-2",
            "replay_time": "2026-03-02 09:49:00 EST",
            "text": "A longer note body for editing",
        }],
        display={},
        reference_levels={},
        calendar={},
    )
    panel.set_case(case, lambda: "")
    panel.resize(360, 640)
    panel.show()
    app.processEvents()

    item = panel.note_list.item(0)
    panel._begin_note_edit(item)
    app.processEvents()
    editor_row = panel.note_list.itemWidget(item)
    editor = editor_row.findChild(QtWidgets.QPlainTextEdit)

    assert editor is not None
    assert editor.width() > 200
    assert editor.height() >= 96
    panel.close()
