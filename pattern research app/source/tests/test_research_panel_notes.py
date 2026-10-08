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


def test_note_editor_opens_in_large_dialog_with_buttons_below_text_area():
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
    item = panel.note_list.item(0)
    geometry = {}

    def inspect_dialog():
        dialog = app.activeModalWidget()
        try:
            editor = dialog.findChild(QtWidgets.QPlainTextEdit)
            buttons = {button.text(): button for button in dialog.findChildren(QtWidgets.QPushButton)}
            geometry.update({
                "minimum_width": dialog.minimumWidth(),
                "minimum_height": dialog.minimumHeight(),
                "editor": editor.geometry().getRect(),
                "save": buttons["儲存"].geometry().getRect(),
                "cancel": buttons["取消"].geometry().getRect(),
            })
        finally:
            dialog.reject()

    QtCore.QTimer.singleShot(0, inspect_dialog)
    panel._begin_note_edit(item)

    assert geometry["minimum_width"] >= 520
    assert geometry["minimum_height"] >= 320
    editor_x, editor_y, editor_width, editor_height = geometry["editor"]
    save_y = geometry["save"][1]
    cancel_y = geometry["cancel"][1]
    assert editor_width >= 480
    assert editor_height >= 180
    assert save_y >= editor_y + editor_height
    assert cancel_y >= editor_y + editor_height
    assert panel._editing_note_id is None
    assert panel._note_editor is None


def test_note_editor_dialog_save_updates_note_and_history():
    app = _app()
    panel = ResearchPanel()
    case = SimpleNamespace(
        patterns=[],
        intraday_notes=[{
            "id": "note-3",
            "replay_time": "2026-03-02 09:49:00 EST",
            "text": "Original note",
        }],
        display={},
        reference_levels={},
        calendar={},
        touch=lambda: None,
    )
    panel.set_case(case, lambda: "")
    changes = []
    history = []
    panel.changed.connect(lambda: changes.append(True))
    panel.history_committed.connect(history.append)

    def save_from_dialog():
        dialog = app.activeModalWidget()
        editor = dialog.findChild(QtWidgets.QPlainTextEdit)
        editor.setPlainText("Updated note")
        buttons = {button.text(): button for button in dialog.findChildren(QtWidgets.QPushButton)}
        buttons["儲存"].click()

    QtCore.QTimer.singleShot(0, save_from_dialog)
    panel._begin_note_edit(panel.note_list.item(0))

    assert case.intraday_notes[0]["text"] == "Updated note"
    assert changes
    assert history == ["Edit Note"]
