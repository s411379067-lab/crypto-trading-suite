from shared_core.undo import ResearchUndoManager


class DummyCase:
    def __init__(self):
        self.drawings = []
        self.patterns = []
        self.intraday_notes = []
        self.display = {"view_timeframe": "M5"}
        self.replay = {"current_time": "x"}


def test_record_undo_redo_research_only():
    case = DummyCase()
    mgr = ResearchUndoManager()
    mgr.reset(case)

    case.drawings.append({"id": "d1", "type": "horizontal_line", "price": 100})
    assert mgr.record(case, "Add Drawing")
    case.display["view_timeframe"] = "M15"
    case.replay["current_time"] = "y"

    assert mgr.undo(case) == "Add Drawing"
    assert case.drawings == []
    assert case.display["view_timeframe"] == "M15"
    assert case.replay["current_time"] == "y"

    assert mgr.redo(case) == "Add Drawing"
    assert case.drawings[0]["price"] == 100


def test_new_edit_clears_redo():
    case = DummyCase()
    mgr = ResearchUndoManager()
    mgr.reset(case)
    case.patterns.append({"id": "p1", "text": "A"})
    mgr.record(case, "Add Pattern")
    mgr.undo(case)
    assert mgr.can_redo()

    case.intraday_notes.append({"id": "n1", "text": "B"})
    mgr.record(case, "Add Note")
    assert not mgr.can_redo()
