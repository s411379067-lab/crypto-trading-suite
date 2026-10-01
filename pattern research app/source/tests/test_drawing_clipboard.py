from shared_core.drawing_clipboard import clone_drawing_with_offset


def test_clone_trend_preserves_style_and_offsets_geometry():
    original = {
        "id": "drawing-old",
        "type": "trend_line",
        "points": [{"time": 100.0, "price": 10.0}, {"time": 200.0, "price": 20.0}],
        "style": {"color": "#ffffff", "width": 3, "line_style": "dashed"},
    }
    clone = clone_drawing_with_offset(original, 5.0, -2.0)
    assert clone["id"] != original["id"]
    assert clone["style"] == original["style"]
    assert clone["points"] == [{"time": 105.0, "price": 8.0}, {"time": 205.0, "price": 18.0}]
    assert original["points"][0] == {"time": 100.0, "price": 10.0}


def test_clone_rectangle_fibo_text_and_hline():
    rect = {"id": "r", "type": "rectangle", "points": [{"time": 1, "price": 2}, {"time": 3, "price": 4}], "style": {"opacity": 20}}
    rc = clone_drawing_with_offset(rect, 10, 1)
    assert rc["points"] == [{"time": 11.0, "price": 3.0}, {"time": 13.0, "price": 5.0}]

    fib = {"id": "f", "type": "fibonacci", "start": {"time": 1, "price": 10}, "end": {"time": 2, "price": 20}, "levels": [{"multiplier": 0.5, "color": "#fff"}]}
    fc = clone_drawing_with_offset(fib, 10, -5)
    assert fc["start"] == {"time": 11.0, "price": 5.0}
    assert fc["end"] == {"time": 12.0, "price": 15.0}
    assert fc["levels"] == fib["levels"]

    txt = {"id": "t", "type": "text", "time": 5, "price": 7, "text": "ABC", "box": {"width": 30, "height": 4}, "style": {"bold": True}}
    tc = clone_drawing_with_offset(txt, 2, 3)
    assert tc["time"] == 7.0 and tc["price"] == 10.0
    assert tc["text"] == "ABC" and tc["box"] == txt["box"] and tc["style"] == txt["style"]

    hl = {"id": "h", "type": "horizontal_line", "price": 100, "style": {"color": "#fff"}}
    hc = clone_drawing_with_offset(hl, 999, -4)
    assert hc["price"] == 96.0
