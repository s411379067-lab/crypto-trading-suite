from shared_core.geometry_picker import point_to_rect_distance, point_to_segment_distance


def test_segment_distance_uses_finite_segment():
    assert point_to_segment_distance(5, 3, 0, 0, 10, 0) == 3
    assert round(point_to_segment_distance(12, 0, 0, 0, 10, 0), 6) == 2


def test_text_box_interior_is_hit():
    assert point_to_rect_distance(5, 5, 0, 0, 10, 10, interior_is_hit=True) == 0


def test_rectangle_interior_measures_nearest_edge():
    assert point_to_rect_distance(5, 5, 0, 0, 10, 10, interior_is_hit=False) == 5
