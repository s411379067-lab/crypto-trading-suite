import pytest

from shared_core.order_brackets import create_pending_bracket, move_bracket_entry, move_bracket_leg


def test_initial_short_bracket_has_symmetric_stop_and_target():
    bracket = create_pending_bracket("short", 100.0, 2.0)

    assert bracket["state"] == "pending"
    assert bracket["groups"][0]["qty"] == 2.0
    assert bracket["groups"][0]["stop_price"] == 101.0
    assert bracket["groups"][0]["target_price"] == 99.0


def test_initial_long_bracket_has_symmetric_stop_and_target():
    bracket = create_pending_bracket("long", 100.0, 1.0)

    assert bracket["groups"][0]["stop_price"] == 99.0
    assert bracket["groups"][0]["target_price"] == 101.0


def test_initial_bracket_requires_positive_quantity():
    with pytest.raises(ValueError, match="positive"):
        create_pending_bracket("short", 100.0, 0)


def test_moving_entry_shifts_its_stop_and_target_together():
    bracket = create_pending_bracket("short", 100.0, 1.0)
    moved = move_bracket_entry(bracket, 105.0)

    assert bracket["entry_price"] == 100.0
    assert moved["entry_price"] == 105.0
    assert moved["groups"][0]["stop_price"] == 106.0
    assert moved["groups"][0]["target_price"] == 104.0


def test_moving_one_leg_keeps_the_other_leg_unchanged():
    bracket = create_pending_bracket("long", 100.0, 1.0)
    group_id = bracket["groups"][0]["id"]
    moved = move_bracket_leg(bracket, group_id, "stop", 97.0)

    assert moved["groups"][0]["stop_price"] == 97.0
    assert moved["groups"][0]["target_price"] == 101.0
