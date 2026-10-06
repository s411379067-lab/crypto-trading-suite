import pytest

from shared_core.order_brackets import (
    available_bracket_qty,
    create_pending_bracket,
    move_bracket_entry,
    move_bracket_leg,
    set_bracket_group_qty,
    set_bracket_qty,
)


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


def test_reducing_entry_lots_is_rejected_when_it_would_over_allocate_groups():
    bracket = create_pending_bracket("short", 100.0, 2.0)

    with pytest.raises(ValueError, match="allocated"):
        set_bracket_qty(bracket, 1.0)


def test_entry_lots_can_increase_without_changing_existing_group_lots():
    bracket = create_pending_bracket("long", 100.0, 1.0)
    edited = set_bracket_qty(bracket, 2.5)

    assert edited["groups"][0]["qty"] == 1.0
    assert available_bracket_qty(edited) == 1.5


def test_sl_and_tp_share_one_editable_group_lot_value():
    bracket = create_pending_bracket("short", 100.0, 2.0)
    group_id = bracket["groups"][0]["id"]
    edited = set_bracket_group_qty(bracket, group_id, 0.75)

    assert edited["groups"][0]["qty"] == 0.75
    assert available_bracket_qty(edited) == 1.25


def test_group_lots_cannot_exceed_entry_lots():
    bracket = create_pending_bracket("long", 100.0, 1.0)
    group_id = bracket["groups"][0]["id"]

    with pytest.raises(ValueError, match="exceeds"):
        set_bracket_group_qty(bracket, group_id, 1.5)
