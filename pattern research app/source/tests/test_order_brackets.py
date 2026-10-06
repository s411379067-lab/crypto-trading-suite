import pytest

from shared_core.order_brackets import create_pending_bracket


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
