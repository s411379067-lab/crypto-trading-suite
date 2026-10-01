from shared_core.models import ResearchCase


def make_case():
    return {
        "schema_version": "0.1",
        "case": {"id": "case-x", "symbol": "TEST", "research_date": "2026-09-17"},
        "market_data": {"source_id": "x", "source_type": "txt", "resolution": "M1", "source_path": "dummy.txt"},
        "time_range": {"data_start": "2026-09-17T00:00:00+00:00", "replay_start": "2026-09-17T01:00:00+00:00", "default_end": "2026-09-17T02:00:00+00:00"},
        "display": {}, "replay": {}, "drawings": []
    }


case = ResearchCase.from_dict(make_case())
assert case.orders == []
case.orders.append({"id": "order-1", "status": "open"})
out = case.to_dict()
assert out["orders"][0]["id"] == "order-1"
print("case order compatibility test passed")
