"""Reload a frozen functional state and run its feature-only query."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from chilling_functional_model import canonical_ids, predict


def pin(b):
    return {"bytes": len(b), "sha256": sha256(b).hexdigest()}


def run(model, schema, packet):
    assert packet["protocol_id"] == model["protocol_id"] == schema["protocol_id"]
    assert model["fold"] == packet["fold"] and packet["role"] == "query"
    assert set(packet) == {"protocol_id", "schema_pin", "fold", "role", "rows"}
    assert packet["schema_pin"] == model["schema_pin"]
    rows = packet["rows"]
    assert [r["row_id"] for r in rows] == canonical_ids(schema, packet["fold"], True)
    heads = predict(model, schema, rows)
    assert set(heads) == {"fractional_logit_additive", "two_way_ols_clipped",
                          "condition_only_median", "line_only_median", "grand_median"}
    return {"protocol_id": packet["protocol_id"], "schema_pin": packet["schema_pin"],
            "fold": packet["fold"], "target": "germinated_fraction",
            "row_ids": [r["row_id"] for r in rows],
            "predictions": {key: value.tolist() for key, value in heads.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "schema", "query", "output"):
        p.add_argument(f"--{key}", type=Path, required=True)
    args = p.parse_args()
    sr, mr, qr = [p.read_bytes() for p in (args.schema, args.model, args.query)]
    schema, model, query = map(json.loads, (sr, mr, qr))
    assert pin(sr) == model["schema_pin"] == query["schema_pin"]
    assert pin(Path(__file__).with_name("chilling_functional_model.py").read_bytes()) == model["predictor_code_pin"]
    output = run(model, schema, query)
    output["model_pin"] = pin(mr)
    output["query_pin"] = pin(qr)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
