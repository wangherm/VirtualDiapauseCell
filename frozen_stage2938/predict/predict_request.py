"""Predict held cells for one saved calibration fold of the 2014 panel."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from chilling_functional_model import canonical_ids, predict


def pin(raw):
    return {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "schema", "request", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    model_raw, schema_raw, request_raw = [p.read_bytes() for p in (args.model, args.schema, args.request)]
    model, schema, request = map(json.loads, (model_raw, schema_raw, request_raw))
    assert set(request) == {"protocol_id", "schema_pin", "fold", "role", "rows"}
    assert request["role"] == "query" and request["protocol_id"] == schema["protocol_id"] == model["protocol_id"]
    assert pin(schema_raw) == request["schema_pin"] == model["schema_pin"]
    assert pin(Path(__file__).with_name("chilling_functional_model.py").read_bytes()) == model["predictor_code_pin"]
    assert schema["protocol_pin"] == model["protocol_pin"] and request["fold"] == model["fold"]
    rows = request["rows"]
    row_ids = [row["row_id"] for row in rows]
    assert row_ids and set(row_ids) <= set(canonical_ids(schema, model["fold"], True))
    heads = predict(model, schema, rows)
    expected_heads = {schema["model_contract"]["candidate"], *schema["model_contract"]["comparators"]}
    assert set(heads) == expected_heads
    result = {
        "protocol_id": model["protocol_id"], "model_version": model["model_version"],
        "fold": model["fold"], "target": model["target"], "row_ids": row_ids,
        "predictions": {name: values.tolist() for name, values in heads.items()},
        "schema_pin": pin(schema_raw), "model_pin": pin(model_raw), "request_pin": pin(request_raw),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
