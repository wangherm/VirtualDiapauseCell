"""Frozen design and stateless prediction for the 2014 calibration panel."""
import numpy as np


def sigmoid(eta):
    return np.exp(-np.logaddexp(0.0, -eta))


def panel(schema):
    lines = schema["genotypes"]
    arms = [x["id"] for x in schema["conditions"]]
    assert len(lines) == len(set(lines)) == 4 and len(arms) == len(set(arms)) == 6
    assert schema["admitted_harvests"] == [2014] and schema["admitted_background"] == "Columbia-0"
    columns = (["intercept"] + [f"line:{g}" for g in lines[1:]] +
               [f"condition:{a}" for a in arms[1:]])
    assert schema["feature_design"]["columns"] == columns
    return lines, arms


def canonical_ids(schema, fold, held):
    lines, arms = panel(schema)
    assert type(fold) is int and 0 <= fold < 4
    return [f"2014/{g}/{a}" for i, g in enumerate(lines) for j, a in enumerate(arms)
            if (((i + j) % 4 == fold) == held)]


def design(schema, rows, *, source=False):
    lines, arms = panel(schema)
    fields = set(schema["request_fields"])
    assert fields == {"row_id", "harvest_year", "genotype", "condition_id"}
    allowed = fields | ({"germinated_fraction"} if source else set())
    x = []
    for r in rows:
        assert set(r) == allowed
        g, a = r["genotype"], r["condition_id"]
        assert type(r["harvest_year"]) is int and r["harvest_year"] == 2014
        assert g in lines and a in arms and r["row_id"] == f"2014/{g}/{a}"
        x.append([1.0] + [float(g == k) for k in lines[1:]]
                 + [float(a == k) for k in arms[1:]])
    assert x and len({r["row_id"] for r in rows}) == len(x)
    return np.asarray(x, dtype=np.float64)


def predict(model, schema, rows):
    x = design(schema, rows)
    assert model["protocol_id"] == schema["protocol_id"]
    assert model["feature_columns"] == schema["feature_design"]["columns"]
    p = {
        "two_way_ols_clipped": np.clip(x @ np.asarray(model["two_way_ols_coefficients"]), 0.0, 1.0),
        "condition_only_median": np.asarray([model["condition_medians"][r["condition_id"]] for r in rows]),
        "line_only_median": np.asarray([model["line_medians"][r["genotype"]] for r in rows]),
        "grand_median": np.full(len(rows), model["grand_median"]),
    }
    b = model["fractional_logit_coefficients"]
    if b is not None:
        p["fractional_logit_additive"] = sigmoid(x @ np.asarray(b, dtype=np.float64))
    assert all(y.shape == (len(rows),) and np.isfinite(y).all() and
               ((0.0 <= y) & (y <= 1.0)).all() for y in p.values())
    return p
