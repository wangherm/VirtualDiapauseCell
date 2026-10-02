"""Optional thin, loopback-only state inference API; training is never run in requests."""
from __future__ import annotations
import numpy as np
from .release import verify_release
from .state import StatePredictor
from .contracts import ObservationBundle


def create_app(release_dir: str):
    from fastapi import FastAPI, HTTPException
    release = verify_release(release_dir)
    predictor = StatePredictor.load(release_dir)
    app = FastAPI(title="Virtual Diapause Cell — local research inference", version="0.4.0")

    @app.get("/health")
    def health():
        return {"ready": True, "usage": "local_research_only"}

    @app.get("/capabilities")
    def capabilities():
        return {"capabilities": release["capabilities"], "scope": release["scope"],
                "unsupported": ["universal_depth", "unseen_species_gene_generation", "causal_rollout"]}

    @app.post("/predict/state")
    def predict(payload: dict):
        try:
            values = np.asarray(payload["values"], dtype=np.float32)
            if values.ndim != 2 or len(values) > 256:
                raise ValueError("Use batches of 1..256 observations")
            n = len(values)
            scope = {"feature_ids": payload["feature_ids"], "context": payload["context"],
                     "clock_reference_id": payload.get("clock_reference_id")}
            rows = [{"observation_id": f"request-{i}", "study_family": "request", "biological_unit": f"u{i}",
                     "split": "test", "origin": "public", "link_ids": []} for i in range(n)]
            bundle = ObservationBundle(values, np.asarray(payload["mask"], bool),
                         np.asarray(payload["coverage"], np.float32), rows=rows, split_policy="unit_holdout",
                         clock=np.zeros(n), clock_mask=np.zeros(n, bool), **scope)
            pred = predictor.predict(bundle)
            result = {"scope": scope, "status": "approved_for_declared_research_scope",
                      "uncertainty_status": pred["uncertainty_status"], "depth": None}
            if "programme_reconstruction" in release["capabilities"]:
                result["programme_prediction"] = pred["programme"].tolist()
                result["measured_mask"] = pred["measured_mask"].tolist()
            if "clock" in release["capabilities"]:
                result["clock"] = pred["clock"].tolist()
            return result
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
