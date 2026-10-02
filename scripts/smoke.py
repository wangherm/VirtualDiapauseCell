"""Explicit synthetic integration run. Never downloads or relabels biological data."""
import argparse
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from tests.fixtures import observation_fixture, response_fixture
from vdc.state import fit_state, StateConfig, StatePredictor
from vdc.response import ResponseRegressor
from vdc.waves import WaveReference
from vdc.io import write_json, save_npz


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", default="work/synthetic_smoke")
    args = p.parse_args(); dest = Path(args.out)
    if dest.exists(): raise FileExistsError("Choose a new smoke output directory")
    dest.mkdir(parents=True)
    torch.set_num_threads(1)
    b = observation_fixture(); b.save(dest / "bundle")
    fit_state(b, dest / "state", steps=15, config=StateConfig(hidden_dim=16, latent_dim=8,
              layers=1, heads=2, validation_every=5))
    pred = StatePredictor.load(dest / "state").predict(b)
    save_npz(dest / "state_predictions.npz", programme=pred["programme"], clock=pred["clock"])
    w = WaveReference.fit(b.clock, b.values, b.mask, b.feature_ids, b.rows,
                         b.clock_reference_id, "synthetic-context-v1")
    w.save(dest / "waves")
    for mode in ("endpoint", "transition", "functional"):
        d = response_fixture(mode); d.save(dest / f"{mode}_data")
        m = ResponseRegressor().fit(d); m.save(dest / mode)
        yp = m.predict(d.current, d.action, d.scope, d.elapsed)
        save_npz(dest / f"{mode}_prediction.npz", prediction=yp)
    write_json(dest / "status.json", {"software_smoke": "passed", "origin": "synthetic",
                "biological_validation": "not_run", "qwen_execution": "not_run",
                "deployment": "not_run", "weights_may_be_promoted": False})
    print(f"Synthetic integration completed: {dest}")


if __name__ == "__main__": main()
