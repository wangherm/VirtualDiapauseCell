"""Small CPU reconstruction pilot plus explicitly chronological reference curves.

This does not train a molecular clock, Qwen, or a biological world model.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from vdc.contracts import ObservationBundle
from vdc.state import fit_state, StateConfig, StatePredictor
from vdc.evaluate import evaluate_state
from vdc.waves import WaveReference
from vdc.metrics import grouped_metrics
from vdc.io import write_json, save_npz, object_hash


def run(bundle_path, output, steps):
    if output.exists():
        raise FileExistsError('Choose a new run directory')
    b = ObservationBundle.load(bundle_path)
    if b.clock_mask.any() or b.context['system'] != 'daf-2 dauer and exit':
        raise ValueError('This fixed pilot expects the admitted reconstruction-only cohort')
    torch.set_num_threads(2)
    config = StateConfig(hidden_dim=32, latent_dim=8, layers=1, heads=2,
                         validation_every=10, clock_weight=0)
    fit_state(b, output / 'state', steps=steps, config=config, device='cpu')
    predictor = StatePredictor.load(output / 'state')
    predictions = predictor.predict(b)
    save_npz(output / 'state_predictions.npz', programme=predictions['programme'],
             latent=predictions['latent'], measured=b.values, mask=b.mask)
    evaluation = evaluate_state(output / 'state', b, output / 'evaluation', split='test')
    curve_reports = {}
    context_id = object_hash(b.context)
    coordinate_id = 'elapsed_hours_since_dauer_exit_NOT_biotime'
    hours = np.array([r['elapsed_hours_since_release'] for r in b.rows], dtype=float)
    for history in (1, 4, 15, 30):
        idx = np.array([i for i, r in enumerate(b.rows) if r['history_days'] == history])
        rows = [b.rows[i] for i in idx]
        wave = WaveReference.fit(hours[idx], b.values[idx], b.mask[idx], b.feature_ids,
                                 rows, coordinate_id, context_id, degree=1)
        folder = output / f'chronological_waves/history_{history}d'
        wave.save(folder)
        result = WaveReference.load(folder).residual(hours[idx], b.values[idx], b.mask[idx],
                                                    coordinate_id, context_id)
        save_npz(folder / 'residuals.npz', **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
        test = np.array([r['split'] == 'test' for r in rows])
        curve_reports[str(history)] = grouped_metrics(result['expected'][test], b.values[idx][test],
                                                       result['residual_mask'][test],
                                                       [r for r in rows if r['split'] == 'test'])
    report = {'public_data': 'GSE288723', 'reconstruction': 'executed_on_real_expression',
              'independent_evaluation': 'within_study_conservative_replicate_blocks_only',
              'state_baselines': evaluation['baseline_comparisons'],
              'clock_training': 'not_run_no_admitted_molecular_clock_labels',
              'biotime_waves': 'not_run_no_admitted_coordinate',
              'chronological_curves': {'status': 'executed', 'coordinate': coordinate_id,
                 'degree': 1, 'distinct_times': 3, 'test_metrics_by_history': curve_reports,
                 'limitation': 'One training pool per time and history; no fine peaks, lags or mechanistic claims'},
              'real_perturbation_training': 'not_run', 'functional_depth': 'unavailable',
              'qwen': 'not_run', 'research_approval': 'not_approved', 'deployment': 'not_run',
              'internal_killifish_used': False, 'world_model_validated': False}
    write_json(output / 'status.json', report)
    print(f'Completed bounded public pilot: {output}/status.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('bundle', type=Path); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=50)
    args = p.parse_args()
    run(args.bundle, args.out, args.steps)
