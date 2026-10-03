"""All-Module Alpha adapters around the existing state, wave and response models."""
from pathlib import Path
import copy
import numpy as np
import torch
from .contracts import ObservationBundle
from .clock_reference import ExitReference
from .state import StateConfig, StatePredictor, fit_state
from .evaluate import evaluate_state
from .response import ResponseDataset, ResponseRegressor
from .waves import WaveReference
from .metrics import regression_metrics
from .io import read_json, write_json, save_npz, sha256, object_hash


def gene_data(path):
    m = read_json(Path(path)/'expression_contract.json')
    if sha256(Path(path)/'gene_expression.npz') != m['expression_sha256']: raise ValueError('Expression fingerprint changed')
    with np.load(Path(path)/'gene_expression.npz', allow_pickle=False) as a:
        return a['expression'].copy(), a['genes'].tolist()


def metric(pred, target, mask=None):
    if mask is None: mask = np.ones_like(target, bool)
    return regression_metrics(pred, target, mask)


def reference_task(run):
    run = Path(run); out = run/'clock_reference'
    b = ObservationBundle.load(run/'data/dauer'); x, genes = gene_data(run/'data/dauer')
    ref = ExitReference().fit(x, genes, b.rows); ref.save(out)
    score = ExitReference.load(out).predict(x, genes)
    b.clock, b.clock_mask, b.clock_reference_id = score, np.ones(len(score), bool), ref.reference_id
    b.save(out/'bundle')
    save_npz(out/'reference_scores.npz', scores=score, hours=np.array([r['elapsed_hours_since_release'] for r in b.rows]))
    write_json(out/'result.json', {'label_source': 'reference_derived', 'clock_truth_independent': False,
        'train_anchor_mean': {str(t): float(score[[i for i in b.indices('train') if b.rows[i]['elapsed_hours_since_release'] == t]].mean()) for t in (0, 6, 24)},
        'clipped': False, 'fit_kind': 'reference_fit', 'axis_genes': len(ref.indices),
        'readout_panel_genes': len(ref.metadata['readout_genes']), 'science_status': 'unvalidated'})


def state_task(run, name, config):
    run = Path(run); out = run/name
    path = run/'data/ard' if name == 'state_ard' else run/'clock_reference/bundle'
    b = ObservationBundle.load(path)
    semantics, provenance = None, None
    if name in {'state_semantic', 'state_matched_zero'}:
        e = run/'semantics'
        provenance = read_json(e/'embeddings.json')
        if provenance.get('weight_kind') != 'domain_adapter': raise ValueError('Domain adapter was not used for embeddings')
        if provenance['arrays_sha256'] != sha256(e/'embeddings.npz'): raise ValueError('Semantic cache checksum mismatch')
        with np.load(e/'embeddings.npz', allow_pickle=False) as a:
            if a['feature_ids'].tolist() != b.feature_ids: raise ValueError('Semantic feature order changed')
            semantics = a['vectors'].copy()
        if name == 'state_matched_zero':
            semantics = np.zeros_like(semantics); provenance = {**provenance, 'ablation': 'matched_capacity_zero_vectors'}
    cfg = StateConfig(**config['state_config'])
    fit_state(b, out, steps=config['state_steps'], config=cfg, semantics=semantics,
        semantic_provenance=provenance, device='cpu')
    evaluate_state(out, b, out/'evaluation', split='validation', seed=1007)
    p = StatePredictor.load(out); query = copy.deepcopy(b); query.clock_mask[:] = False; query.clock[:] = 0
    pred = p.predict(query)
    arrays = {'programme': pred['programme'], 'latent': pred['latent'], 'observed': b.values}
    if pred['clock'] is not None: arrays['clock'] = pred['clock']
    save_npz(out/'clean_predictions.npz', **arrays)
    tr, va = b.indices('train'), b.indices('validation')
    # Fixed ridge completion uses exactly the same corrupted inputs and targets as neural evaluation.
    from .state import corrupt
    mu, scale = p.mean, p.scale
    x = torch.tensor((b.values-mu)/scale, dtype=torch.float32)
    xx, mm, _, hidden = corrupt(x[tr], torch.tensor(b.mask[tr]), torch.tensor(b.coverage[tr]), cfg.corrupt_fraction, cfg.additive_noise, 1006)
    vx, vm, _, vh = corrupt(x[va], torch.tensor(b.mask[va]), torch.tensor(b.coverage[va]), cfg.corrupt_fraction, cfg.additive_noise, 1007)
    design = np.column_stack([xx.numpy(), mm.numpy().astype(float)])
    vdesign = np.column_stack([vx.numpy(), vm.numpy().astype(float)])
    coef, m, s = ridge_fit(design, b.values[tr])
    rp = ridge_predict(vdesign, coef, m, s)
    save_npz(out/'ridge_completion.npz', coefficients=coef, mean=m, scale=s, prediction=rp, observed=b.values[va], evaluation_mask=vh.numpy())
    ratios = np.std(pred['programme'][va], 0)/np.maximum(np.std(b.values[va], 0), 1e-8)
    report = {'ridge_hidden': metric(rp, b.values[va], vh.numpy()),
        'clean_dynamic_std_ratio_median': float(np.median(ratios)),
        'clock_clean_reference_error': metric(pred['clock'][va], b.clock[va]) if pred['clock'] is not None else None,
        'clock_is_reference_distillation': b.clock_reference_id is not None,
        'test_accessed': False, 'science_status': 'unvalidated'}
    write_json(out/'diagnostics.json', report)
    noise_rows=[]
    for level in (0.0, .1, .2, .4):
        # The same predeclared development corruption schedule for all semantic variants.
        nx,nm,nc,nh=corrupt(x[va],torch.tensor(b.mask[va]),torch.tensor(b.coverage[va]),
            max(level,1e-8),level/4,20261002)
        if level==0:
            nx,nm,nc=x[va],torch.tensor(b.mask[va]),torch.tensor(b.coverage[va])
        q=b.subset(va);q.values=(nx.numpy()*scale+mu).astype(np.float32);q.mask=nm.numpy();q.coverage=nc.numpy()
        q.clock[:]=0;q.clock_mask[:]=False
        npred=p.predict(q)
        noise_rows.append({'missing_fraction_requested':level,'actual_missing_values':int((b.mask[va]&~q.mask).sum()),
            'model':metric(npred['programme'],b.values[va],b.mask[va]),
            'input_train_mean_baseline':metric(np.where(q.mask,q.values,mu),b.values[va],b.mask[va]),
            'clock_reference_mse':metric(npred['clock'],b.clock[va]) if npred['clock'] is not None else None})
    write_json(out/'noise_curve.json',noise_rows)


def ridge_fit(x, y, alpha=1.0):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mean = x.mean(0); scale = np.maximum(x.std(0), .001)
    design = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    penalty = np.eye(design.shape[1])*alpha; penalty[0, 0] = 0
    return np.linalg.solve(design.T@design+penalty, design.T@y), mean, scale


def ridge_predict(x, coefficients, mean, scale):
    return np.column_stack([np.ones(len(x)), (x-mean)/scale])@coefficients


def classification_task(run):
    run = Path(run); out = run/'state_readout'; b = ObservationBundle.load(run/'clock_reference/bundle')
    p = StatePredictor.load(run/'state_base'); pred = p.predict(b)
    labels = np.array([0 if r['elapsed_hours_since_release'] == 0 else 1 for r in b.rows])
    tr, va = b.indices('train'), b.indices('validation')
    reports = {}
    for name, x in [('latent', pred['latent']), ('observed_programme', b.values)]:
        co, mu, scale = ridge_fit(x[tr], np.eye(2)[labels[tr]])
        score = ridge_predict(x[va], co, mu, scale); called = score.argmax(1)
        confusion = np.zeros((2, 2), int)
        for true, guess in zip(labels[va], called): confusion[true, guess] += 1
        save_npz(out/(name+'.npz'), coefficients=co, mean=mu, scale=scale, score=score, labels=labels[va], predicted=called)
        reports[name] = {'accuracy': float((called == labels[va]).mean()), 'confusion': confusion.tolist()}
    reports['majority_train_accuracy'] = float((labels[va] == np.bincount(labels[tr]).argmax()).mean())
    reports.update({'classes': ['0h_dauer_condition', 'released_6_or_24h_condition'], 'label_source': 'experimental_condition_not_independent_biological_state',
        'encoder_sha256': sha256(run/'state_base/best.pt'), 'science_status': 'unvalidated', 'fit_kind': 'regression_fit'})
    write_json(out/'result.json', reports)


def waves_task(run):
    run = Path(run); out = run/'waves'; b = ObservationBundle.load(run/'clock_reference/bundle')
    x, genes = gene_data(run/'data/dauer'); tr, va = b.indices('train'), b.indices('validation')
    ref = ExitReference.load(run/'clock_reference'); readout = set(ref.metadata['readout_genes'])
    tf = set(read_json(run/'data/dauer/tf_identity.json')['genes']); context = object_hash(b.context)
    model_clock = StatePredictor.load(run/'state_base').predict(b.subset(va))['clock']
    reports = {}
    for kind, values, ids in [('gene', x, genes), ('TF', x[:, [i for i, g in enumerate(genes) if g in tf]], [g for g in genes if g in tf]),
                              ('programme', b.values, b.feature_ids)]:
        if not ids: raise ValueError('No actual features for '+kind)
        mask = b.mask if kind == 'programme' else np.ones_like(values, bool)
        w = WaveReference.fit(b.clock, values, mask, ids, b.rows, b.clock_reference_id, context, feature_kind=kind, degree=1)
        folder = out/kind; w.save(folder)
        result = w.residual(b.clock[va], values[va], mask[va], b.clock_reference_id, context)
        combined = w.residual(model_clock, values[va], mask[va], b.clock_reference_id, context)
        baseline = np.broadcast_to(values[tr].mean(0), values[va].shape)
        save_npz(folder/'validation.npz', observed=values[va], reference_expected=result['expected'], residual=result['residual'],
            supported=result['residual_mask'], combined_expected=combined['expected'], combined_residual=combined['residual'],
            observed_reference_coordinate=b.clock[va], predicted_coordinate=model_clock, baseline=baseline)
        r = {'reference_coordinate': metric(result['expected'], values[va], result['residual_mask']),
             'training_mean_same_mask': metric(baseline, values[va], result['residual_mask']),
             'model_coordinate': metric(combined['expected'], values[va], combined['residual_mask']),
             'supported_values': int(result['residual_mask'].sum()), 'combined_supported_values': int(combined['residual_mask'].sum()),
             'outside_range_is_unavailable': True, 'features': len(ids)}
        common=result['residual_mask'] & combined['residual_mask']
        r['comparison_on_identical_supported_values']={
            'reference':metric(result['expected'],values[va],common),
            'upstream_clock':metric(combined['expected'],values[va],common),
            'training_mean':metric(baseline,values[va],common)}
        if kind == 'gene':
            panel = np.array([g in readout for g in ids])
            r['disjoint_axis_gene_readout'] = metric(result['expected'][:, panel], values[va][:, panel], result['residual_mask'][:, panel])
            r['readout_limit'] = 'Disjoint axis gene membership only; library normalisation and programme rankings still share information'
        # Actual observed amplitude and coarse trend; 3 elapsed times cannot validate fine peaks or lags.
        write_json(folder/'shapes.json', [{'id': g, 'train_raw_amplitude': float(np.ptp(values[tr, j])),
            'linear_direction': 'up' if w.coefficients[1, j] > 0 else 'down' if w.coefficients[1, j] < 0 else 'flat',
            'fine_peak_or_lag': 'not_identifiable'} for j, g in enumerate(ids)])
        reports[kind] = r
    reports.update({'regulon': 'blocked_data: GO membership is not a TF-target regulon', 'degree': 1,
        'clock_reference_kind': 'reference_derived', 'encoder_sha256': sha256(run/'state_base/best.pt'), 'science_status': 'unvalidated'})
    write_json(out/'result.json', reports)


def response_dataset(run, mode, variant='observed'):
    run = Path(run); b = ObservationBundle.load(run/'data/ard' if mode == 'endpoint' else run/'clock_reference/bundle')
    current_values = b.values
    representation = object_hash(b.scope)
    if variant != 'observed':
        state = 'state_ard' if mode == 'endpoint' else variant
        p = StatePredictor.load(run/state); current_values = p.predict(b)['programme']
        representation = sha256(run/state/'best.pt')
    current, action, targets, rows, elapsed = [], [], [], [], []
    for i, r in enumerate(b.rows):
        if mode == 'endpoint':
            if r['genotype_code'] == 'N2': continue
            candidates = [j for j, c in enumerate(b.rows) if c['genotype_code'] == 'N2' and c['condition'] == r['condition'] and c['replicate_block'] == r['replicate_block']]
            a = [float(r['genotype_code'] in {'hlh30', 'daf1_hlh30'}), float(r['genotype_code'] in {'daf1', 'daf1_hlh30'}), float(r['condition'] == 'REC')]
        else:
            if r['elapsed_hours_since_release'] == 0: continue
            candidates = [j for j, c in enumerate(b.rows) if c['elapsed_hours_since_release'] == 0 and c['history_days'] == r['history_days'] and c['replicate_block'] == r['replicate_block']]
            a = [np.log1p(r['history_days'])]
        if len(candidates) != 1: raise ValueError('Missing or ambiguous group control')
        j = candidates[0]; c = b.rows[j]
        if c['split'] != r['split']: raise ValueError('Shared control crosses split')
        # Endpoint target is measured mutant-WT change. Genotype is a contrast, not a new acute KO.
        target = b.values[i]-b.values[j] if mode == 'endpoint' else b.values[i]
        current.append(current_values[j]); action.append(a); targets.append(target)
        rows.append({**r, 'observation_id': mode+':'+r['observation_id'],
            'control_or_initial_ids': [c['observation_id']], 'outcome_ids': [r['observation_id']],
            'link_ids': list(set(r['link_ids']+c['link_ids']+[c['observation_id'], r['observation_id']])),
            'pairing_kind': 'condition_matched_pooled_samples_not_tracked_individuals',
            'new_acute_ko': False, 'input_contains_future_information': False})
        if mode == 'transition': elapsed.append(r['elapsed_hours_since_release'])
    d = ResponseDataset(np.array(current), np.array(action), np.array(targets), np.ones_like(targets, bool), rows,
        mode, representation, b.feature_ids, ['hlh30_genotype_contrast', 'daf1_genotype_contrast', 'refed_background'] if mode == 'endpoint' else ['log1p_dauer_history_days'],
        b.feature_ids, object_hash(b.context), np.array(elapsed) if mode == 'transition' else None,
        'hours' if mode == 'transition' else None).validate()
    return d


def response_task(run, mode, variant='observed', output=None):
    run = Path(run); suffix = '' if variant == 'observed' else '_'+variant
    out = Path(output) if output is not None else run/(mode+suffix)
    d = response_dataset(run, mode, variant); d.save(out/'dataset')
    model = ResponseRegressor(alpha=1).fit(d); model.save(out)
    va = np.array([i for i, r in enumerate(d.rows) if r['split'] == 'validation']); tr = np.array([i for i, r in enumerate(d.rows) if r['split'] == 'train'])
    elapsed = None if d.elapsed is None else d.elapsed[va]
    pred = ResponseRegressor.load(out).predict(d.current[va], d.action[va], d.scope, elapsed)
    baseline = np.zeros_like(pred) if mode == 'endpoint' else d.current[va]
    mask = d.target_mask[va] & np.broadcast_to(model.target_available, pred.shape)
    report = {'model': metric(pred, d.target[va], mask), 'zero_effect_or_last_state': metric(baseline, d.target[va], mask),
        'scope': d.scope, 'n_train_contrasts': len(tr), 'n_validation_contrasts': len(va),
        'shared_control_contrasts_not_independent': True, 'input_route': variant,
        'science_status': 'unvalidated', 'fit_kind': 'regression_fit'}
    if mode == 'transition':
        coef, mu, scale = ridge_fit(np.column_stack([d.elapsed[tr], d.action[tr]]), d.target[tr])
        time_pred = ridge_predict(np.column_stack([d.elapsed[va], d.action[va]]), coef, mu, scale)
        report['time_history_only_baseline'] = metric(time_pred, d.target[va], mask)
        save_npz(out/'time_baseline.npz', coefficients=coef, mean=mu, scale=scale, prediction=time_pred)
    truth_delta = d.target[va]-baseline; pred_delta = pred-baseline
    report['delta'] = metric(pred_delta, truth_delta, mask)
    report['direction_agreement_nonzero_truth'] = float((np.sign(pred_delta[mask]) == np.sign(truth_delta[mask])).mean())
    save_npz(out/'validation.npz', predicted=pred, observed=d.target[va], baseline=baseline, mask=mask,
        current=d.current[va], action=d.action[va], **({'elapsed': elapsed} if elapsed is not None else {}))
    write_json(out/'result.json', report)


def reload_check(run, task):
    """Called by a second Python process; exercises real saved weights and input contracts."""
    run = Path(run); out = run/task
    if task.startswith('state_') and task != 'state_readout':
        b = ObservationBundle.load(run/'data/ard' if task == 'state_ard' else run/'clock_reference/bundle')
        p = StatePredictor.load(out).predict(b)
        with np.load(out/'clean_predictions.npz', allow_pickle=False) as a:
            delta = float(np.max(np.abs(p['programme']-a['programme'])))
    elif task.startswith('endpoint') or task.startswith('transition') or task == 'functional':
        d = ResponseDataset.load(out/'dataset'); idx = [i for i, r in enumerate(d.rows) if r['split'] == 'validation']
        p = ResponseRegressor.load(out).predict(d.current[idx], d.action[idx], d.scope, None if d.elapsed is None else d.elapsed[idx])
        with np.load(out/'validation.npz', allow_pickle=False) as a: delta = float(np.nanmax(np.abs(p-a['predicted'])))
    elif task == 'clock_reference':
        x, g = gene_data(run/'data/dauer'); p = ExitReference.load(out).predict(x, g)
        with np.load(out/'reference_scores.npz', allow_pickle=False) as a: delta = float(np.max(np.abs(p-a['scores'])))
    elif task == 'waves':
        b = ObservationBundle.load(run/'clock_reference/bundle'); idx = b.indices('validation'); delta = 0.
        for kind in ('gene', 'TF', 'programme'):
            w = WaveReference.load(out/kind); pred = w.predict(b.clock[idx], w.coordinate_id, w.context_id)['expected']
            with np.load(out/kind/'validation.npz', allow_pickle=False) as a:
                if not np.array_equal(np.isnan(pred), np.isnan(a['reference_expected'])): raise RuntimeError('Wave support changed')
                delta = max(delta, float(np.nanmax(np.abs(pred-a['reference_expected']))))
    elif task == 'state_readout':
        b = ObservationBundle.load(run/'clock_reference/bundle'); idx = b.indices('validation'); delta = 0.
        z = StatePredictor.load(run/'state_base').predict(b.subset(idx))['latent']
        for name, x in [('latent', z), ('observed_programme', b.values[idx])]:
            with np.load(out/(name+'.npz'), allow_pickle=False) as a:
                delta = max(delta, float(np.max(np.abs(ridge_predict(x, a['coefficients'], a['mean'], a['scale'])-a['score']))))
    else: raise ValueError('No reload implementation for '+task)
    if delta > 1e-5: raise RuntimeError('Reload prediction mismatch '+str(delta))
    write_json(out/'reload.json', {'status': 'passed', 'new_process': True, 'max_abs_delta': delta})


def functional_task(run, source):
    run = Path(run); out = run/'functional'; m = read_json(source)
    rows, x, actions, y = [], [], [], []
    for r in m['records']:
        if r['split'] not in {'train', 'validation'}: raise ValueError('Functional holdout included')
        if sum(r['counts'].values()) != r['total'] or r['total'] <= 0: raise ValueError('Invalid functional counts')
        key = f"Fig1B:D{r['history_days']}:{r['hours_after_release']}h:R{r['replicate_block']}"
        rows.append({'observation_id': key, 'study_family': 'PMID41104926_functional', 'biological_unit': key,
            'split': r['split'], 'origin': 'public', 'link_ids': [f"PMID41104926:functional_block:{r['replicate_block']}", key],
            'control_or_initial_ids': [key], 'outcome_ids': [key], 'source_cells': r['source_cells'],
            'pairing_kind': 'assayed_group_context_not_expression_pair'})
        x.append([np.log1p(r['history_days'])]); actions.append([r['hours_after_release']])
        y.append([r['counts']['young_adult']/r['total']])
    d = ResponseDataset(np.array(x), np.array(actions), np.array(y), np.ones_like(y, bool), rows, 'functional',
        object_hash({'source': sha256(source), 'representation': 'history_and_protocol_hours_only'}), ['log1p_history_days'],
        ['hours_after_release'], [m['endpoint']], 'daf2_Fig1B_group_context', endpoint=m['endpoint'], protocol_id=m['protocol']).validate()
    d.save(out/'dataset'); model = ResponseRegressor().fit(d); model.save(out)
    tr = [i for i, r in enumerate(rows) if r['split'] == 'train']; va = [i for i, r in enumerate(rows) if r['split'] == 'validation']
    pred = model.predict(d.current[va], d.action[va], d.scope); baseline = np.broadcast_to(d.target[tr].mean(0), pred.shape)
    save_npz(out/'validation.npz', predicted=pred, observed=d.target[va], baseline=baseline, mask=d.target_mask[va])
    write_json(out/'result.json', {'model': metric(pred, d.target[va]), 'training_mean': metric(baseline, d.target[va]),
        'scope': d.scope, 'n_train_groups': len(tr), 'n_validation_groups': len(va), 'fit_kind': 'regression_fit',
        'science_status': 'unvalidated', 'expression_to_function': 'blocked_data_no_cohort_matching',
        'depth': 'unavailable_no_depth_supervision', 'limitations': m['limitations'],
        'predictions_outside_fraction_range': int(((pred<0)|(pred>1)).sum()), 'predictions_clipped': False,
        'source_file_sha256': m['source_sha256'], 'curated_sha256': sha256(source)})
