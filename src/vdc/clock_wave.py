"""CW1: target-isolated observations and an explicit, abstaining linear wave locator.

The coordinate is a train-anchor-derived molecular reference, not clock truth.
All prediction inputs are counts, a declared identity and fixed programme members.
"""
from pathlib import Path
import hashlib
import numpy as np
from .io import read_json, write_json, save_npz, sha256, object_hash
from .application_data import align_counts, non_target_input
from .application import predict_ridge
from .observation import normalise_expression
from .pk2_numeric import dual_ridge, metrics, dynamic_ratio
from .metrics import study_unit_weights
from .admission import audit_internal_task
from .contracts import audit_rows


class NotIdentifiable(ValueError):
    """An explicit scientific/data support limitation, not a swallowed software error."""


def hidden_partition(genes):
    return {g for g in genes if int(hashlib.sha256(g.encode()).hexdigest()[:8], 16) % 5 == 0}


def inputs(counts, genes, definitions, extra_hidden=()):
    x = align_counts(counts, genes, genes)
    if not np.equal(x, np.rint(x)).all(): raise ValueError('CW1 requires exact integer counts')
    hidden = hidden_partition(genes) | set(extra_hidden)
    y, s = non_target_input(x, genes, definitions, hidden)
    index = {g: i for i, g in enumerate(genes)}
    amplitude = np.zeros_like(s['values'], dtype=float)
    for k, p in enumerate(definitions):
        members = [(index[g], w) for g, w in p['members'].items() if g in index and g not in hidden]
        if members:
            j, w = zip(*members)
            amplitude[:, k] = y[:, j] @ np.asarray(w) / sum(abs(v) for v in w)
    return y, s, amplitude


def identities(rows):
    return [r.get('coarse_identity', 'all') for r in rows]


def weighted_mean(x, rows):
    return np.average(x, axis=0, weights=study_unit_weights(rows))


def anchor_axis(y, genes, rows):
    """Fit a safe axis within each identity; never copy a pool label to all its cells."""
    out = np.full(len(rows), np.nan); info = {}
    hidden = hidden_partition(genes)
    for typ in sorted(set(identities(rows))):
        ids = [i for i, r in enumerate(rows) if identities([r])[0] == typ]
        early = [i for i in ids if rows[i]['split'] == 'train' and rows[i]['condition'] == 'Early Diapause']
        end = [i for i in ids if rows[i]['split'] == 'train' and rows[i]['condition'] in {'Developing', 'Developing Day 7'}]
        if not early or not end:
            info[typ] = {'status': 'unsupported_missing_train_anchors'}; continue
        fit = early + end
        variance = y[fit].var(0)
        eligible = np.array([i for i, g in enumerate(genes) if g not in hidden and variance[i] > 1e-8])
        use = eligible[np.argsort(-variance[eligible], kind='stable')[:512]]
        if len(use) < 8:
            info[typ] = {'status': 'unsupported_fewer_than_eight_axis_genes'}; continue
        mu = weighted_mean(y[fit][:, use], [rows[i] for i in fit])
        sd = np.maximum(y[fit][:, use].std(0), .001)
        z = (y[:, use] - mu) / sd
        origin = weighted_mean(z[early], [rows[i] for i in early])
        delta = weighted_mean(z[end], [rows[i] for i in end]) - origin
        if delta @ delta < 1e-8:
            info[typ] = {'status': 'unsupported_coincident_anchors'}; continue
        direction = delta / (delta @ delta)
        out[ids] = (z[ids] - origin) @ direction
        info[typ] = {'status': 'exploratory_reference', 'indices': use.tolist(), 'mean': mu.tolist(),
                     'scale': sd.tolist(), 'origin': origin.tolist(), 'direction': direction.tolist(),
                     'fit_ids': [rows[i]['observation_id'] for i in fit],
                     'anchor_units': [len({rows[i]['biological_unit'] for i in j}) for j in (early, end)]}
    if not np.isfinite(out).any():
        raise NotIdentifiable('No identity has both train Early and Developing anchors')
    return out, info


class ClockWave:
    def _features(self, counts):
        y, score, amp = inputs(counts, self.meta['genes'], self.meta['definitions'], self.meta['config'].get('extra_hidden', []))
        x = score['values'].astype(float)
        if self.meta['representation'] != 'C0': x = x - self.arrays['early_rank']
        mask = score['mask'].copy(); coverage = score['coverage'].copy()
        if self.meta['representation'] == 'C2':
            x = np.column_stack([x, amp - self.arrays['early_amp']])
            mask = np.column_stack([mask, mask]); coverage = np.column_stack([coverage, coverage])
        return y, x, mask, coverage

    @classmethod
    def fit(cls, counts, genes, definitions, rows, representation, config, semantic_graph=None):
        if representation not in {'C0', 'C1', 'C2'}: raise ValueError('Unknown representation')
        if any(r['split'] not in {'train', 'validation'} for r in rows): raise ValueError('Development rows only')
        audit_rows(rows, 'unit_holdout'); audit_internal_task(rows, 'clock'); audit_internal_task(rows, 'waves')
        obj = cls(); obj.arrays = {}
        obj.meta = {'genes': list(genes), 'definitions': definitions, 'representation': representation,
                    'config': config, 'hidden_removed': sorted(hidden_partition(genes) | set(config.get('extra_hidden', []))),
                    'coordinate': 'identity_specific_train_Early_to_Developing_reference_not_clock_truth',
                    'fit_ids': [r['observation_id'] for r in rows if r['split'] == 'train'],
                    'input_rule': 'hidden targets absent from ranks, denominator, amplitude, markers and axis',
                    'future_prediction': 'unavailable', 'depth': 'unavailable', 'uncertainty': 'uncalibrated'}
        obj.meta['target_isolation_scope'] = ('conditional_on_frozen_source_annotation_aggregation; not end-to-end cell reannotation independence'
                                             if any('coarse_identity' in r for r in rows) else 'raw_count_input_paths')
        y, s, amp = inputs(counts, genes, definitions, config.get('extra_hidden', []))
        train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
        early = np.array([i for i in train if rows[i]['condition'] == 'Early Diapause'])
        if not len(early): raise NotIdentifiable('No train Early reference')
        obj.arrays['early_rank'] = weighted_mean(s['values'][early], [rows[i] for i in early])
        obj.arrays['early_amp'] = weighted_mean(amp[early], [rows[i] for i in early])
        _, x, mask, _ = obj._features(counts)
        tau, axes = anchor_axis(y, genes, rows); obj.meta['axes'] = axes
        types = sorted(t for t, a in axes.items() if a['status'] == 'exploratory_reference')
        obj.meta['types'] = types
        fit = np.array([i for i in train if np.isfinite(tau[i]) and rows[i].get('reference_eligible', False)])
        if len(fit) < 3 or np.ptp(tau[fit]) < .01: raise NotIdentifiable('Too few eligible wave observations')
        design = np.array([[float(identities([rows[i]])[0] == t) for t in types] + [tau[i]] for i in fit])
        weight = study_unit_weights([rows[i] for i in fit]).astype(float); weight /= weight.mean()
        coef = np.zeros((len(types) + 1, x.shape[1])); valid = np.zeros(x.shape[1], bool)
        for j in range(x.shape[1]):
            used = mask[fit, j]
            if used.sum() < 3 or np.ptp(tau[fit][used]) < .01: continue
            d = design[used]; w = weight[used]
            coef[:, j] = np.linalg.solve(d.T @ (w[:, None] * d) + .001 * np.eye(d.shape[1]), d.T @ (w * x[fit[used], j]))
            valid[j] = True
        if semantic_graph is not None:
            graph = np.asarray(semantic_graph, float); p = len(definitions)
            if graph.shape != (p, p) or not np.isfinite(graph).all() or not np.allclose(graph, graph.T) or (graph < 0).any():
                raise ValueError('Semantic graph must be finite nonnegative symmetric programme graph')
            lap = np.diag(graph.sum(1)) - graph
            smooth = np.linalg.inv(np.eye(p) + config['semantic_strength'] * lap)
            for start in range(0, x.shape[1], p): coef[:, start:start+p] = coef[:, start:start+p] @ smooth
            obj.arrays['semantic_graph'] = graph
        residual = np.where(mask[fit], x[fit] - design @ coef, 0)
        scale = np.maximum(np.sqrt(np.average(residual**2, axis=0, weights=weight)), .03)
        corr = (residual / scale).T @ ((residual / scale) * weight[:, None]) / weight.sum()
        shrink = config['covariance_shrinkage']; corr = (1 - shrink) * corr + shrink * np.eye(len(scale))
        obj.arrays.update(coef=coef, valid=valid, noise_scale=scale, correlation=corr,
                          lower=np.array([min(tau[i] for i in fit if identities([rows[i]])[0] == t) for t in types]),
                          upper=np.array([max(tau[i] for i in fit if identities([rows[i]])[0] == t) for t in types]))
        query = obj.predict(counts, identities(rows))
        # Targets are relative full-library expression; input normalization excludes them.
        full = normalise_expression(counts, np.ones_like(counts, bool), 'counts')
        target_panel = set(config.get('target_panel', sorted(hidden_partition(genes))))
        if not target_panel.issubset(set(obj.meta['hidden_removed'])): raise ValueError('Readout targets remain visible')
        eligible = [i for i, g in enumerate(genes) if g in target_panel and full[train, i].var() > 1e-8]
        target_ids = sorted(eligible, key=lambda j: (-full[train, j].var(), j))[:config['hidden_gene_limit']]
        if not target_ids: raise NotIdentifiable('No variable train hidden targets')
        obj.meta['target_indices'] = target_ids; obj.meta['target_genes'] = [genes[i] for i in target_ids]
        obj.arrays['target_mean'] = weighted_mean(full[train][:, target_ids], [rows[i] for i in train])
        obj.arrays['identity_target_mean'] = np.array([weighted_mean(full[[i for i in train if identities([rows[i]])[0]==t]][:,target_ids],
                         [rows[i] for i in train if identities([rows[i]])[0]==t]) for t in types])
        direct_x = obj.direct_features(x, mask, identities(rows))
        direct = dual_ridge(direct_x[train], full[train][:, target_ids], [rows[i] for i in train], config['readout_alpha'])
        obj.direct = direct.parameters
        good = np.array([i for i in train if np.isfinite(query['clock'][i])])
        if len(good) < 3: raise NotIdentifiable('Too few supported training locations for independent readout')
        d = obj.clock_design(query['clock'][good], [identities(rows)[i] for i in good])
        obj.readout = dual_ridge(d, full[good][:, target_ids], [rows[i] for i in good], config['readout_alpha']).parameters
        obj.meta['readout_fit_ids'] = [rows[i]['observation_id'] for i in good]
        obj.meta['reference_clock_fit_ids'] = [rows[i]['observation_id'] for i in fit]
        obj.meta['amplitude'] = 'local relative log abundance; not absolute RNA per cell'
        obj.meta['semantic_graph_hash'] = object_hash(semantic_graph.tolist()) if semantic_graph is not None else None
        return obj

    def direct_features(self, x, mask, types):
        # The numerical baseline receives the SAME identity evidence as the locator.
        type_columns=np.array([[float(t==v) for v in self.meta['types']] for t in types])
        return np.column_stack([np.where(mask, x, 0), mask, type_columns])

    def clock_design(self, tau, types):
        return np.array([[float(t == v) for v in self.meta['types']] + [float(s)] for s, t in zip(tau, types)])

    def predict(self, counts, types, programme_keep=None):
        _, x, mask, coverage = self._features(counts)
        if len(types) != len(x): raise ValueError('Identity row count mismatch')
        if programme_keep is not None:
            keep = np.asarray(programme_keep, bool)
            if keep.shape != (len(x), len(self.meta['definitions'])): raise ValueError('Programme mask shape')
            mask &= np.tile(keep, (1, 2 if self.meta['representation'] == 'C2' else 1))
        a = self.arrays; p = len(self.meta['definitions']); cfg = self.meta['config']
        clocks = np.full(len(x), np.nan); expected = np.full_like(x, np.nan); supported = np.zeros_like(mask)
        quality = []; diagnostics = []
        for i, typ in enumerate(types):
            if typ not in self.meta['types']:
                quality.append('unsupported_identity'); diagnostics.append(None); continue
            k = self.meta['types'].index(typ)
            used = mask[i] & a['valid'] & (coverage[i] >= cfg['minimum_coverage'])
            # A rank and amplitude of the same programme are not two independent programmes.
            programmes = used[:p] | used[p:] if len(used) == 2*p else used
            if programmes.sum() < cfg['minimum_programmes']:
                quality.append('unsupported_coverage'); diagnostics.append(None); continue
            j = np.flatnonzero(used); b = a['coef'][-1, j] / a['noise_scale'][j]
            d = (x[i, j] - a['coef'][k, j]) / a['noise_scale'][j]
            eig, vec = np.linalg.eigh(a['correlation'][np.ix_(j, j)])
            whiten = (vec / np.sqrt(np.maximum(eig, .01))) @ vec.T
            b = whiten @ b; d = whiten @ d
            if b @ b < 1e-6:
                quality.append('uncertain_flat_reference'); diagnostics.append(None); continue
            t = float(b @ d / (b @ b))
            for _ in range(20):
                error = d - b * t
                w = np.minimum(1., cfg['huber_delta'] / np.maximum(abs(error), 1e-12))
                nxt = float((w * b) @ d / ((w * b) @ b))
                if abs(nxt - t) < 1e-8: break
                t = nxt
            clocks[i] = t; expected[i] = a['coef'][k] + t * a['coef'][-1]
            inside = a['lower'][k] - 1e-8 <= t <= a['upper'][k] + 1e-8
            supported[i] = mask[i] & a['valid'] & inside
            quality.append('located' if inside else 'outside_reference_range')
            diagnostics.append(float(np.mean(np.minimum((d-b*t)**2, 25))))
        return {'clock': clocks, 'observed': x, 'expected_linear_extrapolation': expected,
                'reference_expected': np.where(supported, expected, np.nan),
                'residual': np.where(supported, x-expected, np.nan), 'residual_mask': supported,
                'input_mask': mask, 'coverage': coverage, 'status': quality, 'robust_distance': diagnostics}

    def hidden_predictions(self, counts, types, programme_keep=None):
        q = self.predict(counts, types, programme_keep)
        _, x, m, _ = self._features(counts)
        m = q['input_mask']
        def fn(params, xx): return (xx-params['mean'])/params['scale'] @ params['coefficients'] + params['target_mean']
        direct = fn(self.direct, self.direct_features(x, m, types))
        got = np.full_like(direct, np.nan); valid = np.isfinite(q['clock'])
        if valid.any(): got[valid] = fn(self.readout, self.clock_design(q['clock'][valid], np.array(types)[valid]))
        type_mean=np.full_like(direct,np.nan)
        for i,t in enumerate(types):
            if t in self.meta['types']:type_mean[i]=self.arrays['identity_target_mean'][self.meta['types'].index(t)]
        q.update(hidden_prediction=got, hidden_direct_ridge=direct,hidden_identity_mean=type_mean,
                 hidden_mean=np.broadcast_to(self.arrays['target_mean'], direct.shape).copy())
        return q

    def save(self, folder):
        folder = Path(folder)
        save_npz(folder/'arrays.npz', **self.arrays); save_npz(folder/'direct.npz', **self.direct); save_npz(folder/'readout.npz', **self.readout)
        write_json(folder/'model.json', {**self.meta, 'files': {n: sha256(folder/n) for n in ('arrays.npz','direct.npz','readout.npz')}})

    @classmethod
    def load(cls, folder):
        folder = Path(folder); obj = cls(); obj.meta = read_json(folder/'model.json')
        for n, h in obj.meta['files'].items():
            if sha256(folder/n) != h: raise ValueError('Clock wave artifact changed')
        for attr, name in [('arrays','arrays.npz'),('direct','direct.npz'),('readout','readout.npz')]:
            with np.load(folder/name, allow_pickle=False) as z: setattr(obj, attr, {k:z[k].copy() for k in z.files})
        return obj


def evaluate(model, counts, rows, out, programme_keep=None, truth_counts=None, types=None):
    out = Path(out); types = identities(rows) if types is None else types
    q = model.hidden_predictions(counts, types, programme_keep)
    target_counts = counts if truth_counts is None else truth_counts
    target = normalise_expression(target_counts, np.ones_like(target_counts, bool), 'counts')[:, model.meta['target_indices']]
    finite = np.isfinite(q['hidden_prediction']); common = finite & np.array([s=='located' for s in q['status']])[:, None]
    result = {'model_all_finite_including_extrapolation': metrics(q['hidden_prediction'], target, finite, rows),
              'model_in_reference': metrics(q['hidden_prediction'], target, common, rows),
              'ridge_same_support': metrics(q['hidden_direct_ridge'], target, common, rows),
              'mean_same_support': metrics(q['hidden_mean'], target, common, rows),
              'identity_mean_same_support': metrics(q['hidden_identity_mean'], target, common, rows),
              'ridge_all': metrics(q['hidden_direct_ridge'], target, np.ones_like(target, bool), rows),
              'dynamic': dynamic_ratio(q['hidden_prediction'], target, common),
              'status_counts': {s:q['status'].count(s) for s in set(q['status'])},
              'independent_units': len({r['biological_unit'] for r in rows}),
              'reference_coordinate_not_gold': True, 'depth': None, 'future_prediction': None}
    result['target_isolation_scope'] = model.meta['target_isolation_scope']
    save_npz(out/'predictions.npz', **{k:v for k,v in q.items() if isinstance(v,np.ndarray)}, hidden_target=target)
    write_json(out/'rows.json', rows); write_json(out/'query_status.json', {'status':q['status'], 'robust_distance':q['robust_distance'], 'identity':types})
    write_json(out/'metrics.json', result)
    return result
