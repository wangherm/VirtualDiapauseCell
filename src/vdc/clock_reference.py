"""A train-only local molecular coordinate, explicitly NOT independent clock truth."""
from pathlib import Path
import hashlib
import numpy as np
from .io import write_json, read_json, save_npz, sha256


class ExitReference:
    reference_id = 'GSE288723_dauer_0h_to_observed_24h_v1'

    def fit(self, expression, genes, rows, max_genes=512):
        x = np.asarray(expression, float)
        if x.shape != (len(rows), len(genes)) or not np.isfinite(x).all():
            raise ValueError('Finite expression and exact gene order required')
        train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
        # Preassigned readout panel, independent of measured variance or evaluation outcome.
        readout = np.array([int(hashlib.sha256(g.encode()).hexdigest()[:8], 16) % 5 == 0 for g in genes])
        variance = x[train].var(0)
        eligible = np.flatnonzero(~readout & (variance > 1e-8))
        self.indices = eligible[np.argsort(-variance[eligible], kind='stable')[:max_genes]]
        if len(self.indices) < 8:
            raise ValueError('not_identifiable: insufficient variable axis genes')
        self.mean = x[train][:, self.indices].mean(0)
        self.scale = np.maximum(x[train][:, self.indices].std(0), .001)
        z = (x[:, self.indices]-self.mean)/self.scale
        anchors = {}
        for hour in (0, 24):
            groups = []
            for history in (1, 4, 15, 30):
                idx = [i for i in train if rows[i]['history_days'] == history and rows[i]['elapsed_hours_since_release'] == hour]
                if not idx: raise ValueError('Missing train anchor history/time')
                groups.append(z[idx].mean(0))
            anchors[hour] = np.mean(groups, axis=0)
        self.origin = anchors[0]
        delta = anchors[24]-self.origin
        if np.dot(delta, delta) < 1e-8: raise ValueError('not_identifiable: anchors coincide')
        self.direction = delta/np.dot(delta, delta)
        self.metadata = {'reference_id': self.reference_id, 'label_source': 'reference_derived',
            'genes': list(genes), 'axis_genes': [genes[i] for i in self.indices],
            'readout_genes': [g for g, keep in zip(genes, readout) if keep],
            'fit_ids': [rows[i]['observation_id'] for i in train], 'history_weighting': 'equal_1_4_15_30',
            'limitation': '0 and observed 24h anchors; not complete recovery, not functional depth; shared library normalisation and programme membership remain'}
        return self

    def predict(self, expression, genes):
        if list(genes) != self.metadata['genes']: raise ValueError('Clock gene order differs')
        x = np.asarray(expression, float)
        if x.ndim != 2 or not np.isfinite(x).all(): raise ValueError('Invalid clock inputs')
        return ((x[:, self.indices]-self.mean)/self.scale-self.origin) @ self.direction

    def save(self, path):
        p = Path(path)
        save_npz(p/'axis.npz', indices=self.indices, mean=self.mean, scale=self.scale, origin=self.origin, direction=self.direction)
        write_json(p/'axis.json', {**self.metadata, 'arrays_sha256': sha256(p/'axis.npz')})

    @classmethod
    def load(cls, path):
        p = Path(path); obj = cls(); obj.metadata = read_json(p/'axis.json')
        if sha256(p/'axis.npz') != obj.metadata['arrays_sha256']: raise ValueError('Axis checksum mismatch')
        with np.load(p/'axis.npz', allow_pickle=False) as a:
            for k in a.files: setattr(obj, k, a[k].copy())
        return obj
