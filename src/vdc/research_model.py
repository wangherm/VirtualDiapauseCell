"""Train-only numerical graph executor for AI-designed experiments.

JSON graphs are data, never Python evaluated with exec/eval. The optional Docker
backend in research.py is the separate route for unrestricted candidate code.
"""
import numpy as np
from .io import write_json, read_json, save_npz

SOURCES = ('rank', 'amplitude', 'genes', 'identity', 'clock', 'clock_available', 'coverage', 'programme_available')
OPS = ('concat', 'standardize', 'pca', 'square', 'tanh', 'interaction', 'rbf', 'scale')


def validate_graph(spec):
    if set(spec) != {'nodes', 'output', 'alpha'}:
        raise ValueError('Graph requires only nodes, output, alpha')
    if not isinstance(spec['nodes'], list) or not 1 <= len(spec['nodes']) <= 16:
        raise ValueError('Graph must have 1..16 nodes')
    alpha = spec['alpha']
    if not isinstance(alpha, (int, float)) or not np.isfinite(alpha) or not 1e-5 <= alpha <= 1e5:
        raise ValueError('alpha outside [1e-5,1e5]')
    seen = set(SOURCES)
    fields = {'concat': set(), 'standardize': set(), 'pca': {'components'},
              'square': set(), 'tanh': set(), 'interaction': set(),
              'rbf': {'gamma'}, 'scale': {'factor'}}
    for node in spec['nodes']:
        if not isinstance(node, dict) or node.get('op') not in OPS:
            raise ValueError('Unsupported graph operation')
        if set(node) - ({'id', 'op', 'inputs'} | fields[node['op']]):
            raise ValueError('Unknown graph field')
        name = node.get('id')
        if not isinstance(name, str) or not name.isidentifier() or name in seen:
            raise ValueError('Duplicate or invalid node ID')
        ins = node.get('inputs')
        if not isinstance(ins, list) or not ins or any(i not in seen for i in ins):
            raise ValueError('Inputs must reference existing nodes')
        if node['op'] == 'interaction' and len(ins) != 2:
            raise ValueError('interaction needs two inputs')
        if node['op'] not in {'concat', 'interaction'} and len(ins) != 1:
            raise ValueError('Unary operation needs one input')
        if node['op'] == 'pca' and (type(node.get('components')) is not int or not 1 <= node['components'] <= 32):
            raise ValueError('PCA components must be 1..32')
        for field in ('gamma', 'factor'):
            if field in fields[node['op']]:
                v = node.get(field)
                if not isinstance(v, (int, float)) or not np.isfinite(v) or not 1e-4 <= v <= 100:
                    raise ValueError('Invalid graph scalar')
        seen.add(name)
    if spec['output'] not in seen:
        raise ValueError('Missing graph output')
    return spec


class GraphModel:
    def __init__(self, spec):
        self.spec = validate_graph(spec)
        self.params = {}

    def features(self, data, fitting=False, weights=None):
        values = {k: np.asarray(data[k], float) for k in SOURCES}
        for node in self.spec['nodes']:
            name, op = node['id'], node['op']
            xs = [values[k] for k in node['inputs']]
            x = xs[0]
            if op == 'concat':
                y = np.column_stack(xs)
            elif op == 'standardize':
                if fitting:
                    mu = np.average(x, axis=0, weights=weights)
                    sd = np.sqrt(np.average((x-mu)**2, axis=0, weights=weights))
                    self.params[name+'_mean'], self.params[name+'_scale'] = mu, np.maximum(sd, 1e-6)
                y = (x-self.params[name+'_mean'])/self.params[name+'_scale']
            elif op == 'pca':
                if fitting:
                    mu = np.average(x, axis=0, weights=weights)
                    _, _, vt = np.linalg.svd((x-mu)*np.sqrt(weights[:, None]), full_matrices=False)
                    n = min(node['components'], x.shape[1], max(1, len(x)-1))
                    self.params[name+'_mean'], self.params[name+'_basis'] = mu, vt[:n].T
                y = (x-self.params[name+'_mean']) @ self.params[name+'_basis']
            elif op == 'interaction':
                if xs[0].shape[1]*xs[1].shape[1] > 4096:
                    raise ValueError('Interaction exceeds 4096 features')
                y = (xs[0][:, :, None]*xs[1][:, None, :]).reshape(len(x), -1)
            elif op == 'rbf':
                if fitting:
                    self.params[name+'_centres'] = x.copy()
                centres = self.params[name+'_centres']
                distance = np.maximum((x*x).sum(1)[:, None]+(centres*centres).sum(1)-2*x@centres.T, 0)
                y = np.exp(-node['gamma']*distance/max(1, x.shape[1]))
            elif op == 'square':
                y = x*x
            elif op == 'tanh':
                y = np.tanh(x)
            else:
                y = x*node['factor']
            if y.shape[1] > 4096 or not np.isfinite(y).all() or np.max(np.abs(y), initial=0) > 1e12:
                raise ValueError('Invalid or excessive graph features')
            values[name] = y
        return values[self.spec['output']]

    def fit(self, data, target, rows):
        from .pk2_numeric import dual_ridge
        from .metrics import study_unit_weights
        weights = study_unit_weights(rows)
        x = self.features(data, True, weights)
        fit = dual_ridge(x, target, rows, self.spec['alpha'])
        self.params.update({'readout_'+k: v for k, v in fit.parameters.items()})
        return self

    def predict(self, data):
        readout = {k[8:]: v for k, v in self.params.items() if k.startswith('readout_')}
        return ((self.features(data)-readout['mean'])/readout['scale']) @ readout['coefficients']+readout['target_mean']

    def save(self, folder):
        write_json(folder/'graph.json', self.spec)
        save_npz(folder/'parameters.npz', **self.params)

    @classmethod
    def load(cls, folder):
        obj = cls(read_json(folder/'graph.json'))
        with np.load(folder/'parameters.npz', allow_pickle=False) as f:
            obj.params = {k: f[k].copy() for k in f.files}
        return obj


def baseline_specs():
    return {
        'identity_mean_ridge': {'nodes': [{'id': 'x', 'op': 'concat', 'inputs': ['identity']}], 'output': 'x', 'alpha': 1.},
        'visible_ridge': {'nodes': [{'id': 'x', 'op': 'concat', 'inputs': ['rank', 'amplitude', 'identity', 'coverage', 'programme_available']},
                                   {'id': 'z', 'op': 'standardize', 'inputs': ['x']}], 'output': 'z', 'alpha': 1.},
        'clock_identity_ridge': {'nodes': [{'id': 'x', 'op': 'concat', 'inputs': ['clock', 'clock_available', 'identity']}],
                                 'output': 'x', 'alpha': 1.},
    }
