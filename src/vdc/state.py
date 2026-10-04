"""Shared numerical core: programme-token encoder, clock and observation decoders.

This is a small trainable reference implementation, not a validated virtual cell.
Qwen is optional; absence of its weights never silently loads another model.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import numpy as np
import torch
from torch import nn
from .contracts import ObservationBundle
from .io import read_json, write_json, object_hash, save_npz
from .metrics import grouped_metrics, study_unit_weights


@dataclass
class StateConfig:
    hidden_dim: int = 64
    latent_dim: int = 16
    layers: int = 2
    heads: int = 4
    learning_rate: float = 0.001
    clock_weight: float = 1.0
    corrupt_fraction: float = 0.2
    additive_noise: float = 0.05
    seed: int = 42
    validation_every: int = 10
    encoder_kind: str = 'programme_transformer'

    def validate(self) -> None:
        if self.hidden_dim <= 0 or self.heads <= 0 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if self.layers < 1 or self.latent_dim < 1 or self.validation_every < 1:
            raise ValueError("Positive dimensions/validation interval required")
        if not 0 < self.corrupt_fraction < 1 or self.additive_noise < 0:
            raise ValueError("Invalid corruption parameters")
        if self.learning_rate <= 0 or self.clock_weight < 0:
            raise ValueError("Invalid learning rate / task weight")
        if self.encoder_kind not in {'programme_transformer','gene_mlp'}:
            raise ValueError('Unknown numerical encoder')


class ProgrammeStateModel(nn.Module):
    def __init__(self, n_features: int, config: StateConfig, semantics: torch.Tensor | None = None, n_contexts: int = 1):
        super().__init__()
        config.validate()
        self.config = config
        d = config.hidden_dim
        self.numeric = nn.Linear(3, d)
        self.feature_id = nn.Embedding(n_features, d)
        # A zero cache supplies the no-semantics control when matching semantic capacity.
        sem = torch.zeros(n_features, 1) if semantics is None else semantics.float()
        if sem.ndim != 2 or sem.shape[0] != n_features or not torch.isfinite(sem).all():
            raise ValueError("Semantic cache must match programme order and be finite")
        self.register_buffer("semantics", sem.detach().clone())
        self.semantic_projection = nn.Linear(sem.shape[1], d, bias=False)
        layer = nn.TransformerEncoderLayer(d, config.heads, dim_feedforward=d * 2,
                                           dropout=0, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.to_latent = nn.Linear(d, config.latent_dim)
        self.query = nn.Linear(config.latent_dim, d)
        self.programme_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.clock_head = nn.Linear(config.latent_dim, 1)
        if config.encoder_kind=='gene_mlp':
            self.gene_encoder=nn.Sequential(nn.Linear(3*n_features,d),nn.GELU(),nn.Linear(d,config.latent_dim))
        if n_contexts>1:
            self.context_readout=nn.Embedding(n_contexts,2*n_features)
            nn.init.zeros_(self.context_readout.weight)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, coverage: torch.Tensor, context_index=None) -> dict:
        if values.ndim != 2 or values.shape != mask.shape or coverage.shape != values.shape:
            raise ValueError("Invalid input shapes")
        if values.shape[1] != self.feature_id.num_embeddings or not mask.any(1).all():
            raise ValueError("Feature order/size mismatch or all-missing sample")
        object_vectors = self.feature_id.weight + self.semantic_projection(self.semantics)
        numeric = torch.stack((torch.where(mask, values, 0), mask.float(),
                               torch.where(mask, coverage, 0)), dim=-1)
        if self.config.encoder_kind=='gene_mlp':
            z=self.gene_encoder(numeric.flatten(1))
        else:
            token = self.numeric(numeric) + object_vectors[None]
            encoded = self.encoder(token, src_key_padding_mask=~mask)
            pooled = (encoded * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            z = self.to_latent(pooled)
        query = self.query(z)[:, None, :].expand(-1, values.shape[1], -1)
        objects = object_vectors[None].expand(values.shape[0], -1, -1)
        reconstructed = self.programme_head(torch.cat((query, objects), dim=-1)).squeeze(-1)
        if hasattr(self,'context_readout'):
            if context_index is None:raise ValueError('Multi-study model requires explicit context')
            slope,offset=self.context_readout(context_index).chunk(2,dim=-1)
            reconstructed=reconstructed*(1+slope)+offset
        return {"latent": z, "programme": reconstructed, "clock": self.clock_head(z).squeeze(-1)}


def masked_weighted_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                        weights: torch.Tensor) -> torch.Tensor:
    # A missing target contributes neither a zero label nor a negative decision.
    if pred.ndim == 1:
        pred, target, mask = pred[:, None], target[:, None], mask[:, None]
    count = mask.sum(1)
    eligible = count > 0
    if not eligible.any():
        return pred.sum() * 0
    error = (torch.where(mask, pred - target, 0) ** 2).sum(1) / count.clamp_min(1)
    w = weights * eligible
    return (w * error).sum() / w.sum()


def corrupt(x: torch.Tensor, mask: torch.Tensor, coverage: torch.Tensor,
            fraction: float, noise: float, seed: int) -> tuple:
    # Generate CPU randomness by step, so an interrupted CPU run can resume exactly.
    g = torch.Generator(device="cpu").manual_seed(seed)
    drop = (torch.rand(mask.shape, generator=g) < fraction).to(mask.device) & mask
    for i in range(len(mask)):
        available = torch.where(mask[i])[0]
        if len(available) > 1 and not drop[i].any():
            drop[i, available[-1]] = True
        if torch.equal(drop[i], mask[i]):
            drop[i, available[0]] = False
    kept = mask & ~drop
    eps = torch.randn(x.shape, generator=g).to(x.device) * noise
    return torch.where(kept, x + eps, 0), kept, torch.where(kept, coverage, 0), drop


def train_scaling(values: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> tuple:
    w = weights[:, None] * mask
    den = w.sum(0)
    if (den <= 0).any():
        raise ValueError("Some programmes are never observed in train; remove explicitly or supply data")
    mean = (np.where(mask, values, 0) * w).sum(0) / den
    var = (np.where(mask, values - mean, 0) ** 2 * w).sum(0) / den
    return mean.astype(np.float32), np.maximum(np.sqrt(var), 1e-3).astype(np.float32)


def atomic_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_name(path.name + ".partial")
    torch.save(state, tmp)
    os.replace(tmp, path)


def fit_state(bundle: ObservationBundle, run_dir: str | Path, steps: int = 100,
              config: StateConfig | None = None, device: str = "cpu", resume: bool = False,
              semantics: np.ndarray | None = None, semantic_provenance: dict | None = None,
              pretrained: str | Path | None = None, initial_weights: str | Path | None = None,
              transfer_mode: str = 'shared_model', checkpoint_steps=(), multi_context: bool = False) -> dict:
    bundle.validate()
    b = bundle.subset([i for i, r in enumerate(bundle.rows) if r["split"] in {"train", "validation"}])
    from .admission import audit_internal_task
    audit_internal_task(b.rows,'state')
    cfg = config or StateConfig(); cfg.validate()
    if steps < 1:
        raise ValueError("steps must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose cpu explicitly for software tests")
    train, valid = b.indices("train"), b.indices("validation")
    if len(train) < 2 or len(valid) < 1:
        raise ValueError("Explicit train and validation observations required")
    run = Path(run_dir)
    if run.exists() and any(run.iterdir()) and not resume:
        raise FileExistsError("Use a new run directory or --resume; results are immutable by default")
    run.mkdir(parents=True, exist_ok=True)
    tr_weights = study_unit_weights([b.rows[i] for i in train])
    contexts=sorted({r['study_family'] for r in b.rows}) if multi_context else []
    ci=np.array([contexts.index(r['study_family']) for r in b.rows]) if contexts else None
    if contexts:
        scalers=[]
        for i in range(len(contexts)):
            idx=train[ci[train]==i]
            if not len(idx):raise ValueError('Context without training observations')
            # Programme absent in this study remains masked; other studies may support it.
            available=b.mask[idx].any(0);mean=np.zeros(len(b.feature_ids),np.float32);std=np.ones_like(mean)
            if available.any():mean[available],std[available]=train_scaling(b.values[idx][:,available],b.mask[idx][:,available],study_unit_weights([b.rows[j] for j in idx]))
            scalers.append((mean,std))
        mu,scale=map(np.stack,zip(*scalers));row_mu,row_scale=mu[ci],scale[ci]
    else:
        mu, scale = train_scaling(b.values[train], b.mask[train], tr_weights);row_mu,row_scale=mu,scale
    torch.manual_seed(cfg.seed)
    sem = None if semantics is None else torch.as_tensor(semantics, dtype=torch.float32)
    if sem is not None and not semantic_provenance:
        raise ValueError("Semantic vectors require object/model/source provenance")
    model = ProgrammeStateModel(len(b.feature_ids), cfg, sem,max(1,len(contexts))).to(device)
    initial_hash = None
    if initial_weights is not None:
        from .io import sha256
        initial = torch.load(initial_weights, map_location='cpu', weights_only=True)
        if initial['feature_ids'] != b.feature_ids:
            raise ValueError('Initial feature ID order differs')
        own = model.state_dict()
        if set(initial['model']) != {k for k in own if not k.startswith('context_readout.')}:
            raise ValueError('Initial architecture/semantic capacity differs')
        for key, value in initial['model'].items():
            if key == 'semantics': continue
            if key not in own or own[key].shape != value.shape:
                raise ValueError('Initial architecture/semantic capacity differs')
            own[key] = value
        model.load_state_dict(own)
        initial_hash = sha256(initial_weights)
    transfer = None
    if pretrained is not None:
        if transfer_mode not in {'shared_model','encoder_only'}:raise ValueError('Unknown transfer mode')
        if resume:
            raise ValueError('Transfer initialization is only applied to a fresh run')
        parent = Path(pretrained)
        pm = read_json(parent / 'run.json')
        if pm['scope']['feature_ids'] != b.feature_ids:
            raise ValueError('Transfer requires identical explicit programme ID order')
        if any(pm['config'][k] != asdict(cfg)[k] for k in ('hidden_dim','latent_dim','layers','heads')):
            raise ValueError('Transfer architecture differs')
        ck = torch.load(parent / 'best.pt', map_location='cpu', weights_only=True)
        own = model.state_dict()
        copied = []
        # Local normalization, semantic cache/projection and clock readout are intentionally refitted.
        for key, value in ck['model'].items():
            if key == 'semantics' or key.startswith(('clock_head.', 'semantic_projection.','context_readout.')):
                continue
            if transfer_mode=='encoder_only' and not key.startswith(('numeric.','feature_id.','encoder.','to_latent.','gene_encoder.')):continue
            if key not in own or own[key].shape != value.shape:
                raise ValueError('Incompatible transferred parameter ' + key)
            own[key] = value; copied.append(key)
        model.load_state_dict(own)
        from .io import sha256
        transfer = {'parent_checkpoint_sha256': sha256(parent/'best.pt'), 'copied_parameters': copied,
                    'local_scaling_and_clock': True, 'parent_scope': pm['scope'],'mode':transfer_mode}
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    sem_hash = object_hash({"shape": list(model.semantics.shape),
                            "values": model.semantics.cpu().tolist(),
                            "provenance": semantic_provenance})
    available_clock = bool(b.clock_mask[train].any() and cfg.clock_weight > 0)
    summary = {"model_kind": "state", "package_version": "0.4.0", "scope": b.scope,
               "bundle_fingerprint": b.fingerprint, "config": asdict(cfg),
               "semantic_hash": sem_hash, "semantic_provenance": semantic_provenance,
               "fitted_split": "train", "split_policy": b.split_policy,
               "fit_ids": [b.rows[i]["observation_id"] for i in train],
               "fit_rows": [b.rows[i] for i in train],
               "is_synthetic": any(r["origin"] == "synthetic" for r in b.rows),
               "capabilities": ["programme_reconstruction"] + (["clock"] if available_clock else []),
               "science_status": "unvalidated", "uncertainty_status": "not_calibrated"}
    summary['transfer'] = transfer
    summary['initial_weights_sha256'] = initial_hash
    if contexts:summary['contexts']=contexts
    start, best = 0, float("inf")
    if resume:
        old = read_json(run / "run.json")
        summary['transfer'] = old.get('transfer')
        if initial_hash and initial_hash != old.get('initial_weights_sha256'):
            raise ValueError('Initial weights changed during resume')
        summary['initial_weights_sha256'] = old.get('initial_weights_sha256')
        for k in ("bundle_fingerprint", "config", "semantic_hash"):
            if old[k] != summary[k]:
                raise ValueError(f"Cannot resume: {k} changed")
        if old.get('contexts',[])!=summary.get('contexts',[]):raise ValueError('Cannot resume: study contexts changed')
        ck = torch.load(run / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(ck["model"]); optimiser.load_state_dict(ck["optimiser"])
        start, best = ck["step"], ck["best_score"]
        if steps < start:
            raise ValueError("Requested total steps precede the saved checkpoint")
    write_json(run / "run.json", summary)
    tensor = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype, device=device)
    x = tensor(np.where(b.mask, (b.values - row_mu) / row_scale, 0))
    ctx=tensor(ci,torch.long) if contexts else None
    mask, cov = tensor(b.mask, torch.bool), tensor(b.coverage)
    targets, clock_mask = tensor(b.clock), tensor(b.clock_mask, torch.bool)
    weights = tensor(tr_weights)
    vt = tensor(study_unit_weights([b.rows[i] for i in valid]))
    vx, vm, vc, hidden = corrupt(x[valid], mask[valid], cov[valid], cfg.corrupt_fraction,
                                 cfg.additive_noise, cfg.seed + 1_000_000)
    if not hidden.any():
        raise ValueError("Validation requires at least two measured programmes in a sample")
    log_path = run / "steps.jsonl"
    for step in range(start, steps):
        model.train()
        xx, mm, cc, _ = corrupt(x[train], mask[train], cov[train], cfg.corrupt_fraction,
                                cfg.additive_noise, cfg.seed + step)
        out = model(xx, mm, cc,None if ctx is None else ctx[train])
        recon = masked_weighted_mse(out["programme"], x[train], mask[train], weights)
        closs = masked_weighted_mse(out["clock"], targets[train], clock_mask[train], weights)
        loss = recon + (cfg.clock_weight * closs if available_clock else 0)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss; run stopped without replacing prior valid checkpoint")
        optimiser.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimiser.step()
        if (step + 1) % cfg.validation_every == 0 or step + 1 == steps or step+1 in checkpoint_steps:
            model.eval()
            with torch.no_grad():
                vo = model(vx, vm, vc,None if ctx is None else ctx[valid])
                vr = masked_weighted_mse(vo["programme"], x[valid], hidden, vt)
                vclock = masked_weighted_mse(vo["clock"], targets[valid], clock_mask[valid], vt)
                score = float(vr + (cfg.clock_weight * vclock if available_clock else 0))
            row = {"step": step + 1, "train_reconstruction": float(recon.detach()),
                   "train_clock": float(closs.detach()) if available_clock else None,
                   "validation_hidden_mse_scaled": float(vr),
                   "validation_clock_mse": float(vclock) if available_clock and clock_mask[valid].any() else None,
                   "selection_score": score}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            improved = score < best
            if improved:
                best = score
            ck = {"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                  "step": step + 1, "best_score": best, "mean": tensor(mu), "scale": tensor(scale)}
            atomic_checkpoint(run / "last.pt", ck)
            if improved:
                atomic_checkpoint(run / "best.pt", ck)
            if step+1 in checkpoint_steps:atomic_checkpoint(run/f'step_{step+1}.pt',ck)
    predictor = StatePredictor.load(run, device=device)
    # This validation view hides only additional measurements, never biological replicates.
    with torch.no_grad():
        raw = predictor.model(vx, vm, vc,None if ctx is None else ctx[valid])
    vm_scale,vm_mean=(row_scale[valid],row_mu[valid]) if contexts else (scale,mu)
    pred = raw["programme"].cpu().numpy() * vm_scale + vm_mean
    report = {
        "run_kind": "synthetic_software_test" if summary["is_synthetic"] else "research_candidate",
        "scientific_success": False, "test_split_evaluated": False,
        "validation_hidden_reconstruction": grouped_metrics(pred, b.values[valid], hidden.cpu().numpy(),
                                                             [b.rows[i] for i in valid]),
        "training_mean_hidden_baseline": grouped_metrics(np.broadcast_to(vm_mean, pred.shape), b.values[valid],
                                                          hidden.cpu().numpy(), [b.rows[i] for i in valid]),
        "clock": grouped_metrics(raw["clock"].cpu().numpy(), b.clock[valid], b.clock_mask[valid],
                                  [b.rows[i] for i in valid]) if available_clock else {"status": "unavailable"},
        "steps_completed": steps, "checkpoint_selection": "validation_hidden_mse_scaled + weighted_clock_mse",
        "limitations": ["Validation reconstruction concerns artificial extra corruption, not latent biological truth",
                        "Task-level study metrics are reported; no population generalisation claim is made",
                        "No calibrated uncertainty or functional depth is fitted"]}
    write_json(run / "metrics.json", report)
    arrays = {"programme": pred, "target": b.values[valid], "evaluation_mask": hidden.cpu().numpy()}
    if available_clock:
        arrays["clock"] = raw["clock"].cpu().numpy()
    save_npz(run / "validation_predictions.npz", **arrays)
    return report


class StatePredictor:
    def __init__(self, model: ProgrammeStateModel, mean: np.ndarray, scale: np.ndarray, manifest: dict, device: str):
        self.model, self.mean, self.scale, self.manifest, self.device = model, mean, scale, manifest, device
        self.model.eval()

    @classmethod
    def load(cls, run_dir: str | Path, device: str = "cpu") -> "StatePredictor":
        run = Path(run_dir); m = read_json(run / "run.json")
        ck = torch.load(run / "best.pt", map_location=device, weights_only=True)
        cfg = StateConfig(**m["config"])
        model = ProgrammeStateModel(len(m["scope"]["feature_ids"]), cfg, ck["model"]["semantics"],max(1,len(m.get('contexts',[])))).to(device)
        model.load_state_dict(ck["model"])
        return cls(model, ck["mean"].cpu().numpy(), ck["scale"].cpu().numpy(), m, device)

    def predict(self, bundle: ObservationBundle) -> dict:
        b = bundle.validate()
        if b.scope != self.manifest["scope"]:
            raise ValueError("Context/feature/preprocessing/clock reference mismatch; explicit calibration is required")
        contexts=self.manifest.get('contexts',[])
        ci=np.array([contexts.index(r['study_family']) for r in b.rows]) if contexts else None
        mean,scale=(self.mean[ci],self.scale[ci]) if contexts else (self.mean,self.scale)
        x = np.where(b.mask, (b.values - mean) / scale, 0)
        with torch.no_grad():
            out = self.model(torch.as_tensor(x, dtype=torch.float32, device=self.device),
                             torch.as_tensor(b.mask, device=self.device),
                             torch.as_tensor(b.coverage, dtype=torch.float32, device=self.device),
                             None if ci is None else torch.as_tensor(ci,dtype=torch.long,device=self.device))
        return {"programme": out["programme"].cpu().numpy() * scale + mean,
                "latent": out["latent"].cpu().numpy(),
                "clock": out["clock"].cpu().numpy() if "clock" in self.manifest["capabilities"] else None,
                "measured_mask": b.mask.copy(), "uncertainty_status": "not_calibrated",
                "science_status": self.manifest["science_status"]}
