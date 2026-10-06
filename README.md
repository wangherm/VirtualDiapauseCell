# Virtual Diapause Cell

Virtual Diapause Cell models how embryos maintain diapause and resume development. It uses gene expression to locate a sample on a reference reactivation coordinate, describe the programmes associated with that position, and identify changes that the reference does not explain.

The main application is embryonic diapause in the African turquoise killifish, *Nothobranchius furzeri*. Public dormancy and quiescence datasets support pretraining and separate perturbation experiments.

## Model

The project combines a clock–wave numerical model with a domain-adapted Qwen model. The numerical model represents measured expression, reference position and sample-specific deviations. Qwen provides functional representations and helps interpret evidence from the measurements and literature.

CW-stage-1 is the completed numerical reference release. VDC-INT1 implements one input and report interface for numerical prediction, evidence retrieval and live domain-Qwen interpretation. Numerical new-input, serialization and interface checks have run locally; full-mode GPU qualification is performed by the assembly runner on the deployment machine. Read that run's `status.json` before treating it as a qualified full release.

The application workflow is:

```text
Expression and sample metadata
              |
Programme ranks, expression amplitude and coverage
              |
Identity and context checks
              |
Domain-semantic clock–wave model
              |
Reference position, waves, deviations and gene predictions
              |
Evidence retrieval and live domain-Qwen interpretation
              |
A single report with measurements, predictions and sources
```

Fixed programme descriptions are encoded during model assembly. Those representations enter numerical fitting, rather than being added only to a final explanation. The runtime Qwen call then works with the current request and its numerical results. It does not invent expression values or override a measured result.

The view-specific structure is retained:

| View | Numerical readout |
| --- | --- |
| Whole-embryo bulk | Clock, identity and visible programme residuals |
| Whole-pool single-cell aggregate | Clock and identity |
| Coarse-identity profile | Identity-conditioned clock and waves |

Direct Ridge regression and the frozen numerical reference remain available for comparison. Known cell identities take precedence over generated suggestions. A proposed identity, a supported clock position and a functional outcome are different outputs.

## Outputs

A report separates five quantities:

| Quantity | Interpretation |
| --- | --- |
| Observed expression | Values calculated from the supplied measurements |
| Reference position | Molecular position relative to the training anchors |
| Expected expression | The fitted pattern at that position and identity |
| Residual | Observed expression minus its reference value |
| Predicted expression | An estimate from a fitted readout, with its target panel and scale recorded |

The clock is not elapsed time or a percentage of recovery. Its values are not clipped to 0–1. Missing identity references and out-of-range positions are reported explicitly.

The current fixed-panel expression readout uses a fixed panel of 256 held-out genes. Available gene and transcription-factor RNA curves are reference summaries; they do not establish accurate whole-transcriptome prediction or directly measured TF activity.

The full integration report includes the numerical outputs, retrieved evidence, identity suggestions, Qwen interpretation and the actual model versions. A requested perturbation or future-state calculation runs only when a fitted model matches the biological system and input representation.

## Full and numerical modes

The application makes `full` the default execution mode. It requires the domain adapter, the numerical components and a real Qwen call. A missing or failed language-model component must be visible as a partial run, not silently labelled complete.

An explicit `numeric_only` mode remains useful for comparisons and CPU analysis. Disabling an online language-model call does not remove semantic information already used during numerical fitting. A no-semantics comparison therefore uses a separately fitted numerical baseline.

The unified application is a composite model, not a claim that every component was jointly trained or that all dormancy mechanisms have been learned.

## Data

The killifish collection includes whole-embryo RNA-seq, pooled and individual-embryo Exit measurements, a single-cell atlas, later Exit samples, regional libraries and stress comparisons. Sample roles are tracked at the embryo or pool level. Private measurements, sample-role files and fitted artefacts are not distributed in this repository.

Completed public pretraining experiments used:

| Dataset | System and main use |
| --- | --- |
| [GSE288723](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE288723) | *C. elegans* dauer maintenance and recovery |
| [GSE291659](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE291659) | Reproductive diapause, recovery and genotype contrasts |
| [GSE202844](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE202844) | Mouse ESC pausing and METTL3-related perturbation |
| [GSE221467](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE221467) | Mouse ESC pausing and TET1/2 perturbation |
| [GSE124109](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124109) | Rat fibroblast quiescence and maintenance duration |
| [GSE3169](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE3169) | The Dauer MTC microarray branch |

Each dataset retains its measurement scale and experimental context. Sharing a programme representation does not establish a universal physiological clock across species. The [data catalogue](knowledge/datasets.json) distinguishes screened resources from those actually used in a run.

## Installation

Python 3.10 or later is required. Use a working PyTorch environment for your machine.

```bash
git clone https://github.com/wangherm/VirtualDiapauseCell.git
cd VirtualDiapauseCell
python -m pip install -e ".[test,app]"
python -m vdc --help
```

Qwen execution also requires the separately configured base snapshot, domain adapter and the knowledge dependencies used by the validated training environment. Model weights and private data are not included in a fresh clone.

## Use the application

Use an assembled private model bundle and its matching code release. The repository itself contains no private fitted model or expression matrices. See the [INT1 deployment guide](docs/INT1_APPLICATION_CN.md) for the bounded assembly and GPU acceptance run.

```bash
# Default: fitted domain-semantic numerical model AND live domain Qwen.
export VDC_QWEN_BASE=/path/to/verified/base/snapshot
python -m vdc analyse --model /path/to/bundle --request request.json --output result

# Explicit CPU numerical comparison with the frozen no-semantics baseline.
python -m vdc analyse --model /path/to/bundle --request request.json \
  --mode numeric_only --variant cw_stage_baseline --output baseline-result

# A real input application on localhost, using the same Python inference path.
python -m vdc serve-app --model /path/to/bundle --output /path/to/request-results
```

The service listens on `127.0.0.1:8769`. Use an SSH tunnel to access it remotely. It accepts JSON or matrix/sample-table uploads, computes predictions and provides report/ZIP downloads. It is an internal research application, without public authentication or multi-user isolation.

```python
from vdc.integration import VirtualDiapauseCell, read_request
model = VirtualDiapauseCell("/path/to/bundle", base_path="/path/to/base/snapshot")
result = model.analyse(read_request("request.json"), output="result")
```

Each query requires a new request ID, species, context, view, measurement scale, gene IDs and sample metadata. CSV has `sample_id` followed by gene columns. H5AD requires an explicit named counts layer; `.X` is never guessed. Inputs are whole-embryo, pool or coarse-identity aggregate profiles in the fitted gene universe, not an unaggregated single-cell atlas. Missing genes are rejected rather than silently imputed. The [input contract and examples](docs/INT1_APPLICATION_CN.md) describe the metadata and scoped public-response requests.

Outputs include `report.html`, `summary.md`, `result.json`, `provenance.json`, CSV tables, figures, retrieved evidence and raw Qwen traces. `partial` preserves successful numerical outputs and the actual failed subcall; `unsupported` identifies a biological scope or reference that is unavailable. No fallback model is silently substituted.

Export is a separate, offline step:

```bash
python -m vdc export-model --stage /path/to/completed-stage \
  --integration /path/to/completed-int1/assembly --output /path/to/new-bundle
```

The bundle holds relative parameter paths, programme/identity definitions, semantic caches, the domain adapter and compatible public-response models. Base Qwen weights remain in an external verified cache. Inference never fits; it does not need the original training run or old validation predictions. Keep the matching code release alongside the bundle.

Historical experiments and the saved-results stage viewer remain available through the [stage guide](docs/STAGE_RELEASE_CN.md); they are not new-input inference.

## Interpretation

Performance varies by data view. Simple numerical baselines remain competitive. Qwen has been more reliable at extracting facts from supplied evidence than at assigning cell identity without numerical guidance. Full integration does not remove these limitations.

Functional depth, causal lineage, universal cross-species mapping and arbitrary killifish intervention forecasts are not established capabilities. Unavailable results are kept separate from failed software calls and from biological uncertainty. Historical experiments and the results of a new integrated model should be reported separately.

## Repository

```text
src/vdc/       Numerical models, identity, knowledge and application code
configs/       Reference, model and execution settings
knowledge/     Functional definitions, sources and dataset catalogues
scripts/       Training, analysis and packaging entry points
tests/         Numerical and interface tests
docs/          Methods, run guides and experiment history
```

The scientific tasks are state and maintenance, Exit biotime, gene/TF waves, and programme/regulon waves. The functional areas index these tasks; they are not separate large language models.

```bash
python -m pytest -q
python scripts/validate_catalogue.py
```

Software tests establish that an implementation behaves as intended. Biological claims require the corresponding data and evaluation. Keep the code, model, reference definitions, input provenance and sample roles with each result.
