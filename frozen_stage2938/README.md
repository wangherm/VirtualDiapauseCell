# Frozen VDC stage 2938 code

This directory contains the numerical code behind the four-task stage-2938
package and its independent array/parameter audit. It is a code snapshot,
not a claim of a universal dormancy or functional-recovery predictor.

## Contents

- predict/ and score/: unchanged Python source from the final verified kits.
- backends/: seed and apricot learning plugins, functional source-only fitter,
  and the yeast source-only fitter that produced the retained parameter family.
- scripts/recompute.py: independent saved-parameter prediction and score audit.
- CODE_PROVENANCE.json: source origins and SHA256 hashes.

The tasks concern seed after-ripening RNA, apricot chilling-associated RNA,
late yeast relative RNA, and genotype-treatment germination calibration.
Their numerical models remain task-specific. The functional route has no
RNA input and does not select an operational default.

## Reproduction with authorised evidence

Python 3.11+ and NumPy are required. The existing general_learning execution
driver uses Linux fcntl locking; use Linux/WSL for that driver. The independent
audit can run on Windows or Linux.

Place the locally supplied evidence/ directory and EVIDENCE_MANIFEST.json
at this directory's root, then run:

    python -B scripts/recompute.py

The audit reads saved parameters, predictions and targets and writes score
tables to source_data/. It does not train, resume a controller, or call APIs.
The predict/ and score/ scripts retain their kit contracts and require the
matching verified configuration/input packets. Copying Python alone does not
replace those packets. With an authorised complete kit, its existing command is:

    python -B run_packed.py --task all --output-dir NEW_OUTPUT_DIRECTORY

For the scoring kit also supply --frozen-dir PREDICTION_OUTPUT_DIRECTORY.
Use new output directories; preserve the frozen source kit and inventories.

The backends are preserved for inspection and future authorised reuse.
They are not invoked by this publication or the paper audit. In particular,
prepare.py fit programs retain the original isolated /code, /input and /out
contract rather than a new unrestricted training launcher.

## Release scope

This commit contains code and its provenance only. It does not publish fitted
weights, raw arrays, target packets, sample-level predictions, internal
research conversations, credentials or the manuscript draft. The corresponding
local evidence package and figure scripts remain separate. No dataset or
third-party licensing grant is implied by this snapshot.

No old run is resumed and no research guard is changed. Existing repository
code remains intact. Code validation for this snapshot uses syntax and hash
checks, plus the previously completed frozen-parameter audit, without training.
