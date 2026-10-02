# Definition data contract

This directory contains a **definition snapshot**, not a training dataset.

| File | Records | Meaning |
| --- | --- | --- |
| core_tasks.json | `core_tasks[]`, `coordinate_registry[]` | DC01–DC04 priorities, named biotime objects, wave contract and frozen-data design boundary |
| modules.json | `modules[]` | Proposed functional questions, observation channels, limits and vocabulary anchors |
| sources.json | `sources[]` | Primary literature, an architecture perspective and official database entry points |
| evidence_seed.json | `records[]` | Unreviewed source-grounded reading notes; never gold labels |
| datasets.json | `datasets[]` | Candidate admission work and a frozen internal boundary; not downloaded expression files |
| snapshots/go_2026-10-01 | raw response + manifest | Actual QuickGO ontology metadata, with URL, time and SHA256 |

IDs are versioned within this project. M01–M14 deliberately use a new namespace rather than imply identity with the attachments' DM01–DM06. Submodules permit overlap. No GO ancestor expansion, taxon-specific membership or direction-of-effect labels have been generated.

The validator for v0.2 rejects training-ready claims and downloaded-expression claims by design. To admit real data later, revise the contract and validation with a documented schema migration; do not simply flip booleans. File paths and SHA256, donor/embryo/culture grouping, assay and scale, condition/time, missingness, paper family and split must then be audited.

Future reviewed evidence needs experimental units, initial state, intervention/dose, time, observed endpoint, effect and uncertainty when reported, figure/table locator, limitations, reviewer/date, permitted use and split. A missing measurement remains null. A model-generated extraction is a candidate awaiting review; a prediction cannot be relabelled as an experiment. Current summary records sometimes cover more than one assay and must be split into experimental records before any supervision is derived.

The internal boundary includes result-derived labels and text, not just matrices. Duplicate papers, preprints, supplements and reused experiments require a shared family identity before train/validation/test assignment. The present `paper_family` IDs are seed identifiers, not evidence that deduplication is complete.

GO data attribution: [Gene Ontology Consortium](https://geneontology.org/docs/go-citation-policy/), retrieved through [QuickGO](https://www.ebi.ac.uk/QuickGO/api/index.html). This snapshot contains ontology terms only; their presence does not demonstrate involvement in diapause.
