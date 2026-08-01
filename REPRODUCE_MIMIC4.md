# Reproducing MHGRL on MIMIC-IV

This patch keeps the released MHGRL architecture and adapts the public
preprocessing code to the `mimiciv/hosp` schema. It does **not** add notes,
laboratory values, vital signs, demographics, or a 24-hour cutoff.

## 1. Expected raw files

Place the MIMIC-IV `hosp` files anywhere accessible to the machine. The adapter
accepts either `.csv.gz` or `.csv` and reads only the needed columns:

```text
admissions.csv.gz
diagnoses_icd.csv.gz
procedures_icd.csv.gz
prescriptions.csv.gz
d_icd_diagnoses.csv.gz     # optional; used only for cohort names in the report
```

The paper cites MIMIC-IV v1.0. This adapter targets MIMIC-IV v3.1 and selects
cohorts from the eligible records actually present after filtering. It is an
adapted baseline, not an exact reconstruction of the paper's Table 1 cohorts.

## 2. Build the released MHGRL artifacts

From the repository root:

```bash
python data/preprocess_mimic4.py \
  --hosp-dir /path/to/mimiciv/1.0/hosp \
  --output-dir data/mimic4
```

The defaults preserve the original notebook's intended admission filters and
cohort-selection rule:

```text
minimum code frequency = 50
admission filters      = exclude NEWBORN and recorded deathtime
cohorts                = 6 most frequent eligible primary ICD-9 codes
split                  = 60% / 20% / 20%
positive pairs         = 5 per admission
negative pairs         = 5 per admission
seed                   = 1002
ICD version            = ICD-9 only
```

ICD-9 is intentional. The released `build_tree.py`, diagnosis/procedure text
embeddings, and the paper's ontology are ICD-9 based. Sending MIMIC-IV ICD-10 or
ICD-10-PCS codes into those assets would not be a faithful reproduction.

To make the cohort-selection policy explicit, the equivalent command is:

```bash
python data/preprocess_mimic4.py \
  --hosp-dir /path/to/mimiciv/hosp \
  --output-dir data/mimic4 \
  --cohort-mode most-frequent
```

Inspect these two audit files before training:

```text
data/mimic4/cohort_summary.csv
data/mimic4/preprocessing_report.json
```

The generated `data/mimic4/` directory is git-ignored because it contains
derived MIMIC data and must not be committed or redistributed.

`cohort_summary.csv` records the selected ICD-9 codes, their titles (when the
dictionary file is available), and their observed EHR counts. Record these
data-derived cohort definitions when reporting baseline results; do not label
them as an exact reproduction of Table 1.

## 3. Install the environment

Use Python 3.10, which is the version tested by the upstream repository. The
upstream package versions can be installed with:

```bash
python -m pip install -r requirements-mimic4.txt
```

Install the CUDA build of PyTorch appropriate for the machine instead of the
CPU wheel when reproducing the reported GPU experiment.

## 4. Train and evaluate

The MIMIC-IV paper configuration is encoded in `code/run_mimic4.sh`:

```text
A-DGN, 2 layers, hidden size 100, tensor neurons 30,
dropout 0.4, batch size 256, 30 epochs, learning rate 0.0001
```

Train disease prediction:

```bash
cd code
TASK=knn ACTION=train bash run_mimic4.sh
```

Windows PowerShell (run from the repository root):

```powershell
.\code\run_mimic4.ps1 -Task knn -Action train
```

Evaluate the saved checkpoint and report Accuracy@1, Accuracy@3, Accuracy@5:

```bash
TASK=knn ACTION=test \
RESUME_PATH=res/mimic4/knn/pytorch_prediction.bin \
bash run_mimic4.sh
```

Windows PowerShell:

```powershell
.\code\run_mimic4.ps1 -Task knn -Action test `
  -ResumePath "res\mimic4\knn\pytorch_prediction.bin"
```

For clustering, replace `TASK=knn` with `TASK=cluster`. The clustering output is
Purity, NMI, and Rand Index.

## What was changed from upstream

| File | Reproduction fix |
|---|---|
| `data/preprocess_mimic4.py` | Reads lowercase MIMIC-IV hosp tables, excludes newborn/death admissions, filters `icd_version=9`, maps NDC to RxNorm, selects the six most frequent eligible primary diagnoses, reports progress, and emits the CSV/PKL contract expected by MHGRL. |
| `code/data_loader.py` | Loads PMI relations from the selected dataset instead of hard-coded `mimic3`; preserves leading zeroes in medical codes; uses stable cohort IDs and repository-relative paths. |
| `code/graph_model.py` | Applies the configured dropout probability, making the paper's MIMIC-IV value `0.4` effective. |
| `code/train.py` | Adds MIMIC-IV data-directory arguments and reports all paper K values (`1,3,5`) in one evaluation. |
| `code/util.py` | Makes nearest-neighbour ordering and vote tie-breaking deterministic. |
| `code/run_mimic4.sh` | Removes the hard-coded MIMIC-III paths/checkpoint and supplies the MIMIC-IV hyperparameters from the paper. |
| `code/run_mimic4.ps1` | Native PowerShell equivalent of the MIMIC-IV training/testing command. |
| `.gitignore` | Prevents raw/derived MIMIC records, graph caches, and checkpoints from being committed. |

`graph_model.py`'s architecture, the heterogeneous edge types, ontology encoder,
neural tensor similarity module, attention aggregation, and pairwise
cross-entropy objective remain unchanged.
