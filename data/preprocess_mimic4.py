#!/usr/bin/env python3
"""Build the MIMIC-IV artifacts consumed by the original MHGRL code.

This is an adapter, not a new feature pipeline.  It intentionally preserves the
paper/repository representation:

* one graph per hospital admission;
* diagnosis, ICD procedure, and medication nodes;
* the six most frequent eligible primary ICD-9 diagnosis cohorts;
* a 60/20/20 split and five positive/five negative training pairs;
* PMI relations fitted on the selected cohort, as in data_process.ipynb.

The ontology and released text embeddings are ICD-9 based, so MIMIC-IV rows
with ``icd_version == 10`` are excluded rather than silently sent through an
incompatible tree builder.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import pickle
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_TABLES = {
    "admissions": [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "admission_type",
    ],
    "diagnoses_icd": [
        "subject_id",
        "hadm_id",
        "seq_num",
        "icd_code",
        "icd_version",
    ],
    "procedures_icd": [
        "subject_id",
        "hadm_id",
        "seq_num",
        "icd_code",
        "icd_version",
    ],
    "prescriptions": ["subject_id", "hadm_id", "ndc"],
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Convert MIMIC-IV hosp tables to the original MHGRL format."
    )
    parser.add_argument(
        "--hosp-dir",
        type=Path,
        required=True,
        help="Directory containing the MIMIC-IV hosp CSV/CSV.GZ tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "mimic4",
        help="Destination for MHGRL CSV/PKL artifacts.",
    )
    parser.add_argument(
        "--ndc-rxnorm-map",
        type=Path,
        default=repo_root / "data" / "ndc_atc" / "ndc2rxnorm_mapping.txt",
        help="Released MHGRL NDC-to-RxNorm mapping.",
    )
    parser.add_argument("--min-code-count", type=int, default=50)
    parser.add_argument(
        "--cohort-mode",
        choices=["most-frequent"],
        default="most-frequent",
        help="Select the most frequent eligible primary ICD-9 diagnoses (default).",
    )
    parser.add_argument(
        "--cohort-codes",
        nargs="+",
        default=None,
        help="Optional explicit ICD-9 primary-diagnosis codes; overrides --cohort-mode.",
    )
    parser.add_argument(
        "--num-cohorts",
        type=int,
        default=6,
        help="Number of cohorts used only with --cohort-mode most-frequent.",
    )
    parser.add_argument(
        "--exclude-newborn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude NEWBORN admissions (default: enabled).",
    )
    parser.add_argument(
        "--exclude-in-hospital-deaths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude admissions with a recorded deathtime (default: enabled).",
    )
    parser.add_argument("--positive-per-anchor", type=int, default=5)
    parser.add_argument("--negative-per-anchor", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1002)
    return parser.parse_args()


def progress(step: int, total: int, message: str, started: float | None = None) -> float:
    """Print an immediately visible preprocessing progress message."""
    now = time.perf_counter()
    if started is None:
        print(f"[{step}/{total}] {message}...", flush=True)
    else:
        print(f"[{step}/{total}] {message} ({now - started:.1f}s)", flush=True)
    return now


def find_table(hosp_dir: Path, stem: str, required: bool = True) -> Path | None:
    for suffix in (".csv.gz", ".csv"):
        candidate = hosp_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(
            f"Missing {stem}.csv.gz (or {stem}.csv) in {hosp_dir}"
        )
    return None


def read_table(hosp_dir: Path, stem: str, columns: list[str]) -> pd.DataFrame:
    path = find_table(hosp_dir, stem)
    string_dtypes = {
        name: "string" for name in ("icd_code", "ndc") if name in columns
    }
    frame = pd.read_csv(
        path,
        usecols=lambda name: name.lower() in columns,
        dtype=string_dtypes,
        low_memory=False,
    )
    frame.columns = frame.columns.str.lower()
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    return frame[columns]


def normalize_code(value: object) -> str | None:
    if pd.isna(value):
        return None
    code = re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()
    return code or None


def normalize_ndc(value: object) -> str | None:
    if pd.isna(value):
        return None
    raw = str(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    digits = re.sub(r"\D", "", raw)
    if not digits or int(digits) == 0:
        return None
    return digits.zfill(11)


def load_ndc_to_rxnorm(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        raw_mapping = ast.literal_eval(handle.read())
    mapping: dict[str, str] = {}
    for ndc, rxnorm in raw_mapping.items():
        normalized = normalize_ndc(ndc)
        if normalized is not None and pd.notna(rxnorm):
            mapping[normalized] = str(rxnorm)
    if not mapping:
        raise ValueError(f"No valid NDC-to-RxNorm entries found in {path}")
    return mapping


def ordered_unique_join(values: Iterable[object]) -> str:
    return ",".join(dict.fromkeys(str(value) for value in values if pd.notna(value)))


def filter_frequent_codes(
    frame: pd.DataFrame, column: str, minimum: int
) -> tuple[pd.DataFrame, set[str]]:
    counts = frame[column].value_counts()
    valid_codes = set(counts[counts >= minimum].index.astype(str))
    return frame[frame[column].isin(valid_codes)].copy(), valid_codes


def aggregate_modality(
    frame: pd.DataFrame, value_columns: list[str]
) -> pd.DataFrame:
    aggregations = {column: ordered_unique_join for column in value_columns}
    return (
        frame.groupby(["subject_id", "hadm_id"], sort=False, as_index=False)
        .agg(aggregations)
    )


def split_admissions(
    cohort: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    admission_ids = cohort["HADM_ID"].tolist()
    random.Random(seed).shuffle(admission_ids)
    train_end = math.floor(0.6 * len(admission_ids))
    valid_end = math.floor(0.8 * len(admission_ids))
    split_ids = {
        "train": set(admission_ids[:train_end]),
        "valid": set(admission_ids[train_end:valid_end]),
        "test": set(admission_ids[valid_end:]),
    }
    frames = tuple(
        cohort[cohort["HADM_ID"].isin(split_ids[name])].copy()
        for name in ("train", "valid", "test")
    )
    assert sum(len(frame) for frame in frames) == len(cohort)
    return frames


def validate_split_support(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    selected_diseases: list[str],
    positive_per_anchor: int,
) -> None:
    minimum = positive_per_anchor + 1
    for split_name, frame in (("train", train), ("valid", valid)):
        counts = frame["disease"].value_counts()
        insufficient = counts[counts < minimum]
        missing = set(selected_diseases) - set(counts.index)
        if not insufficient.empty or missing:
            raise ValueError(
                f"{split_name} split cannot provide {positive_per_anchor} positives "
                f"per anchor. Counts below {minimum}: {insufficient.to_dict()}; "
                f"missing cohorts: {sorted(missing)}. Choose another --seed only if "
                "you intend to reproduce a different random split."
            )
    train_diseases = set(train["disease"])
    unseen_test = set(test["disease"]) - train_diseases
    if unseen_test:
        raise ValueError(f"Test contains cohorts absent from train: {sorted(unseen_test)}")


def construct_labels(
    frame: pd.DataFrame,
    output_file: Path,
    positive_per_anchor: int,
    negative_per_anchor: int,
    seed: int,
) -> tuple[int, int]:
    rng = random.Random(seed)
    by_disease = {
        disease: list(group["HADM_ID"])
        for disease, group in frame.groupby("disease", sort=True)
    }
    all_ids = frame["HADM_ID"].tolist()
    rows: list[tuple[int, int, int]] = []
    positive_count = 0
    negative_count = 0

    for disease, disease_ids in by_disease.items():
        other_ids = [
            hadm_id
            for other_disease, ids in by_disease.items()
            if other_disease != disease
            for hadm_id in ids
        ]
        if len(disease_ids) <= positive_per_anchor:
            raise ValueError(
                f"Cohort {disease} has only {len(disease_ids)} records; "
                f"{positive_per_anchor + 1} are required."
            )
        if len(other_ids) < negative_per_anchor:
            raise ValueError("Not enough cross-cohort records for negative sampling.")

        for anchor in disease_ids:
            positives = rng.sample(
                [hadm_id for hadm_id in disease_ids if hadm_id != anchor],
                positive_per_anchor,
            )
            negatives = rng.sample(other_ids, negative_per_anchor)
            rows.extend((anchor, candidate, 1) for candidate in positives)
            rows.extend((anchor, candidate, 0) for candidate in negatives)
            positive_count += len(positives)
            negative_count += len(negatives)

    pd.DataFrame(
        rows, columns=["left_hadm_id", "right_hadm_id", "label"]
    ).to_csv(output_file, sep="\t", index=False)
    return positive_count, negative_count


def construct_relation(
    frame: pd.DataFrame, head_col: str, tail_col: str
) -> list[tuple[str, str, float]]:
    pair_count: Counter[tuple[str, str]] = Counter()
    entity_count: Counter[str] = Counter()
    window_count = 0

    for head_values, tail_values in frame[[head_col, tail_col]].itertuples(
        index=False, name=None
    ):
        heads = str(head_values).split(",")
        tails = str(tail_values).split(",")
        for head in heads:
            for tail in tails:
                pair_count[(head, tail)] += 1
                entity_count[head] += 1
                entity_count[tail] += 1
                window_count += 1

    relations = []
    for (head, tail), count in pair_count.items():
        pmi = math.log(
            (count / window_count)
            / ((entity_count[head] / window_count) * (entity_count[tail] / window_count))
        )
        if pmi >= 0:
            relations.append((head, tail, pmi))
    return relations


def write_relation(rows: list[tuple[str, str, float]], path: Path) -> None:
    pd.DataFrame(rows, columns=["head ent", "tail ent", "pmi"]).to_csv(
        path, sep="\t", index=False
    )


def disease_titles(hosp_dir: Path) -> dict[str, str]:
    path = find_table(hosp_dir, "d_icd_diagnoses", required=False)
    if path is None:
        return {}
    frame = pd.read_csv(
        path,
        usecols=lambda name: name.lower() in {"icd_code", "icd_version", "long_title"},
        dtype={"icd_code": "string"},
        low_memory=False,
    )
    frame.columns = frame.columns.str.lower()
    frame = frame[pd.to_numeric(frame["icd_version"], errors="coerce") == 9].copy()
    frame["icd_code"] = frame["icd_code"].map(normalize_code)
    return frame.dropna(subset=["icd_code"]).set_index("icd_code")["long_title"].to_dict()


def preprocess(args: argparse.Namespace) -> dict[str, object]:
    cohort_mode = getattr(args, "cohort_mode", "most-frequent")
    cohort_codes_arg = getattr(args, "cohort_codes", None)
    exclude_newborn = getattr(args, "exclude_newborn", True)
    exclude_deaths = getattr(args, "exclude_in_hospital_deaths", True)

    if args.min_code_count < 1:
        raise ValueError("--min-code-count must be at least 1")
    if args.num_cohorts < 2:
        raise ValueError("--num-cohorts must be at least 2")
    if cohort_codes_arg:
        requested_cohort_codes = [normalize_code(code) for code in cohort_codes_arg]
        if any(code is None for code in requested_cohort_codes):
            raise ValueError("--cohort-codes contains an empty or invalid code")
        if len(set(requested_cohort_codes)) != len(requested_cohort_codes):
            raise ValueError("--cohort-codes must not contain duplicates")
        effective_cohort_mode = "explicit"
    else:
        requested_cohort_codes = None
        effective_cohort_mode = "most-frequent"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_steps = 12
    step_started = progress(1, total_steps, "Loading NDC-to-RxNorm mapping")
    ndc_to_rxnorm = load_ndc_to_rxnorm(args.ndc_rxnorm_map)
    # Keep the exact mapping beside the generated relations so training cannot
    # accidentally use a different mapping from preprocessing.
    with (args.output_dir / "ndc2rxnorm_mapping.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(repr(ndc_to_rxnorm))
    progress(1, total_steps, f"Loaded {len(ndc_to_rxnorm):,} NDC mappings", step_started)

    step_started = progress(2, total_steps, "Loading admissions.csv(.gz)")
    admissions = read_table(args.hosp_dir, "admissions", REQUIRED_TABLES["admissions"])
    progress(2, total_steps, f"Loaded {len(admissions):,} admission rows", step_started)
    step_started = progress(3, total_steps, "Loading diagnoses_icd.csv(.gz)")
    diagnoses = read_table(
        args.hosp_dir, "diagnoses_icd", REQUIRED_TABLES["diagnoses_icd"]
    )
    progress(3, total_steps, f"Loaded {len(diagnoses):,} diagnosis rows", step_started)
    step_started = progress(4, total_steps, "Loading procedures_icd.csv(.gz)")
    procedures = read_table(
        args.hosp_dir, "procedures_icd", REQUIRED_TABLES["procedures_icd"]
    )
    progress(4, total_steps, f"Loaded {len(procedures):,} procedure rows", step_started)
    step_started = progress(5, total_steps, "Loading prescriptions.csv(.gz)")
    prescriptions = read_table(
        args.hosp_dir, "prescriptions", REQUIRED_TABLES["prescriptions"]
    )
    progress(5, total_steps, f"Loaded {len(prescriptions):,} prescription rows", step_started)

    raw_counts = {
        "admissions": len(admissions),
        "diagnoses": len(diagnoses),
        "procedures": len(procedures),
        "prescriptions": len(prescriptions),
    }

    step_started = progress(6, total_steps, "Applying admission and ICD-9 filters")
    admission_mask = pd.Series(True, index=admissions.index)
    if exclude_newborn:
        newborn = admissions["admission_type"].astype("string").str.upper().eq("NEWBORN")
        admission_mask &= newborn.fillna(False).eq(False)
    if exclude_deaths:
        deaths = pd.to_datetime(admissions["deathtime"], errors="coerce").notna()
        admission_mask &= ~deaths
    eligible_admissions = admissions[admission_mask]
    eligible_ids = set(eligible_admissions["hadm_id"])

    diagnoses["icd_version"] = pd.to_numeric(diagnoses["icd_version"], errors="coerce")
    procedures["icd_version"] = pd.to_numeric(procedures["icd_version"], errors="coerce")
    diagnoses = diagnoses[diagnoses["icd_version"] == 9].copy()
    procedures = procedures[procedures["icd_version"] == 9].copy()
    diagnoses["icd_code"] = diagnoses["icd_code"].map(normalize_code)
    procedures["icd_code"] = procedures["icd_code"].map(normalize_code)
    diagnoses = diagnoses.dropna(subset=["icd_code", "hadm_id", "subject_id"])
    procedures = procedures.dropna(subset=["icd_code", "hadm_id", "subject_id"])
    diagnoses = diagnoses[diagnoses["hadm_id"].isin(eligible_ids)]
    procedures = procedures[procedures["hadm_id"].isin(eligible_ids)]
    progress(6, total_steps, f"Retained {len(eligible_ids):,} eligible admissions", step_started)

    step_started = progress(7, total_steps, "Mapping medication NDC codes to RxNorm")
    prescriptions["ndc"] = prescriptions["ndc"].map(normalize_ndc)
    prescriptions = prescriptions.dropna(subset=["ndc", "hadm_id", "subject_id"])
    prescriptions = prescriptions[prescriptions["hadm_id"].isin(eligible_ids)]
    mapped_medication_rows = int(prescriptions["ndc"].isin(ndc_to_rxnorm).sum())
    prescriptions = prescriptions[prescriptions["ndc"].isin(ndc_to_rxnorm)].copy()
    prescriptions["rxnorm"] = prescriptions["ndc"].map(ndc_to_rxnorm)
    progress(7, total_steps, f"Mapped {mapped_medication_rows:,} medication rows", step_started)

    step_started = progress(8, total_steps, "Filtering infrequent medical codes")
    diagnoses, diag_codes = filter_frequent_codes(
        diagnoses, "icd_code", args.min_code_count
    )
    procedures, procedure_codes = filter_frequent_codes(
        procedures, "icd_code", args.min_code_count
    )
    prescriptions, ndc_codes = filter_frequent_codes(
        prescriptions, "ndc", args.min_code_count
    )
    progress(
        8,
        total_steps,
        f"Retained {len(diag_codes):,} diagnosis, {len(procedure_codes):,} procedure, and {len(ndc_codes):,} NDC codes",
        step_started,
    )

    step_started = progress(9, total_steps, "Aggregating modalities into admission-level EHRs")
    diagnoses = diagnoses.sort_values(["subject_id", "hadm_id", "seq_num"])
    procedures = procedures.sort_values(["subject_id", "hadm_id", "seq_num"])
    diag_by_admission = aggregate_modality(diagnoses, ["icd_code"]).rename(
        columns={"icd_code": "ICD9_DIAG"}
    )
    procedure_by_admission = aggregate_modality(procedures, ["icd_code"]).rename(
        columns={"icd_code": "ICD9_PROCE"}
    )
    medication_by_admission = aggregate_modality(
        prescriptions, ["ndc", "rxnorm"]
    ).rename(columns={"ndc": "NDC", "rxnorm": "ATC"})

    common = diag_by_admission.merge(
        procedure_by_admission, on=["subject_id", "hadm_id"], how="inner"
    ).merge(medication_by_admission, on=["subject_id", "hadm_id"], how="inner")
    progress(9, total_steps, f"Constructed {len(common):,} complete admission-level EHRs", step_started)

    step_started = progress(10, total_steps, "Selecting single-visit disease cohorts")
    visit_counts = common.groupby("subject_id")["hadm_id"].nunique()
    single_visit_subjects = set(visit_counts[visit_counts == 1].index)
    cohort = common[common["subject_id"].isin(single_visit_subjects)].copy()
    cohort["disease"] = cohort["ICD9_DIAG"].str.split(",").str[0]
    primary_counts = cohort["disease"].value_counts()
    if requested_cohort_codes is None:
        selected_diseases = list(primary_counts.head(args.num_cohorts).index)
        if len(selected_diseases) < args.num_cohorts:
            raise ValueError(
                f"Only {len(selected_diseases)} disease cohorts survived preprocessing; "
                f"{args.num_cohorts} were requested."
            )
    else:
        selected_diseases = list(requested_cohort_codes)
        missing_cohorts = [code for code in selected_diseases if primary_counts.get(code, 0) == 0]
        if missing_cohorts:
            raise ValueError(
                "Requested primary-diagnosis cohorts have no eligible EHRs after "
                f"filtering: {missing_cohorts}. Check the MIMIC version and filters."
            )
    cohort = cohort[cohort["disease"].isin(selected_diseases)].copy()
    cohort = cohort.rename(columns={"subject_id": "SUBJECT_ID", "hadm_id": "HADM_ID"})
    progress(
        10,
        total_steps,
        f"Selected {len(selected_diseases)} cohorts with {len(cohort):,} EHRs",
        step_started,
    )

    step_started = progress(11, total_steps, "Splitting data and constructing pair labels")
    train, valid, test = split_admissions(cohort, args.seed)
    validate_split_support(
        train, valid, test, selected_diseases, args.positive_per_anchor
    )

    output_columns = [
        "SUBJECT_ID",
        "HADM_ID",
        "ICD9_DIAG",
        "ICD9_PROCE",
        "NDC",
        "ATC",
        "disease",
    ]
    for split_name, frame in (("train", train), ("valid", valid), ("test", test)):
        frame[output_columns].to_csv(
            args.output_dir / f"{split_name}_admissions.csv", index=False
        )

    train_pos, train_neg = construct_labels(
        train,
        args.output_dir / "train_label.csv",
        args.positive_per_anchor,
        args.negative_per_anchor,
        args.seed + 1,
    )
    valid_pos, valid_neg = construct_labels(
        valid,
        args.output_dir / "valid_label.csv",
        args.positive_per_anchor,
        args.negative_per_anchor,
        args.seed + 2,
    )
    progress(11, total_steps, "Created train/validation/test splits and pair labels", step_started)

    step_started = progress(12, total_steps, "Building vocabulary, PMI relations, and audit reports")
    all_selected = pd.concat([train, valid, test], ignore_index=True)
    vocabulary = {
        "diag_codes": sorted(
            {code for values in all_selected["ICD9_DIAG"] for code in values.split(",")}
        ),
        "proce_codes": sorted(
            {code for values in all_selected["ICD9_PROCE"] for code in values.split(",")}
        ),
        "atc_codes": sorted(
            {code for values in all_selected["ATC"] for code in values.split(",")}
        ),
    }
    with (args.output_dir / "vocab.pkl").open("wb") as handle:
        pickle.dump(vocabulary, handle)

    relation_specs = [
        ("ICD9_DIAG", "ICD9_PROCE", "diag_proce_rel.csv"),
        ("ICD9_DIAG", "NDC", "diag_pres_rel.csv"),
        ("ICD9_PROCE", "NDC", "proce_pres_rel.csv"),
    ]
    relation_counts: dict[str, int] = {}
    for head, tail, filename in relation_specs:
        rows = construct_relation(all_selected, head, tail)
        write_relation(rows, args.output_dir / filename)
        relation_counts[filename] = len(rows)

    titles = disease_titles(args.hosp_dir)
    cohort_summary = (
        all_selected.groupby("disease", as_index=False)
        .agg(n_ehrs=("HADM_ID", "nunique"))
        .sort_values("n_ehrs", ascending=False)
    )
    cohort_summary["long_title"] = cohort_summary["disease"].map(titles)
    cohort_summary.to_csv(args.output_dir / "cohort_summary.csv", index=False)

    split_summary = (
        pd.concat(
            [
                frame.assign(split=name)
                for name, frame in (("train", train), ("valid", valid), ("test", test))
            ],
            ignore_index=True,
        )
        .groupby(["split", "disease"], as_index=False)
        .agg(n_ehrs=("HADM_ID", "nunique"))
    )
    split_summary.to_csv(args.output_dir / "split_summary.csv", index=False)

    report: dict[str, object] = {
        "parameters": {
            "min_code_count": args.min_code_count,
            "num_cohorts": args.num_cohorts,
            "cohort_mode": effective_cohort_mode,
            "cohort_codes": selected_diseases,
            "positive_per_anchor": args.positive_per_anchor,
            "negative_per_anchor": args.negative_per_anchor,
            "seed": args.seed,
            "icd_version": 9,
            "exclude_newborn": exclude_newborn,
            "exclude_in_hospital_deaths": exclude_deaths,
            "split": [0.6, 0.2, 0.2],
        },
        "raw_rows": raw_counts,
        "eligible_admissions_after_optional_filters": len(eligible_ids),
        "mapped_medication_rows": mapped_medication_rows,
        "frequent_code_counts": {
            "diagnosis": len(diag_codes),
            "procedure": len(procedure_codes),
            "ndc": len(ndc_codes),
        },
        "common_admissions": len(common),
        "single_visit_admissions": int(
            common.loc[
                common["subject_id"].isin(single_visit_subjects), "hadm_id"
            ].nunique()
        ),
        "selected_diseases": selected_diseases,
        "cohort_counts": {
            code: int((cohort["disease"] == code).sum()) for code in selected_diseases
        },
        "split_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
        "pair_counts": {
            "train_positive": train_pos,
            "train_negative": train_neg,
            "valid_positive": valid_pos,
            "valid_negative": valid_neg,
        },
        "relation_counts_before_loader_pmi_gt_1": relation_counts,
        "vocabulary_sizes": {key: len(value) for key, value in vocabulary.items()},
    }
    with (args.output_dir / "preprocessing_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    progress(12, total_steps, "Finished writing MHGRL artifacts", step_started)
    return report


def main() -> None:
    args = parse_args()
    report = preprocess(args)
    print(json.dumps(report, indent=2))
    print(f"\nMHGRL MIMIC-IV artifacts written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
