import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data.preprocess_mimic4 import parse_args, preprocess


class Mimic4PreprocessingTest(unittest.TestCase):
    def test_cli_defaults_to_frequent_cohorts_and_original_admission_filters(self):
        with patch.object(
            sys,
            "argv",
            ["preprocess_mimic4.py", "--hosp-dir", "mimic-iv/hosp"],
        ):
            args = parse_args()

        self.assertEqual(args.cohort_mode, "most-frequent")
        self.assertTrue(args.exclude_newborn)
        self.assertTrue(args.exclude_in_hospital_deaths)

    def test_builds_mhgrl_contract_and_excludes_icd10(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            hosp = root / "hosp"
            output = root / "mimic4"
            hosp.mkdir()

            disease_codes = ["0389", "4019", "53081"]
            procedure_codes = ["0066", "8872", "9904"]
            ndc_codes = ["00064100133", "00069077038", "00085043104"]
            rxnorm_codes = ["545106", "616287", "746189"]

            admissions = []
            diagnoses = []
            procedures = []
            prescriptions = []
            for cohort_index in range(3):
                for offset in range(20):
                    subject_id = cohort_index * 100 + offset + 1
                    hadm_id = 100000 + subject_id
                    admissions.append(
                        {
                            "subject_id": subject_id,
                            "hadm_id": hadm_id,
                            "admittime": "2110-01-01 00:00:00",
                            "dischtime": "2110-01-03 00:00:00",
                            "deathtime": None,
                            "admission_type": "URGENT",
                        }
                    )
                    diagnoses.extend(
                        [
                            {
                                "subject_id": subject_id,
                                "hadm_id": hadm_id,
                                "seq_num": 1,
                                "icd_code": disease_codes[cohort_index],
                                "icd_version": 9,
                            },
                            {
                                "subject_id": subject_id,
                                "hadm_id": hadm_id,
                                "seq_num": 2,
                                "icd_code": "I10",
                                "icd_version": 10,
                            },
                        ]
                    )
                    procedures.extend(
                        [
                            {
                                "subject_id": subject_id,
                                "hadm_id": hadm_id,
                                "seq_num": 1,
                                "icd_code": procedure_codes[cohort_index],
                                "icd_version": 9,
                            },
                            {
                                "subject_id": subject_id,
                                "hadm_id": hadm_id,
                                "seq_num": 2,
                                "icd_code": "0DBN0ZZ",
                                "icd_version": 10,
                            },
                        ]
                    )
                    prescriptions.append(
                        {
                            "subject_id": subject_id,
                            "hadm_id": hadm_id,
                            "ndc": ndc_codes[cohort_index],
                        }
                    )

            pd.DataFrame(admissions).to_csv(hosp / "admissions.csv", index=False)
            pd.DataFrame(diagnoses).to_csv(hosp / "diagnoses_icd.csv", index=False)
            pd.DataFrame(procedures).to_csv(hosp / "procedures_icd.csv", index=False)
            pd.DataFrame(prescriptions).to_csv(hosp / "prescriptions.csv", index=False)
            pd.DataFrame(
                {
                    "icd_code": disease_codes,
                    "icd_version": [9, 9, 9],
                    "long_title": ["Septicemia", "Hypertension", "Esophageal reflux"],
                }
            ).to_csv(hosp / "d_icd_diagnoses.csv", index=False)

            mapping_path = root / "ndc2rxnorm_mapping.txt"
            mapping_path.write_text(
                repr(dict(zip(ndc_codes, rxnorm_codes))), encoding="utf-8"
            )
            args = argparse.Namespace(
                hosp_dir=hosp,
                output_dir=output,
                ndc_rxnorm_map=mapping_path,
                min_code_count=1,
                cohort_mode="most-frequent",
                cohort_codes=None,
                num_cohorts=3,
                exclude_newborn=False,
                exclude_in_hospital_deaths=False,
                positive_per_anchor=1,
                negative_per_anchor=1,
                seed=1002,
            )

            report = preprocess(args)

            expected = {
                "train_admissions.csv",
                "valid_admissions.csv",
                "test_admissions.csv",
                "train_label.csv",
                "valid_label.csv",
                "vocab.pkl",
                "diag_proce_rel.csv",
                "diag_pres_rel.csv",
                "proce_pres_rel.csv",
                "ndc2rxnorm_mapping.txt",
                "cohort_summary.csv",
                "split_summary.csv",
                "preprocessing_report.json",
            }
            self.assertTrue(expected.issubset({path.name for path in output.iterdir()}))
            self.assertEqual(report["split_sizes"], {"train": 36, "valid": 12, "test": 12})

            all_admissions = pd.concat(
                [
                    pd.read_csv(output / f"{split}_admissions.csv", dtype=str)
                    for split in ("train", "valid", "test")
                ],
                ignore_index=True,
            )
            self.assertNotIn("I10", ",".join(all_admissions["ICD9_DIAG"]))
            self.assertNotIn("0DBN0ZZ", ",".join(all_admissions["ICD9_PROCE"]))
            self.assertIn("0389", set(all_admissions["disease"]))
            self.assertIn("0066", set(all_admissions["ICD9_PROCE"]))
            self.assertEqual(set(all_admissions["ATC"]), set(rxnorm_codes))

            train_labels = pd.read_csv(output / "train_label.csv", sep="\t")
            self.assertEqual(set(train_labels["label"]), {0, 1})
            self.assertEqual(len(train_labels), 72)

    def test_default_mode_excludes_newborn_and_death_records(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            hosp = root / "hosp"
            output = root / "mimic4"
            hosp.mkdir()

            disease_codes = ["41401", "V3001", "V3000", "4019", "V290", "53081"]
            disease_names = [
                "Coronary atherosclerosis of native coronary artery",
                "Single liveborn, born in hospital, delivered by cesarean delivery",
                "Single liveborn, born in hospital, delivered without mention of cesarean delivery",
                "Unspecified essential hypertension",
                "Observation for suspected infectious condition",
                "Esophageal reflux",
            ]
            ndc = "00064100133"
            rxnorm = "545106"
            admissions = []
            diagnoses = []
            procedures = []
            prescriptions = []

            for cohort_index, disease_code in enumerate(disease_codes):
                for offset in range(20):
                    subject_id = cohort_index * 100 + offset + 1
                    hadm_id = 200000 + subject_id
                    is_neonatal = disease_code in {"V3001", "V3000", "V290"}
                    admissions.append(
                        {
                            "subject_id": subject_id,
                            "hadm_id": hadm_id,
                            "admittime": "2110-01-01 00:00:00",
                            "dischtime": "2110-01-03 00:00:00",
                            "deathtime": "2110-01-02 00:00:00" if offset == 0 else None,
                            "admission_type": "NEWBORN" if is_neonatal else "URGENT",
                        }
                    )
                    diagnoses.append(
                        {
                            "subject_id": subject_id,
                            "hadm_id": hadm_id,
                            "seq_num": 1,
                            "icd_code": disease_code,
                            "icd_version": 9,
                        }
                    )
                    procedures.append(
                        {
                            "subject_id": subject_id,
                            "hadm_id": hadm_id,
                            "seq_num": 1,
                            "icd_code": "9904",
                            "icd_version": 9,
                        }
                    )
                    prescriptions.append(
                        {"subject_id": subject_id, "hadm_id": hadm_id, "ndc": ndc}
                    )

            pd.DataFrame(admissions).to_csv(hosp / "admissions.csv", index=False)
            pd.DataFrame(diagnoses).to_csv(hosp / "diagnoses_icd.csv", index=False)
            pd.DataFrame(procedures).to_csv(hosp / "procedures_icd.csv", index=False)
            pd.DataFrame(prescriptions).to_csv(hosp / "prescriptions.csv", index=False)
            pd.DataFrame(
                {
                    "icd_code": disease_codes,
                    "icd_version": [9] * len(disease_codes),
                    "long_title": disease_names,
                }
            ).to_csv(hosp / "d_icd_diagnoses.csv", index=False)

            mapping_path = root / "ndc2rxnorm_mapping.txt"
            mapping_path.write_text(repr({ndc: rxnorm}), encoding="utf-8")
            args = argparse.Namespace(
                hosp_dir=hosp,
                output_dir=output,
                ndc_rxnorm_map=mapping_path,
                min_code_count=1,
                cohort_codes=None,
                num_cohorts=3,
                positive_per_anchor=0,
                negative_per_anchor=1,
                seed=1002,
            )

            report = preprocess(args)

            expected = {"41401", "4019", "53081"}
            self.assertEqual(set(report["selected_diseases"]), expected)
            self.assertEqual(report["cohort_counts"], {code: 19 for code in report["selected_diseases"]})
            self.assertEqual(report["eligible_admissions_after_optional_filters"], 57)
            self.assertEqual(sum(report["split_sizes"].values()), 57)
            summary = pd.read_csv(output / "cohort_summary.csv", dtype={"disease": str})
            self.assertEqual(set(summary["disease"]), expected)


if __name__ == "__main__":
    unittest.main()
