import argparse
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.preprocess_mimic4 import preprocess


class Mimic4PreprocessingTest(unittest.TestCase):
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
                num_cohorts=3,
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


if __name__ == "__main__":
    unittest.main()
