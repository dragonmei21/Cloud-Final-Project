"""
Generates validation_expected.json from validation_pieces.csv using the
same diagnose() logic as the API. Run this script whenever validation_pieces.csv
changes to keep the two files in sync.

Usage:
    python generate_validation.py
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from diagnose import diagnose


def main():
    base = Path(__file__).parent
    ref_path = base / "reference_times.json"
    csv_path = base / "validation_pieces.csv"
    out_path = base / "validation_expected.json"

    with open(ref_path) as f:
        reference_times = json.load(f)

    results = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            piece = {
                "piece_id": row["piece_id"],
                "die_matrix": int(row["die_matrix"]),
                "lifetime_2nd_strike_s": float(row["lifetime_2nd_strike_s"]) if row["lifetime_2nd_strike_s"] else None,
                "lifetime_3rd_strike_s": float(row["lifetime_3rd_strike_s"]) if row["lifetime_3rd_strike_s"] else None,
                "lifetime_4th_strike_s": float(row["lifetime_4th_strike_s"]) if row["lifetime_4th_strike_s"] else None,
                "lifetime_auxiliary_press_s": float(row["lifetime_auxiliary_press_s"]) if row["lifetime_auxiliary_press_s"] else None,
                "lifetime_bath_s": float(row["lifetime_bath_s"]) if row["lifetime_bath_s"] else None,
            }
            results.append(diagnose(piece, reference_times))

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Generated {out_path} with {len(results)} pieces.")


if __name__ == "__main__":
    main()
