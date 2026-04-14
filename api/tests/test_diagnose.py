import csv
import json
from pathlib import Path

import pytest

from diagnose import diagnose


HERE = Path(__file__).parent
REF_PATH = Path(__file__).parent.parent / "reference_times.json"
CSV_PATH = Path(__file__).parent.parent / "validation_pieces.csv"
EXPECTED_PATH = Path(__file__).parent.parent / "validation_expected.json"


@pytest.fixture(scope="session")
def reference_times():
    with open(REF_PATH) as f:
        return json.load(f)


def build_piece(reference_times: dict, die_matrix: int, partial_overrides: dict):
    ref = reference_times[str(die_matrix)]
    partials = {
        "furnace_to_2nd_strike": ref["furnace_to_2nd_strike"],
        "2nd_to_3rd_strike": ref["2nd_to_3rd_strike"],
        "3rd_to_4th_strike": ref["3rd_to_4th_strike"],
        "4th_strike_to_aux_press": ref["4th_strike_to_aux_press"],
        "aux_press_to_bath": ref["aux_press_to_bath"],
    }
    partials.update(partial_overrides)

    t2 = partials["furnace_to_2nd_strike"]
    t3 = t2 + partials["2nd_to_3rd_strike"]
    t4 = t3 + partials["3rd_to_4th_strike"]
    ta = t4 + partials["4th_strike_to_aux_press"]
    tb = ta + partials["aux_press_to_bath"]

    return {
        "piece_id": "TEST",
        "die_matrix": die_matrix,
        "lifetime_2nd_strike_s": round(t2, 1),
        "lifetime_3rd_strike_s": round(t3, 1),
        "lifetime_4th_strike_s": round(t4, 1),
        "lifetime_auxiliary_press_s": round(ta, 1),
        "lifetime_bath_s": round(tb, 1),
    }


@pytest.mark.parametrize("die_matrix", [4974, 5052, 5090, 5091])
def test_all_ok(reference_times, die_matrix):
    piece = build_piece(reference_times, die_matrix, {})
    result = diagnose(piece, reference_times)
    assert result["delay"] is False
    assert all(s["penalized"] is False for s in result["segments"])


@pytest.mark.parametrize("die_matrix", [4974, 5052, 5090, 5091])
def test_penalize_furnace_to_2nd(reference_times, die_matrix):
    piece = build_piece(reference_times, die_matrix, {"furnace_to_2nd_strike": reference_times[str(die_matrix)]["furnace_to_2nd_strike"] + 2.0})
    result = diagnose(piece, reference_times)
    assert result["delay"] is True
    assert result["segments"][0]["penalized"] is True


@pytest.mark.parametrize("die_matrix", [4974, 5052, 5090, 5091])
def test_penalize_2nd_to_3rd(reference_times, die_matrix):
    piece = build_piece(reference_times, die_matrix, {"2nd_to_3rd_strike": reference_times[str(die_matrix)]["2nd_to_3rd_strike"] + 2.0})
    result = diagnose(piece, reference_times)
    assert result["delay"] is True
    assert result["segments"][1]["penalized"] is True


@pytest.mark.parametrize("die_matrix", [4974, 5052, 5090, 5091])
def test_penalize_3rd_to_4th(reference_times, die_matrix):
    piece = build_piece(reference_times, die_matrix, {"3rd_to_4th_strike": reference_times[str(die_matrix)]["3rd_to_4th_strike"] + 2.0})
    result = diagnose(piece, reference_times)
    assert result["delay"] is True
    assert result["segments"][2]["penalized"] is True


@pytest.mark.parametrize("die_matrix", [4974, 5052, 5090, 5091])
def test_penalize_4th_to_aux(reference_times, die_matrix):
    piece = build_piece(reference_times, die_matrix, {"4th_strike_to_aux_press": reference_times[str(die_matrix)]["4th_strike_to_aux_press"] + 2.0})
    result = diagnose(piece, reference_times)
    assert result["delay"] is True
    assert result["segments"][3]["penalized"] is True


@pytest.mark.parametrize("die_matrix", [4974, 5052, 5090, 5091])
def test_penalize_aux_to_bath(reference_times, die_matrix):
    piece = build_piece(reference_times, die_matrix, {"aux_press_to_bath": reference_times[str(die_matrix)]["aux_press_to_bath"] + 2.0})
    result = diagnose(piece, reference_times)
    assert result["delay"] is True
    assert result["segments"][4]["penalized"] is True


@pytest.mark.parametrize("row", list(csv.DictReader(CSV_PATH.open())))
def test_golden_set(reference_times, row):
    with open(EXPECTED_PATH) as f:
        expected = json.load(f)
    expected_map = {r["piece_id"]: r for r in expected}

    # convert row values
    piece = {
        "piece_id": row["piece_id"],
        "die_matrix": int(row["die_matrix"]),
        "lifetime_2nd_strike_s": float(row["lifetime_2nd_strike_s"]) if row["lifetime_2nd_strike_s"] else None,
        "lifetime_3rd_strike_s": float(row["lifetime_3rd_strike_s"]) if row["lifetime_3rd_strike_s"] else None,
        "lifetime_4th_strike_s": float(row["lifetime_4th_strike_s"]) if row["lifetime_4th_strike_s"] else None,
        "lifetime_auxiliary_press_s": float(row["lifetime_auxiliary_press_s"]) if row["lifetime_auxiliary_press_s"] else None,
        "lifetime_bath_s": float(row["lifetime_bath_s"]) if row["lifetime_bath_s"] else None,
    }

    result = diagnose(piece, reference_times)

    # round to 1 decimal
    def round_floats(obj):
        if isinstance(obj, float):
            return round(obj, 1)
        if isinstance(obj, dict):
            return {k: round_floats(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [round_floats(v) for v in obj]
        return obj

    assert round_floats(result) == expected_map[piece["piece_id"]]
