"""
Pure diagnosis function — no FastAPI dependency.
All delay-detection logic lives here.
"""

SEGMENTS = [
    "furnace_to_2nd_strike",
    "2nd_to_3rd_strike",
    "3rd_to_4th_strike",
    "4th_strike_to_aux_press",
    "aux_press_to_bath",
]

# Cause table from §1.1 — maps segment → list of probable causes
CAUSE_TABLE = {
    "furnace_to_2nd_strike": [
        "Billet pick", "gripper close", "grip retries",
        "trajectory", "permissions", "queues",
    ],
    "2nd_to_3rd_strike": [
        "Retraction", "gripper", "press/PLC handshake",
        "wait points", "regrip",
    ],
    "3rd_to_4th_strike": [
        "Retraction", "conservative trajectory", "synchronization",
        "positioning", "confirmations",
    ],
    "4th_strike_to_aux_press": [
        "Pick micro-corrections", "transfer",
        "queue at Auxiliary Press entry", "interlocks",
    ],
    "aux_press_to_bath": [
        "Retraction", "transport", "bath queues",
        "permissions", "bath deposit",
    ],
}


def compute_partials(piece: dict) -> dict:
    """
    Derive the 5 partial times from the 5 cumulative lifetime fields.
    Returns a dict mapping segment name -> float or None.
    If either operand is None/missing the result is None.
    """
    t2 = piece.get("lifetime_2nd_strike_s")
    t3 = piece.get("lifetime_3rd_strike_s")
    t4 = piece.get("lifetime_4th_strike_s")
    ta = piece.get("lifetime_auxiliary_press_s")
    tb = piece.get("lifetime_bath_s")

    def diff(a, b):
        if a is None or b is None:
            return None
        return round(a - b, 1)

    return {
        "furnace_to_2nd_strike":   t2,
        "2nd_to_3rd_strike":       diff(t3, t2),
        "3rd_to_4th_strike":       diff(t4, t3),
        "4th_strike_to_aux_press": diff(ta, t4),
        "aux_press_to_bath":       diff(tb, ta),
    }


def classify_segment(actual, reference: float) -> dict:
    """
    Apply §1.3 rule to one segment.
    Returns dict with actual_s, reference_s, deviation_s, penalized.
    """
    if actual is None:
        return {
            "actual_s": None,
            "reference_s": reference,
            "deviation_s": None,
            "penalized": None,
        }

    deviation = round(actual - reference, 1)

    if deviation > 5.0:
        penalized = None        # sensor anomaly
    elif deviation > 1.0:
        penalized = True
    else:
        penalized = False

    return {
        "actual_s": round(actual, 1),
        "reference_s": reference,
        "deviation_s": deviation,
        "penalized": penalized,
    }


def diagnose(piece: dict, reference_times: dict) -> dict:
    """
    Core diagnosis function.

    Args:
        piece: dict with keys:
            piece_id, die_matrix,
            lifetime_2nd_strike_s, lifetime_3rd_strike_s,
            lifetime_4th_strike_s, lifetime_auxiliary_press_s,
            lifetime_bath_s
        reference_times: loaded from reference_times.json

    Returns:
        Response dict matching the schema in §1.4.
        Raises KeyError if die_matrix is unknown.
    """
    matrix_key = str(int(piece["die_matrix"]))
    if matrix_key not in reference_times:
        raise KeyError(f"unknown die_matrix {piece['die_matrix']}")

    refs = reference_times[matrix_key]
    partials = compute_partials(piece)

    segments = []
    for seg in SEGMENTS:
        result = classify_segment(partials[seg], refs[seg])
        segments.append({"segment": seg, **result})

    # Piece is delayed if at least one segment is penalized=True
    delayed = any(s["penalized"] is True for s in segments)

    # Probable causes = union of cause lists for penalized segments, in order
    probable_causes = []
    for s in segments:
        if s["penalized"] is True:
            for cause in CAUSE_TABLE[s["segment"]]:
                if cause not in probable_causes:
                    probable_causes.append(cause)

    return {
        "piece_id": piece["piece_id"],
        "die_matrix": int(piece["die_matrix"]),
        "delay": delayed,
        "segments": segments,
        "probable_causes": probable_causes,
    }
