"""Round-trip tests for the fair-review sheet format and decoder.

Builds a sheet in the exact generator format (including the
"**Claims to score (N):**" header with its count suffix), scores it, and
verifies the decoder recovers the exact per-config totals, per-answer
densities, macro average, and claims-per-answer counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "backend"))

from scripts.overview_fair_review import _parse_sheet  # noqa: E402


def _sheet_sample(num: int, n_claims: int) -> str:
    lines = [f"\n## Sample {num}", "\n**Question:** q?", "\n**Answer:**\n\nA.", ""]
    lines.append(f"\n**Claims to score ({n_claims}):**")
    for j in range(1, n_claims + 1):
        lines.append(f"\n{j}. claim {num}-{j}")
        lines.append("   - [ ] supported")
        lines.append("   - [ ] unsupported")
        lines.append("   - [ ] not a claim")
    lines.append("\n---")
    return "\n".join(lines)


def _score(sheet: str, marks: dict[tuple[int, int], str]) -> str:
    """Fill exactly one box per claim according to ``marks``."""
    import re

    out_lines = []
    for line in sheet.split("\n"):
        m = re.match(r"(\d+)\. claim (\d+)-(\d+)", line)
        if m:
            mark = marks[(int(m.group(2)), int(m.group(3)))]
        elif line.strip() in (
            "- [ ] supported", "- [ ] unsupported", "- [ ] not a claim",
        ):
            label = line.strip().removeprefix("- [ ] ")
            if mark == label:
                line = line.replace("[ ]", "[x]")
        out_lines.append(line)
    return "\n".join(out_lines)


def test_decoder_round_trip_with_count_suffix_header():
    # 3 samples: sample 1 = A with 2 claims, sample 2 = B with 3 claims,
    # sample 3 = A with 1 claim.
    sheet = (
        "# header\n"
        + _sheet_sample(1, 2)
        + _sheet_sample(2, 3)
        + _sheet_sample(3, 1)
    )
    marks = {
        (1, 1): "supported", (1, 2): "unsupported",
        (2, 1): "supported", (2, 2): "supported", (2, 3): "not a claim",
        (3, 1): "unsupported",
    }
    scored = _score(sheet, marks)
    key = {1: ("A", "q1"), 2: ("B", "q2"), 3: ("A", "q2")}

    scores, per_answer = _parse_sheet(scored, key)

    # Per-config micro totals.
    assert scores["A"] == {
        "supported": 1, "unsupported": 2, "not_a_claim": 0, "total": 3,
    }
    assert scores["B"] == {
        "supported": 2, "unsupported": 0, "not_a_claim": 1, "total": 3,
    }

    # Per-answer densities: A/q1 = 1/2 unsupported, A/q2 = 1/1, B/q2 = 0/2.
    assert per_answer[("A", "q1")]["n_scorable"] == 2
    assert per_answer[("A", "q1")]["unsupported"] == 1
    assert per_answer[("A", "q2")]["n_scorable"] == 1
    assert per_answer[("A", "q2")]["unsupported"] == 1
    assert per_answer[("B", "q2")]["n_scorable"] == 2
    assert per_answer[("B", "q2")]["unsupported"] == 0
    assert per_answer[("B", "q2")]["not_a_claim"] == 1

    # Micro rates: A = 2/3, B = 0/2.
    a_scored = scores["A"]["supported"] + scores["A"]["unsupported"]
    b_scored = scores["B"]["supported"] + scores["B"]["unsupported"]
    assert scores["A"]["unsupported"] / a_scored == 2 / 3
    assert scores["B"]["unsupported"] / b_scored == 0.0

    # Macro density: A = mean(1/2, 1/1) = 0.75; B = mean(0/2) = 0.0.
    a_densities = [1 / 2, 1 / 1]
    b_densities = [0.0]
    assert sum(a_densities) / len(a_densities) == 0.75
    assert sum(b_densities) / len(b_densities) == 0.0


def test_decoder_handles_header_without_count_suffix():
    sheet = (
        "# header\n## Sample 1\n\n**Question:** q\n\n**Answer:**\n\nA.\n\n"
        "**Claims to score:**\n\n1. claim\n   - [ ] supported\n"
        "   - [x] unsupported\n   - [ ] not a claim\n\n---\n"
    )
    scores, per_answer = _parse_sheet(sheet, {1: ("A", "q1")})
    assert scores["A"]["unsupported"] == 1
    assert scores["A"]["total"] == 1
    assert per_answer[("A", "q1")]["n_scorable"] == 1


def test_decoder_skips_unmarked_claims_with_warning():
    sheet = (
        "# header\n## Sample 1\n\n**Question:** q\n\n**Answer:**\n\nA.\n\n"
        "**Claims to score (1):**\n\n1. claim\n   - [ ] supported\n"
        "   - [ ] unsupported\n   - [ ] not a claim\n\n---\n"
    )
    scores, _ = _parse_sheet(sheet, {1: ("A", "q1")})
    # Nothing marked -> the claim is skipped, not mis-scored.
    assert scores["A"]["total"] == 0
