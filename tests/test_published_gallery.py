"""Check the released scientific evidence, not a second copy of solver formulas."""

import csv
import json
import re
from pathlib import Path

import pytest

from aeriallab.batch import load_catalog
from aeriallab.experiments import CASES, ExperimentConfig
from aeriallab.reporting import verify_report

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "docs/gallery"


@pytest.fixture(scope="module")
def captured():
    return {r["name"]: r for r in load_catalog(GALLERY)}


def test_publication_has_all_default_cases_from_one_clean_revision(captured):
    publication = json.loads((GALLERY / "publication.json").read_text(encoding="utf-8"))
    assert set(captured) == set(CASES)
    assert publication["case_count"] == 27
    for case, result in captured.items():
        assert result["configuration"] == ExperimentConfig(case).to_dict()
        assert result["provenance"]["git_revision"] == publication["source_revision"]
        assert result["provenance"]["git_dirty"] is False


@pytest.mark.parametrize("case", list(CASES))
def test_published_report_integrity_and_csv_agreement(case, captured):
    assert verify_report(GALLERY / case)
    result = captured[case]
    for table, rows in result["tables"].items():
        if not rows:
            continue
        with (GALLERY / case / ("table_" + table + ".csv")).open(encoding="utf-8", newline="") as stream:
            exported = list(csv.DictReader(stream))
        expected = [{k: "" if v is None else str(v) for k, v in row.items()} for row in rows]
        assert exported == expected
    with (GALLERY / case / "summary.csv").open(encoding="utf-8", newline="") as stream:
        exported = {row["metric"]: row["value"] for row in csv.DictReader(stream)}
    assert exported == {k: "" if v is None else str(v) for k, v in result["summary"].items()}


def test_repository_documentation_links_resolve_to_delivered_files():
    paths = [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]
    for path in paths:
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if link.startswith(("https://", "http://", "#")):
                continue
            target = link.split("#", 1)[0]
            assert (path.parent / target).exists(), (path, link)


def test_published_evidence_retains_reaction_mismatch_failure_and_conservation(captured):
    nominal = captured["reaction-aware-arm-motion"]["summary"]
    assert nominal["termination"] == "completed" and nominal["final_arm_error_rad"] < 0.01
    assert (
        captured["reaction-ignorant-baseline"]["summary"]["position_rmse_m"] > nominal["position_rmse_m"] * 10
    )
    assert captured["payload-model-mismatch"]["summary"]["final_arm_error_rad"] > 0.1
    assert captured["complete-actuator-loss"]["summary"]["termination"] == "flight-envelope-exceeded"
    assert captured["partial-actuator-loss"]["summary"]["completion_fraction"] < 1
    assert captured["joint-stop-base-reaction"]["summary"]["maximum_base_pitch_rate_rad_s"] > 0.01
    assert captured["payload-capture-release"]["summary"]["maximum_momentum_defect_n_s"] < 1e-12
