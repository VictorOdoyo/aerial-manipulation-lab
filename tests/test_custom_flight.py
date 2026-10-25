import hashlib
import json

import pytest

from aeriallab.cli import main
from aeriallab.reporting import load_archive, verify_report


def test_custom_flight_cli_preserves_input_fingerprint_and_exact_duration(tmp_path):
    request = tmp_path / "flight.json"
    request.write_text(json.dumps({"duration_s": 0.017, "dt_s": 0.005, "seed": 7}), encoding="utf-8")
    target = tmp_path / "report"
    assert main(["flight-replay", "--config", str(request), "--output", str(target)]) == 0
    assert verify_report(target)
    saved = load_archive(target / "fields.npz")
    assert saved["configuration"]["input_sha256"] == hashlib.sha256(request.read_bytes()).hexdigest()
    assert saved["summary"]["achieved_duration_s"] == 0.017


def test_custom_cli_rejects_unknown_physical_keys(tmp_path):
    request = tmp_path / "flight.json"
    request.write_text('{"hidden_option":1}', encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["flight-replay", "--config", str(request), "--output", str(tmp_path / "report")])
