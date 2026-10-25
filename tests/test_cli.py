import pytest

from aeriallab.cli import main


def test_cli_lists_actual_ev_catalog(capsys):
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "reaction-aware-arm-motion" in output and "spatial-free-rotation" in output


def test_cli_runs_an_independent_mtpa_study(tmp_path):
    from aeriallab.reporting import verify_report

    assert main(["run", "--case", "coupled-energy-convergence", "--output", str(tmp_path)]) == 0
    assert verify_report(tmp_path)


def test_cli_rejects_ambiguous_json_and_scale_overrides(tmp_path):
    path = tmp_path / "request.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--scale", "2", "--output", str(tmp_path)])
    assert error.value.code == 2


def test_cli_data_capture_render_and_replay_preserve_original_source(tmp_path):
    from aeriallab.reporting import load_archive

    source, replayed = tmp_path / "source", tmp_path / "replayed"
    main(["batch", "--cases", "coupled-energy-convergence", "--data-only", "--output", str(source)])
    main(["render", str(source)])
    main(["verify", str(source)])
    archive = source / "coupled-energy-convergence" / "fields.npz"
    provenance = load_archive(archive)["provenance"]
    main(["replay", str(archive), "--output", str(replayed)])
    assert load_archive(replayed / "fields.npz")["provenance"] == provenance
    main(["verify", str(replayed)])
    (replayed / "summary.csv").write_text("damaged")
    with pytest.raises(SystemExit) as error:
        main(["verify", str(replayed)])
    assert error.value.code == 2
