import csv
import runpy
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_beir_retrieval.py"
)


@pytest.mark.parametrize(
    "error_type",
    [OSError, ValueError, csv.Error],
)
def test_preparation_errors_exit_cleanly(
    error_type,
    monkeypatch,
    capsys,
    tmp_path,
):
    namespace = runpy.run_path(str(SCRIPT_PATH))
    main = namespace["main"]

    def fail_preparation(*args, **kwargs):
        raise error_type("private internal diagnostic")

    monkeypatch.setitem(
        main.__globals__,
        "prepare_scifact",
        fail_preparation,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--source-dir",
            str(tmp_path / "source"),
            "--output-dir",
            str(tmp_path / "prepared"),
        ],
    )

    with pytest.raises(SystemExit) as caught:
        main()

    captured = capsys.readouterr()

    assert caught.value.code == 2
    assert captured.out == ""
    assert "Dataset preparation failed" in captured.err
    assert "private internal diagnostic" not in captured.err
    assert not (tmp_path / "prepared").exists()
