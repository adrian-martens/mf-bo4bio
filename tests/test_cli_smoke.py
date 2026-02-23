import sys

import run.run_single_bo_scenario as single


def test_run_single_dispatches(monkeypatch):
    called = {"name": None}

    monkeypatch.setattr(
        single.qUCB, "run", lambda *args, **kwargs: called.update(name="qUCB")
    )
    monkeypatch.setattr(
        single.qLogEI, "run", lambda *args, **kwargs: called.update(name="qLogEI")
    )
    monkeypatch.setattr(
        single.GIBBON, "run", lambda *args, **kwargs: called.update(name="GIBBON")
    )

    argv = [
        "prog",
        "--test_type",
        "qUCB",
        "--iterations",
        "0",
        "--seed",
        "1",
        "--mbr_level",
        "7",
        "--clone_distribution",
        "alpha",
        "--date",
        "20260101",
        "--task_representation",
        "HYBRID",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    single.main()
    assert called["name"] == "qUCB"
