import pytest

from csfinet_repro.segmentation_guard import active_job_pid, needs_restart


def test_guard_only_restarts_an_unfinished_suite_without_a_live_worker():
    running = {"status": "running", "jobs": [
        {"status": "running", "pid": 42}, {"status": "pending"},
    ]}
    assert active_job_pid(running) == 42
    assert needs_restart(running, lambda pid: pid == 42) is False
    assert needs_restart(running, lambda pid: False) is True
    assert needs_restart({"status": "failed", "jobs": running["jobs"]}, lambda pid: False) is True
    assert needs_restart({"status": "completed", "jobs": []}, lambda pid: False) is False


def test_guard_rejects_ambiguous_or_unknown_state():
    assert active_job_pid({"jobs": [{"status": "running", "pid": 1},
                                    {"status": "running", "pid": 2}]}) is None
    assert needs_restart({"status": "running", "jobs": []}, lambda pid: True) is True
    with pytest.raises(ValueError, match="Unexpected"):
        needs_restart({"status": "waiting"}, lambda pid: False)
