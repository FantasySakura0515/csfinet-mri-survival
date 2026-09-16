"""Guard a long segmentation suite and resume it after an external process loss."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .files import sha256, write_json


def pid_alive(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            return False
        try:
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code))
            return bool(ok) and exit_code.value == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(process)
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def active_job_pid(suite):
    running = [job.get("pid") for job in suite.get("jobs", []) if job.get("status") == "running"]
    return running[0] if len(running) == 1 else None


def needs_restart(suite, alive):
    if suite.get("status") == "completed":
        return False
    if suite.get("status") not in {"running", "failed"}:
        raise ValueError(f"Unexpected segmentation suite status: {suite.get('status')!r}")
    pid = active_job_pid(suite)
    return pid is None or not alive(pid)


def _claim_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            owner = int(path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            owner = None
        if pid_alive(owner):
            raise RuntimeError(f"A live segmentation guard already owns {path}: PID {owner}")
        path.unlink()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{os.getpid()}\n")


def guard(config, patients, audit, root, runs, worktree, tag="v2", interval=30,
          max_restarts=3, state=None):
    paths = [Path(value).resolve() for value in (config, patients, audit, root, runs, worktree)]
    config, patients, audit, root, runs, worktree = paths
    if not tag.isalnum() or interval < 5 or max_restarts < 1:
        raise ValueError("Invalid guard tag, interval, or restart limit")
    suite_path = runs / f"segmentation-suite-{tag}.json"
    state_path = Path(state).resolve() if state else runs / f"segmentation-guard-{tag}.json"
    lock_path = runs / f"segmentation-guard-{tag}.lock"
    expected = {
        "config_sha256": sha256(config),
        "split_sha256": sha256(patients),
        "audit_sha256": sha256(audit),
        "training_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=worktree, text=True, encoding="utf-8"
        ).strip(),
    }
    command = [sys.executable, "-u", "-m", "csfinet_repro.run_suite",
               "--config", str(config), "--patients", str(patients), "--audit", str(audit),
               "--root", str(root), "--runs", str(runs), "--tag", tag]
    record = {"status": "monitoring", "scope": "segmentation_suite_recovery_guard",
              **expected, "suite": str(suite_path), "worktree": str(worktree),
              "interval_seconds": interval, "max_restarts": max_restarts, "restarts": []}
    _claim_lock(lock_path)
    try:
        while True:
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            for key in ("config_sha256", "split_sha256", "audit_sha256"):
                if suite.get(key) != expected[key]:
                    raise ValueError(f"Segmentation suite lineage mismatch: {key}")
            record.update(last_checked_at=datetime.now(timezone.utc).isoformat(),
                          observed_suite_status=suite.get("status"),
                          observed_job_pid=active_job_pid(suite))
            if suite.get("status") == "completed":
                record["status"] = "completed"
                write_json(state_path, record)
                return record
            if not needs_restart(suite, pid_alive):
                write_json(state_path, record)
                time.sleep(interval)
                continue
            if len(record["restarts"]) >= max_restarts:
                record.update(status="failed", reason="restart_limit_exhausted")
                write_json(state_path, record)
                raise RuntimeError("Segmentation guard restart limit exhausted")
            event = {"number": len(record["restarts"]) + 1,
                     "started_at": datetime.now(timezone.utc).isoformat(),
                     "reason": "suite_not_completed_and_recorded_worker_not_alive"}
            record["restarts"].append(event)
            record.update(status="restarting", active_restart=event["number"])
            write_json(state_path, record)
            print(f"Guard restart {event['number']}/{max_restarts}: resuming frozen segmentation suite", flush=True)
            code = subprocess.run(command, cwd=worktree).returncode
            event.update(finished_at=datetime.now(timezone.utc).isoformat(), exit_code=code)
            record.pop("active_restart", None)
            record["status"] = "monitoring"
            write_json(state_path, record)
            if code == 0:
                continue
            time.sleep(interval)
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error),
                      failed_at=datetime.now(timezone.utc).isoformat())
        write_json(state_path, record)
        raise
    finally:
        if lock_path.exists() and lock_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--patients", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--tag", default="v2")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--state")
    args = parser.parse_args()
    guard(**vars(args))


if __name__ == "__main__":
    main()
