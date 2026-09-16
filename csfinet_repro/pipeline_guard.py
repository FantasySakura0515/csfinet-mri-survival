"""Monitor postprocess and delivery watchers and restart only missing unfinished processes."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .files import sha256, write_json
from .segmentation_guard import _claim_lock, pid_alive


def should_restart(status, pid, alive=pid_alive):
    if status == "completed":
        return False
    if status not in {"waiting_for_segmentation", "waiting_for_postprocess", "running", "auditing",
                      "packaging_release", "failed"}:
        raise ValueError(f"Unexpected pipeline suite status: {status!r}")
    return not alive(pid)


def _read_state(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
        return None


def _commit(worktree):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=worktree, text=True, encoding="utf-8"
    ).strip()


def _spawn(command, worktree, stdout_path, stderr_path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = stdout_path.open("a", encoding="utf-8", newline="\n")
    stderr = stderr_path.open("a", encoding="utf-8", newline="\n")
    try:
        process = subprocess.Popen(command, cwd=worktree, stdout=stdout, stderr=stderr)
    finally:
        stdout.close()
        stderr.close()
    return process.pid


def guard(project_root, postprocess_worktree, delivery_worktree, postprocess_pid,
          delivery_pid, tag="v2", interval=30, max_restarts=3):
    root = Path(project_root).resolve()
    post_worktree, delivery_worktree = Path(postprocess_worktree).resolve(), Path(delivery_worktree).resolve()
    if not tag.isalnum() or interval < 5 or max_restarts < 1:
        raise ValueError("Invalid pipeline guard tag, interval, or restart limit")
    runs, logs = root / "runs", root / "logs"
    post_state = runs / f"postprocess-suite-{tag}.json"
    delivery_state = runs / f"delivery-suite-{tag}.json"
    seg_state = runs / f"segmentation-suite-{tag}.json"
    state_path, lock_path = runs / f"pipeline-guard-{tag}.json", runs / f"pipeline-guard-{tag}.lock"
    patients = root / "data" / "manifests" / "cohort" / "patients.csv"
    raw = root / "data" / "raw" / "brats2020-nifti"
    config = post_worktree / "configs" / "reconstruction-v21.json"
    expected = {
        "postprocess_commit": _commit(post_worktree),
        "delivery_commit": _commit(delivery_worktree),
        "config_sha256": sha256(config),
        "split_sha256": sha256(patients),
    }
    commands = {
        "postprocess": [sys.executable, "-u", "-m", "csfinet_repro.postprocess_suite",
                        "--config", str(config), "--patients", str(patients), "--root", str(raw),
                        "--runs", str(runs), "--results", str(root / "results"),
                        "--artifacts", str(root / "artifacts"), "--cache", str(root / "data" / "cache"),
                        "--segmentation-suite", str(seg_state), "--tag", tag, "--poll-seconds", "60"],
        "delivery": [sys.executable, "-u", "-m", "csfinet_repro.delivery_suite",
                     "--project-root", str(root), "--postprocess-suite", str(post_state),
                     "--tag", tag, "--poll-seconds", "60"],
    }
    processes = {"postprocess": int(postprocess_pid), "delivery": int(delivery_pid)}
    unreadable = {"postprocess": 0, "delivery": 0}
    states = {"postprocess": post_state, "delivery": delivery_state}
    worktrees = {"postprocess": post_worktree, "delivery": delivery_worktree}
    record = {"status": "monitoring", "scope": "postprocess_and_delivery_recovery_guard", **expected,
              "interval_seconds": interval, "max_restarts_per_watcher": max_restarts,
              "processes": processes, "restarts": {"postprocess": [], "delivery": []}}
    _claim_lock(lock_path)
    try:
        while True:
            complete = []
            for name in ("postprocess", "delivery"):
                state = _read_state(states[name])
                if state is None:
                    unreadable[name] += 1
                    record[f"{name}_observed_status"] = "temporarily_unreadable"
                    record[f"{name}_unreadable_checks"] = unreadable[name]
                    if pid_alive(processes[name]) or unreadable[name] < 2:
                        continue
                    status = "failed"
                else:
                    unreadable[name] = 0
                    record[f"{name}_unreadable_checks"] = 0
                    status = state.get("status")
                    record[f"{name}_observed_status"] = status
                    if name == "postprocess":
                        if (state.get("git_commit") != expected["postprocess_commit"]
                                or state.get("config_sha256") != expected["config_sha256"]
                                or state.get("split_sha256") != expected["split_sha256"]):
                            raise ValueError("Postprocess watcher lineage mismatch")
                    elif state.get("git_commit") != expected["delivery_commit"]:
                        raise ValueError("Delivery watcher lineage mismatch")
                if status == "completed":
                    complete.append(name)
                    continue
                if not should_restart(status, processes[name]):
                    continue
                events = record["restarts"][name]
                if len(events) >= max_restarts:
                    raise RuntimeError(f"{name} watcher restart limit exhausted")
                event = {"number": len(events) + 1,
                         "started_at": datetime.now(timezone.utc).isoformat(),
                         "reason": "suite_unfinished_and_watcher_pid_not_alive"}
                events.append(event)
                processes[name] = _spawn(
                    commands[name], worktrees[name], logs / f"{name}-suite-v2.stdout.log",
                    logs / f"{name}-suite-v2.stderr.log",
                )
                record["processes"][name] = processes[name]
                event["pid"] = processes[name]
                print(f"Pipeline guard restarted {name} watcher as PID {processes[name]}", flush=True)
            record["last_checked_at"] = datetime.now(timezone.utc).isoformat()
            if len(complete) == 2:
                record["status"] = "completed"
                write_json(state_path, record)
                return record
            write_json(state_path, record)
            time.sleep(interval)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error),
                      failed_at=datetime.now(timezone.utc).isoformat())
        write_json(state_path, record)
        raise
    finally:
        if lock_path.exists() and lock_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--postprocess-worktree", required=True)
    parser.add_argument("--delivery-worktree", required=True)
    parser.add_argument("--postprocess-pid", required=True, type=int)
    parser.add_argument("--delivery-pid", required=True, type=int)
    parser.add_argument("--tag", default="v2")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--max-restarts", type=int, default=3)
    guard(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
