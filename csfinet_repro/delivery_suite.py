"""Wait for formal postprocessing, then build and audit the manuscript delivery package."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from .delivery import audit_delivery
from .files import sha256, write_json
from .release import create_release, validate_release


def completed_delivery(output, tag):
    output = Path(output)
    audit_path, manifest_path = output / "delivery-audit.json", output / "delivery-manifest.csv"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Existing delivery output is incomplete or invalid: {error}") from error
    if (audit.get("status") != "completed" or audit.get("tag") != tag
            or not manifest_path.is_file() or sha256(manifest_path) != audit.get("manifest_sha256")):
        raise ValueError("Existing delivery output failed completion or manifest validation")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--postprocess-suite", required=True)
    parser.add_argument("--tag", default="study")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if not args.tag.isalnum() or args.poll_seconds < 10:
        raise ValueError("Tag must be alphanumeric and polling interval >=10 seconds")
    root = Path(args.project_root).resolve()
    post_path = Path(args.postprocess_suite).resolve()
    state_path = root / "runs" / f"delivery-suite-{args.tag}.json"
    state = dict(status="waiting_for_postprocess", postprocess_suite=str(post_path),
                 git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 code_sha256={path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))})
    write_json(state_path, state)
    try:
        while True:
            if post_path.exists():
                post = json.loads(post_path.read_text(encoding="utf-8"))
                if post.get("status") == "failed":
                    raise RuntimeError("Formal postprocess suite failed")
                if post.get("status") == "completed":
                    break
            time.sleep(args.poll_seconds)
        state["status"] = "auditing"
        write_json(state_path, state)
        output = root / "results" / "delivery" / args.tag
        audit = completed_delivery(output, args.tag) if output.exists() else audit_delivery(root, args.tag, output)
        state.update(status="packaging_release",
                     delivery_audit_sha256=sha256(output / "delivery-audit.json"),
                     delivery_manifest_sha256=audit["manifest_sha256"])
        write_json(state_path, state)
        release_output = root / "release" / args.tag
        release = (validate_release(release_output, args.tag, audit["split_sha256"])
                   if release_output.exists() else create_release(root, args.tag, release_output))
        state.update(status="completed",
                     release_manifest_sha256=sha256(release_output / "release-manifest.json"),
                     release_git_commit=release["git_commit"])
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(state_path, state)


if __name__ == "__main__":
    main()
