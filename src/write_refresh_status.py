"""Write metadata describing the latest successful data-refresh check."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = PROJECT_ROOT / "status" / "data-refresh.json"
SNAPSHOT_PATH = PROJECT_ROOT / "models" / "data_snapshot.sha256"

DATA_SOURCE = "davidcariboo/player-scores"
PACIFIC_TIME = ZoneInfo("America/Los_Angeles")


def environment_flag(name: str) -> bool:
    """Convert a true/false environment variable to a Boolean."""
    return os.environ.get(name, "false").lower() == "true"


def optional_integer(name: str) -> int | None:
    """Read an optional integer environment variable safely."""
    value = os.environ.get(name)

    if value and value.isdigit():
        return int(value)

    return None


def build_status() -> dict:
    """Build metadata for the current refresh run."""
    now_utc = datetime.now(timezone.utc)
    now_pacific = now_utc.astimezone(PACIFIC_TIME)

    data_changed = environment_flag("DATA_CHANGED")
    repository = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")

    workflow_url = None

    if repository and run_id:
        workflow_url = (
            f"https://github.com/{repository}/actions/runs/{run_id}"
        )

    snapshot = None

    if SNAPSHOT_PATH.exists():
        snapshot = SNAPSHOT_PATH.read_text(
            encoding="utf-8"
        ).strip()

    return {
        "status": "success",
        "source": DATA_SOURCE,
        "last_checked_utc": now_utc.isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z"),
        "last_checked_pacific": now_pacific.isoformat(
            timespec="seconds"
        ),
        "dataset_changed": data_changed,
        "model_retrained": data_changed,
        "dataset_snapshot_sha256": snapshot,
        "workflow_event": os.environ.get(
            "GITHUB_EVENT_NAME",
            "local",
        ),
        "workflow_run_number": optional_integer(
            "GITHUB_RUN_NUMBER"
        ),
        "workflow_run_url": workflow_url,
        "checked_commit": os.environ.get("GITHUB_SHA"),
    }


def main() -> None:
    """Write the status atomically or print a dry-run preview."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the status without writing a file.",
    )
    args = parser.parse_args()

    status = build_status()
    rendered_status = json.dumps(status, indent=2) + "\n"

    if args.dry_run:
        print(rendered_status)
        return

    STATUS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = STATUS_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        rendered_status,
        encoding="utf-8",
    )
    temporary_path.replace(STATUS_PATH)

    print(f"Refresh status written to: {STATUS_PATH}")
    print(rendered_status)


if __name__ == "__main__":
    main()