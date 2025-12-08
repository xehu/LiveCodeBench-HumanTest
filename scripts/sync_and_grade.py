#!/usr/bin/env python3
"""One-stop helper to sync the production SQLite DB and grade every run locally."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], *, env: dict[str, str] | None = None):
    print("+", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        raise


def sync_database(env_name: str, remote_path: str, local_path: Path, instance_number: int | None):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # On the Docker-based Elastic Beanstalk platform, the Flask app runs
    # inside a container and writes its SQLite database inside the
    # container filesystem (default: /app/app.db). Reading
    # /var/app/current/app.db on the host only shows the bundled source
    # file, not the live DB. Instead, exec into the running container and
    # stream the DB file from there.
    remote_cmd = (
        "sudo docker exec $(sudo docker ps --format '{{.ID}}' | head -n1) "
        f"cat {remote_path}"
    )
    ssh_cmd = [
        "eb",
        "ssh",
        env_name,
        "--command",
        remote_cmd,
    ]
    if instance_number is not None:
        ssh_cmd.extend(["-n", str(instance_number)])
    print("+", " ".join(ssh_cmd))
    result = subprocess.run(ssh_cmd, check=True, capture_output=True)
    local_path.write_bytes(result.stdout)


def grade_all(local_db: Path, only_missing: bool):
    env = os.environ.copy()
    env.setdefault("APP_ENV", "local")
    env["APP_DATABASE_PATH"] = str(local_db)
    cmd = ["flask", "--app", "app", "grade-all"]
    if only_missing:
        cmd.append("--only-missing")
    run_command(cmd, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync EB SQLite DB and grade all runs locally.")
    parser.add_argument("--eb-env", required=True, help="Elastic Beanstalk environment name (e.g., livecodebench-prod)")
    parser.add_argument(
        "--remote-path",
        default="/app/app.db",
        help="Path to the SQLite file inside the Docker container (default: /app/app.db)",
    )
    parser.add_argument(
        "--local-path",
        default="./prod-app.db",
        help="Where to store the downloaded DB locally (default: ./prod-app.db)",
    )
    parser.add_argument(
        "--instance-number",
        type=int,
        default=None,
        help="Optional EB instance number if your environment has multiple instances.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only grade runs that still have pending submissions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    local_path = Path(args.local_path).expanduser().resolve()
    print(f"Syncing database from {args.eb_env}:{args.remote_path} -> {local_path}")
    sync_database(args.eb_env, args.remote_path, local_path, args.instance_number)
    print("Database sync complete. Starting local grading...")
    grade_all(local_path, args.only_missing)
    print("All done.")


if __name__ == "__main__":
    main()
