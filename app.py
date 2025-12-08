import hashlib
import json
import logging
import os
import random
import secrets
import sqlite3
import threading
import time
from typing import List

import click

from datasets import load_dataset
from flask import Flask, flash, redirect, render_template, request, session, url_for

from benchmark import ProblemTestState, get_problem_set
from judge import LightCPVerifierJudge, ProblemNotFoundError, SupportedLanguage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_env_from_file(env_path: str | None) -> bool:
    if not env_path:
        return False
    env_path = os.path.abspath(env_path)
    if not os.path.exists(env_path):
        return False
    try:
        with open(env_path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        logger.info("Loaded environment overrides from %s", env_path)
        return True
    except OSError as exc:
        logger.warning("Failed to read env file %s: %s", env_path, exc)
        return False


_project_root = os.path.dirname(__file__)
_env_candidates = [
    os.environ.get("APP_ENV_FILE"),
    os.path.join(_project_root, "aws.env"),
    os.path.join(_project_root, ".env"),
]
for candidate in _env_candidates:
    if _load_env_from_file(candidate):
        break

DATASET_NAME = "QAQAQAQAQ/LiveCodeBench-Pro"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "change-me")
app.config["DATABASE_PATH"] = os.environ.get(
    "APP_DATABASE_PATH",
    os.path.join(os.path.dirname(__file__), "app.db"),
)
app.config["JUDGE_WORKERS"] = int(os.environ.get("JUDGE_WORKERS", "2"))
app.config["APP_ENV"] = os.environ.get("APP_ENV", "local").lower()
app.config["RESULTS_DISABLED"] = app.config["APP_ENV"] == "prod"
APP_PASSWORD = os.environ.get("APP_PASSWORD", "MITcoding")
PASSWORD_SESSION_KEY = "is_authenticated"
SQLITE_INT_MAX = 2**63 - 1

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) must be set to access the gated dataset"
    )

logger.info("Loading dataset '%s' ...", DATASET_NAME)
_dataset_kwargs = {"token": HF_TOKEN}
try:
    DATASET = load_dataset(DATASET_NAME, **_dataset_kwargs)
except TypeError:
    # Older versions of datasets expect use_auth_token instead of token.
    DATASET = load_dataset(DATASET_NAME, use_auth_token=HF_TOKEN)
PROBLEM_SET = get_problem_set(DATASET)

PROBLEMS_BY_DIFFICULTY: dict[str, List[str]] = {"easy": [], "medium": [], "hard": []}
for pid, info in PROBLEM_SET.items():
    difficulty = (info.difficulty or "").lower()
    if difficulty in PROBLEMS_BY_DIFFICULTY:
        PROBLEMS_BY_DIFFICULTY[difficulty].append(pid)
logger.info(
    "Loaded %d problems (easy=%d, medium=%d, hard=%d)",
    len(PROBLEM_SET),
    len(PROBLEMS_BY_DIFFICULTY["easy"]),
    len(PROBLEMS_BY_DIFFICULTY["medium"]),
    len(PROBLEMS_BY_DIFFICULTY["hard"]),
)

grading_lock = threading.Lock()
active_grading_runs: set[str] = set()


def get_db_connection():
    conn = sqlite3.connect(app.config["DATABASE_PATH"])
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                user_identifier TEXT NOT NULL,
                seed INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        duplicates_removed = conn.execute(
            """
            DELETE FROM users
            WHERE id NOT IN (
                SELECT MIN(id) FROM users GROUP BY user_identifier
            )
            """
        ).rowcount
        if duplicates_removed:
            logger.warning(
                "Removed %d duplicate user IDs prior to adding unique constraint.",
                duplicates_removed,
            )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_user_identifier ON users(user_identifier)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_problems (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                user_identifier TEXT NOT NULL,
                problem_id TEXT NOT NULL,
                title TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                statement TEXT NOT NULL,
                position INTEGER NOT NULL,
                UNIQUE(run_id, problem_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                user_identifier TEXT NOT NULL,
                problem_id TEXT NOT NULL,
                solution TEXT,
                judge_result TEXT,
                submission_sid INTEGER,
                metadata TEXT,
                UNIQUE(run_id, problem_id)
            )
            """
        )
    conn.close()


init_db()


def results_disabled() -> bool:
    return bool(app.config.get("RESULTS_DISABLED"))


def is_authenticated() -> bool:
    return session.get(PASSWORD_SESSION_KEY, False) is True


def mark_authenticated():
    session[PASSWORD_SESSION_KEY] = True


def user_exists(user_identifier: str) -> bool:
    conn = get_db_connection()
    row = conn.execute(
        "SELECT 1 FROM users WHERE user_identifier = ? LIMIT 1",
        (user_identifier,),
    ).fetchone()
    conn.close()
    return row is not None


def derive_seed_from_user(user_identifier: str) -> int:
    digest = hashlib.sha256(user_identifier.encode("utf-8")).digest()
    seed_value = int.from_bytes(digest[:8], "big", signed=False)
    return seed_value % SQLITE_INT_MAX  # keep value within SQLite INTEGER bounds


def select_problems(seed_value: int) -> list[ProblemTestState]:
    for difficulty, bucket in PROBLEMS_BY_DIFFICULTY.items():
        if len(bucket) < 2:
            raise ValueError(f"Not enough {difficulty} problems to sample")

    rng = random.Random(seed_value)
    selected_ids = (
        rng.sample(PROBLEMS_BY_DIFFICULTY["easy"], 2)
        + rng.sample(PROBLEMS_BY_DIFFICULTY["medium"], 2)
        + rng.sample(PROBLEMS_BY_DIFFICULTY["hard"], 2)
    )
    return [PROBLEM_SET[pid] for pid in selected_ids]


def store_user_session(run_id: str, user_identifier: str, seed_value: int, problems: list[ProblemTestState]):
    conn = get_db_connection()
    with conn:
        conn.execute(
            "INSERT INTO users (run_id, user_identifier, seed) VALUES (?, ?, ?)",
            (run_id, user_identifier, seed_value),
        )
        conn.execute(
            "DELETE FROM user_problems WHERE run_id = ?",
            (run_id,),
        )
        conn.execute(
            "DELETE FROM submissions WHERE run_id = ?",
            (run_id,),
        )
        for position, problem in enumerate(problems):
            conn.execute(
                """
                INSERT INTO user_problems (
                    run_id, user_identifier, problem_id, title, difficulty, statement, position
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    user_identifier,
                    problem.problem_id,
                    problem.problem_title,
                    problem.difficulty,
                    problem.problem_statement,
                    position,
                ),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO submissions (
                    run_id, user_identifier, problem_id, solution, judge_result, submission_sid, metadata
                ) VALUES (?, ?, ?, '', NULL, NULL, NULL)
                """,
                (run_id, user_identifier, problem.problem_id),
            )
    conn.close()


def get_session_context():
    run_id = session.get("run_id")
    user_identifier = session.get("user_identifier")
    problem_ids = session.get("problem_ids", [])
    if not run_id or not user_identifier or not problem_ids:
        return None
    return run_id, user_identifier, problem_ids


def get_problem_for_run(run_id: str, problem_id: str):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM user_problems WHERE run_id = ? AND problem_id = ?",
        (run_id, problem_id),
    ).fetchone()
    conn.close()
    return row


def get_saved_solution(run_id: str, problem_id: str) -> str:
    conn = get_db_connection()
    row = conn.execute(
        "SELECT solution FROM submissions WHERE run_id = ? AND problem_id = ?",
        (run_id, problem_id),
    ).fetchone()
    conn.close()
    return row["solution"] if row and row["solution"] else ""


def save_solution(run_id: str, user_identifier: str, problem_id: str, solution: str):
    conn = get_db_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO submissions (run_id, user_identifier, problem_id, solution)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(run_id, problem_id)
            DO UPDATE SET solution=excluded.solution, judge_result=NULL, submission_sid=NULL, metadata=NULL
            """,
            (run_id, user_identifier, problem_id, solution),
        )
    conn.close()


def fetch_submissions_for_run(run_id: str):
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT up.problem_id, up.title, up.difficulty, up.statement, sub.solution, sub.judge_result,
               sub.submission_sid, sub.metadata
        FROM user_problems up
        LEFT JOIN submissions sub
            ON sub.run_id = up.run_id AND sub.problem_id = up.problem_id
        WHERE up.run_id = ?
        ORDER BY up.position
        """,
        (run_id,),
    ).fetchall()
    conn.close()
    return rows


def list_all_runs():
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT run_id, user_identifier, created_at FROM users ORDER BY created_at"
    ).fetchall()
    conn.close()
    return rows


def pending_submission_count(run_id: str) -> int:
    conn = get_db_connection()
    row = conn.execute(
        """
        SELECT COUNT(*) as pending
        FROM submissions
        WHERE run_id = ? AND (judge_result IS NULL OR judge_result = '')
        """,
        (run_id,),
    ).fetchone()
    conn.close()
    return int(row["pending"] if row is not None else 0)


def _grade_rows_with_judge(judge: LightCPVerifierJudge, run_id: str, user_identifier: str, rows):
    for row in rows:
        problem_id = row["problem_id"]
        solution = row["solution"] or ""
        if not solution.strip():
            update_judge_result(run_id, problem_id, "No Submission", None, None)
            logger.info("Run %s problem %s had no submission; skipping judge", run_id, problem_id)
            continue

        try:
            submission_sid = judge.submit(problem_id, SupportedLanguage.CPP, solution)
            result = wait_for_result(judge, submission_sid)
            metadata = {"graded_at": time.time()}
            update_judge_result(run_id, problem_id, result, submission_sid, metadata)
            logger.info(
                "Judge result for run %s problem %s (sid=%s): %s",
                run_id,
                problem_id,
                submission_sid,
                result,
            )
        except ProblemNotFoundError:
            update_judge_result(run_id, problem_id, "Problem Not Found", None, None)
            logger.warning("Problem %s not found for run %s", problem_id, run_id)
        except TimeoutError as exc:
            update_judge_result(
                run_id,
                problem_id,
                "Judge Timeout",
                None,
                {"error": str(exc)},
            )
            logger.error("Judge timeout for run %s problem %s: %s", run_id, problem_id, exc)
        except Exception as exc:  # noqa: BLE001
            update_judge_result(
                run_id,
                problem_id,
                "Judge Failed",
                None,
                {"error": str(exc)},
            )
            logger.exception(
                "Judge failed for run %s problem %s", run_id, problem_id
            )


def update_judge_result(run_id: str, problem_id: str, result: str, submission_sid: int | None, metadata: dict | None):
    conn = get_db_connection()
    meta_json = json.dumps(metadata) if metadata else None
    with conn:
        conn.execute(
            """
            UPDATE submissions
            SET judge_result = ?, submission_sid = ?, metadata = ?
            WHERE run_id = ? AND problem_id = ?
            """,
            (result, submission_sid, meta_json, run_id, problem_id),
        )
    conn.close()


def wait_for_result(judge: LightCPVerifierJudge, submission_id: int, timeout_seconds: int = 300) -> str:
    start = time.time()
    while time.time() - start < timeout_seconds:
        result = judge.get_result(submission_id)
        if result != "Judging":
            return result
        time.sleep(1)
    raise TimeoutError("Judge response timed out")


def grade_user_submissions(run_id: str, user_identifier: str, judge: LightCPVerifierJudge | None = None):
    rows = fetch_submissions_for_run(run_id)
    if not rows:
        return

    logger.info(
        "Starting grading for run_id=%s user=%s (%d problems)", run_id, user_identifier, len(rows)
    )

    def handle_unavailable(exc: Exception):
        logger.exception("Judge service unavailable for run %s: %s", run_id, exc)
        for row in rows:
            problem_id = row["problem_id"]
            solution = row["solution"] or ""
            if not solution.strip():
                update_judge_result(run_id, problem_id, "No Submission", None, None)
                continue
            update_judge_result(
                run_id,
                problem_id,
                "Judge Unavailable",
                None,
                {"error": str(exc)},
            )
            logger.error(
                "Marked run %s problem %s as Judge Unavailable due to startup failure",
                run_id,
                problem_id,
            )

    try:
        if judge is None:
            with LightCPVerifierJudge(worker=app.config["JUDGE_WORKERS"]) as owned_judge:
                _grade_rows_with_judge(owned_judge, run_id, user_identifier, rows)
        else:
            _grade_rows_with_judge(judge, run_id, user_identifier, rows)
    except Exception as exc:  # noqa: BLE001
        handle_unavailable(exc)
        if judge is not None:
            raise


def start_grading_async(run_id: str, user_identifier: str) -> bool:
    """Kick off grading in a background thread once per run."""
    with grading_lock:
        if run_id in active_grading_runs:
            return False
        active_grading_runs.add(run_id)

    def _worker():
        try:
            grade_user_submissions(run_id, user_identifier)
        finally:
            with grading_lock:
                active_grading_runs.discard(run_id)

    thread = threading.Thread(
        target=_worker,
        name=f"grading-{run_id}",
        daemon=True,
    )
    thread.start()
    return True


def is_grading(run_id: str) -> bool:
    with grading_lock:
        return run_id in active_grading_runs


@app.before_request
def enforce_password_gate():
    exempt_endpoints = {"login", "static"}
    if request.endpoint is None:
        return
    if request.endpoint in exempt_endpoints:
        return
    if not is_authenticated():
        return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if is_authenticated():
        return redirect(url_for("index"))

    if request.method == "POST":
        password = request.form.get("password", "")
        if password == APP_PASSWORD:
            mark_authenticated()
            flash("Access granted.")
            return redirect(url_for("index"))
        flash("Incorrect password. Try again.")

    return render_template("password.html")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_identifier = request.form.get("user_id", "").strip()

        if not user_identifier:
            flash("User ID is required.")
            return render_template("index.html")

        if user_exists(user_identifier):
            flash("User ID already exists. Pick another one.")
            return render_template("index.html")

        seed_value = derive_seed_from_user(user_identifier)

        try:
            selected_problems = select_problems(seed_value)
        except ValueError as exc:
            flash(str(exc))
            return render_template("index.html")

        run_id = secrets.token_hex(8)
        store_user_session(run_id, user_identifier, seed_value, selected_problems)

        session["run_id"] = run_id
        session["user_identifier"] = user_identifier
        session["problem_ids"] = [problem.problem_id for problem in selected_problems]
        return redirect(url_for("problem_page", index=0))

    return render_template("index.html")


@app.route("/problem/<int:index>", methods=["GET", "POST"])
def problem_page(index: int):
    context = get_session_context()
    if not context:
        flash("Start a session from the home page first.")
        return redirect(url_for("index"))

    run_id, user_identifier, problem_ids = context
    if index < 0 or index >= len(problem_ids):
        flash("Invalid problem index.")
        return redirect(url_for("index"))

    problem_id = problem_ids[index]
    problem_row = get_problem_for_run(run_id, problem_id)
    if not problem_row:
        flash("Problem not found. Restart session.")
        return redirect(url_for("index"))

    if request.method == "POST":
        solution = request.form.get("solution", "")
        save_solution(run_id, user_identifier, problem_id, solution)
        next_index = index + 1
        if next_index >= len(problem_ids):
            if results_disabled():
                flash(
                    "Thank you for participating! Your answers are saved, and you can still edit them before final review."
                )
            else:
                started = start_grading_async(run_id, user_identifier)
                if started:
                    flash("Grading in progress. The results page will update shortly.")
                else:
                    flash("Grading already running. Check the results page for updates.")
            return redirect(url_for("results"))
        return redirect(url_for("problem_page", index=next_index))

    saved_solution = get_saved_solution(run_id, problem_id)
    return render_template(
        "problem.html",
        problem=problem_row,
        index=index,
        total=len(problem_ids),
        saved_solution=saved_solution,
        results_disabled=results_disabled(),
    )


@app.route("/results")
def results():
    context = get_session_context()
    if not context:
        flash("Session expired. Start again.")
        return redirect(url_for("index"))

    run_id, user_identifier, problem_ids = context

    if results_disabled():
        submissions = fetch_submissions_for_run(run_id)
        submission_map = {row["problem_id"]: row for row in submissions}
        review_rows = []
        for idx, pid in enumerate(problem_ids):
            row = submission_map.get(pid)
            review_rows.append(
                {
                    "index": idx,
                    "problem_id": pid,
                    "title": row["title"] if row else pid,
                    "difficulty": (row["difficulty"] or "").capitalize() if row else "Unknown",
                    "has_solution": bool(row and (row["solution"] or "").strip()),
                }
            )
        completed = sum(1 for entry in review_rows if entry["has_solution"])
        return render_template(
            "review.html",
            problems=review_rows,
            user_identifier=user_identifier,
            completed=completed,
            total=len(review_rows),
        )

    submissions = fetch_submissions_for_run(run_id)
    pending_count = sum(1 for row in submissions if not row["judge_result"])
    has_saved_code = any((row["solution"] or "").strip() for row in submissions)

    requested_grade = request.args.get("grade") == "1"
    should_trigger = requested_grade or pending_count > 0
    if should_trigger:
        started = start_grading_async(run_id, user_identifier)
        if requested_grade:
            if started:
                flash("Grading started. This page will refresh automatically while judging finishes.")
            else:
                flash("Grading is already running. This page will refresh automatically.")

    grading_active = is_grading(run_id)
    auto_refresh = grading_active or pending_count > 0

    return render_template(
        "results.html",
        submissions=submissions,
        user_identifier=user_identifier,
        pending_count=pending_count,
        grading_active=grading_active,
        has_saved_code=has_saved_code,
        auto_refresh=auto_refresh,
    )


@app.cli.command("grade-run")
@click.argument("run_id")
def grade_run_cli(run_id: str):
    """Grade a stored run using the local LightCPVerifier instance."""
    conn = get_db_connection()
    row = conn.execute(
        "SELECT user_identifier FROM users WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    conn.close()
    if not row:
        click.echo(f"No run found for run_id={run_id}.")
        return

    user_identifier = row["user_identifier"]
    click.echo(f"Grading run {run_id} for user {user_identifier}...")
    grade_user_submissions(run_id, user_identifier)
    click.echo("Done.")


@app.cli.command("grade-all")
@click.option(
    "--only-missing",
    is_flag=True,
    help="Skip runs that already have judge results for every problem.",
)
def grade_all_cli(only_missing: bool):
    """Grade every stored run (optionally only those with pending submissions)."""
    runs = list_all_runs()
    if not runs:
        click.echo("No runs found in the database.")
        return

    click.echo(f"Found {len(runs)} runs. Starting judge...")
    graded = 0
    skipped = 0
    with LightCPVerifierJudge(worker=app.config["JUDGE_WORKERS"]) as judge:
        for row in runs:
            run_id = row["run_id"]
            user_identifier = row["user_identifier"]
            if only_missing and pending_submission_count(run_id) == 0:
                skipped += 1
                continue

            click.echo(f"Grading run {run_id} ({user_identifier})...")
            try:
                grade_user_submissions(run_id, user_identifier, judge=judge)
                graded += 1
            except Exception as exc:  # noqa: BLE001
                click.echo(f"Failed to grade run {run_id}: {exc}", err=True)
    click.echo(f"Graded {graded} run(s). Skipped {skipped} run(s).")


if __name__ == "__main__":
    host = os.environ.get("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_RUN_PORT", "5000"))
    app.run(host=host, port=port, debug=True)
