# LiveCodeBench Pro - LLM Benchmarking Toolkit

<p align="center">

<img width="1415" height="420" alt="image" src="https://github.com/user-attachments/assets/f5a7a439-3526-4ff4-97ce-c325d4ddc8fb" />

</p>

This repository contains a benchmarking toolkit for evaluating Large Language Models (LLMs) on competitive programming tasks. The toolkit provides a standardized way to test your LLM's code generation capabilities across a diverse set of problems.

## Overview

LiveCodeBench Pro evaluates LLMs on their ability to generate solutions for programming problems. The benchmark includes problems of varying difficulty levels from different competitive programming platforms.

## Getting Started

### Prerequisites

- Ubuntu 20.04 or higher (or other distros with kernel version >= 3.10, and cgroup support. Refer to [go-judge](https://github.com/criyle/go-judge) for more details)
- Python 3.12 or higher
- pip package manager
- docker (for running the judge server), and ensure the user has permission to run docker commands

### Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install directly using `uv`:
   ```bash
   uv sync
   ```

2. Ensure Docker is installed and running:
   ```bash
   docker --version
   ```
   
   Make sure your user has permission to run Docker commands. On Linux, you may need to add your user to the docker group:
   ```bash
   sudo usermod -aG docker $USER
   ```
   Then log out and back in for the changes to take effect.

## How to Use

### Step 1: Implement Your LLM Interface

Create your own LLM class by extending the abstract `LLMInterface` class in `api_interface.py`. Your implementation needs to override the `call_llm` method.

Example:
```python
from api_interface import LLMInterface

class YourLLM(LLMInterface):
    def __init__(self):
        super().__init__()
        # Initialize your LLM client or resources here
        
    def call_llm(self, user_prompt: str):
        # Implement your logic to call your LLM with user_prompt
        # Return a tuple containing (response_text, metadata)
        
        # Example:
        response = your_llm_client.generate(user_prompt)
        return response.text, response.metadata
```

You can use the `ExampleLLM` class as a reference, which shows how to integrate with OpenAI's API.

### Step 2: Configure the Benchmark

Edit the `benchmark.py` file to use your LLM implementation:

```python
from your_module import YourLLM

# Replace this line:
llm_instance = YourLLM()  # Update with your LLM class
```

And change the number of judge workers (recommended to <= physical CPU cores).

### Step 3: Run the Benchmark

Execute the benchmark script:

```bash
python benchmark.py
```

The script will:
1. Load the LiveCodeBench-Pro dataset from Hugging Face
2. Process each problem with your LLM
3. Extract C++ code from LLM responses automatically
4. Submit solutions to the integrated judge system for evaluation
5. Collect judge results and generate comprehensive statistics
6. Save the results to `benchmark_result.json`

### (Optional) Step 4: Submit Your Results

Email your `benchmark_result.json` file to zz4242@nyu.edu to have it displayed on the leaderboard.

Please include the following information in your submission:
- LLM name and version
- Any specific details
- Contact information

## Human Workflow Web App

The repository now ships with a lightweight Flask app (`app.py`) that guides a human through the same sampling logic implemented in `benchmark.py`.

### Features

- Deterministic sampling of 2 easy, 2 medium, and 2 hard problems based on a numeric seed.
- Storage of `(user_id, run_id, seed, problem_ids)` plus per-problem solutions and judge verdicts inside `app.db` (SQLite).
- Step-by-step UI that shows each problem statement with a solution text box and an explicit "Next Page" button.
- Automatic grading via `LightCPVerifierJudge` once all solutions are submitted, with verdicts persisted alongside the solutions.

### Deploy and Run Locally

1. Ensure Docker is running because the judge container is required for final grading.
2. Install dependencies (`pip install -r requirements.txt`) if you have not already.
3. Export any optional settings:
    ```bash
    export FLASK_SECRET_KEY="change-this"
    export APP_DATABASE_PATH="/absolute/path/to/app.db"  # defaults to repo/app.db
    export APP_ENV=local  # set to "prod" to disable judging/results in production
    export JUDGE_WORKERS=2
    ```
    Environment variables can also be stored in a file. By default the app will look for a path specified via `APP_ENV_FILE`, then fall back to `aws.env`, then `.env` in the repository root and automatically load any `KEY=VALUE` pairs (including `HUGGINGFACEHUB_API_TOKEN`, AWS keys, etc.). Values already present in the process environment always win. The previously hosted judge service has been decommissioned, so leave `JUDGE_BASE_URL` unset and rely on the local LightCPVerifier container when grading.
4. Launch the web server:
    ```bash
    flask --app app run --host 0.0.0.0 --port 8000
    ```
5. Visit `http://localhost:8000`, enter your user ID and numeric seed, then follow the guided pages to submit solutions. Results can be revisited at any time on the final screen and are also stored in `app.db` for further analysis.

### Production Review-Only Mode

Set `APP_ENV=prod` before launching Flask on the production environment. This disables the automated judging workflow and repurposes the `/results` page into a "Review & Update" hub:

- Participants see a thank-you message confirming that every answer was recorded.
- Each problem is listed with its completion status plus an "Edit / Resubmit" button that links back to the appropriate problem page.
- Users can edit responses as many times as they want; each save immediately updates the SQLite database.

Use this mode whenever the Docker-based judge is unavailable—you can still capture high-quality human answers without exposing a broken results screen.

### How the Production Database Gets Updated

- Every time a participant clicks **Next Page**, **Finish & Review**, or the **Edit / Resubmit** button on a problem, the latest C++ solution is written to the `submissions` table (columns: `run_id`, `problem_id`, `solution`).
- The `users` table stores the original `run_id`, `user_identifier`, and deterministic seed. The `user_problems` table stores the sampled problems in order so the UI can rebuild each session.
- Nothing special is required from the participant beyond using the UI—the database always reflects the latest edits.

To inspect the live database over SSH, point `sqlite3` at the `APP_DATABASE_PATH` used in production (for example `/var/app/current/app.db`):

```bash
sqlite3 /var/app/current/app.db ".tables"
sqlite3 /var/app/current/app.db "SELECT run_id, user_identifier, seed FROM users;"
```

### Copying the Database from Production

1. Record the absolute path from `APP_DATABASE_PATH` on the production host.
2. Use `scp` (or your preferred file-transfer tool) to download the database. Example:
    ```bash
    scp user@prod-host:/var/app/current/app.db ./prod-app.db
    ```
3. Keep the file alongside this repository or anywhere convenient on your workstation.

### Running the Judge Locally on Pulled Data

1. Ensure Docker Desktop (or your local Docker daemon) is running—the LightCPVerifier container still grades solutions locally.
2. Export variables so Flask loads the copied database in local mode:
    ```bash
    export APP_DATABASE_PATH="$PWD/prod-app.db"
    export APP_ENV=local
    export HUGGINGFACEHUB_API_TOKEN="..."  # required for dataset access
    ```
3. For grading, you can either process a single run or everything at once:
     - Single run:
         ```bash
         flask --app app grade-run <run_id>
         ```
     - All runs (optionally only those with pending submissions):
         ```bash
         flask --app app grade-all --only-missing
         ```
     List available runs via `sqlite3`:
     ```bash
     sqlite3 "$APP_DATABASE_PATH" "SELECT run_id, user_identifier FROM users;"
     ```
4. After grading, you can launch the Flask app locally (with `APP_ENV=local`) and navigate to `/results` while logged in as that user to see the verdicts, or inspect the `submissions` table directly.

This workflow lets you keep production traffic judge-free while still gathering answers, then perform all heavy grading on a controlled local machine.

### One-Command Sync + Grade Helper

To make the entire workflow a single command, use the helper script:

```bash
python scripts/sync_and_grade.py --eb-env <ElasticBeanstalkEnvName> \
        --remote-path /var/app/current/app.db \
        --local-path ./prod-app.db \
        --only-missing
```

- Requires the Elastic Beanstalk CLI (`eb`) to be configured locally.
- The script runs `eb scp` to download the SQLite file, sets `APP_DATABASE_PATH`/`APP_ENV` for you, launches the local LightCPVerifier judge once via `flask --app app grade-all`, and writes verdicts back into the copied database.
- Omit `--only-missing` to force regrading of every stored run. Use `--instance-number N` if your EB environment has multiple EC2 instances and you need a specific one.

## Understanding the Codebase

### api_interface.py

This file defines the abstract interface for LLM integration:
- `LLMInterface`: Abstract base class with methods for LLM interaction
- `ExampleLLM`: Example implementation with OpenAI's GPT-4o

### benchmark.py

The main benchmarking script that:
- Loads the dataset
- Processes each problem through your LLM
- Extracts C++ code from responses
- Submits solutions to the judge system
- Collects results and generates statistics
- Saves comprehensive results with judge verdicts

### judge.py

Contains the judge system integration:
- `Judge`: Abstract base class for judge implementations
- `LightCPVerifierJudge`: LightCPVerifier integration for local solution evaluation
- Automatic problem data downloading from Hugging Face

### util.py

Utility functions for code processing:
- `extract_longest_cpp_code()`: Intelligent C++ code extraction from LLM responses


### Dataset

The benchmark uses the [QAQAQAQAQ/LiveCodeBench-Pro](https://huggingface.co/datasets/QAQAQAQAQ/LiveCodeBench-Pro) and [QAQAQAQAQ/LiveCodeBench-Pro-Testcase](https://huggingface.co/datasets/QAQAQAQAQ/LiveCodeBench-Pro-Testcase) datasets from Hugging Face, which contains competitive programming problems with varying difficulty levels.




## Contact

For questions or support, please contact us at zz4242@nyu.edu.
