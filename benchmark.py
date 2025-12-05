from datasets import load_dataset, DatasetDict
import pydantic
import typing
import api_interface
import json
import tqdm
import time
import logging
import random
from judge import LightCPVerifierJudge, SupportedLanguage, ProblemNotFoundError
from util import extract_longest_cpp_code

# *************************** Change this before use ****************************

# llm_instance = (
#     api_interface.ExampleLLM()
# )  # change this to the LLM class you want to benchmark on

# # change this to the number of workers you want to use in LightCPVerifier
# # recommended to be <= number of CPU physical cores
# worker = 8

# *******************************************************************************


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkResult(pydantic.BaseModel):
    problem_id: str
    problem_title: str
    difficulty: str
    platform: str
    text_response: str
    code: str | None
    judge_result: str
    response_meta: typing.Any

class ProblemTestState(pydantic.BaseModel):
    problem_id: str
    problem_title: str
    difficulty: str
    platform: str
    problem_statement: str
    text_response: str | None = None
    code: str | None = None
    submission_id: int | None = None
    judge_result: str = "Judging"
    response_meta: typing.Any = None

def get_problem_set(dataset: DatasetDict) -> dict[str, ProblemTestState]:
    problem_set = {}
    for split in dataset.values():
        for row in split:
            if row["problem_id"] not in problem_set:
                problem_set[row["problem_id"]] = ProblemTestState(**row)
    return problem_set

def print_stats(dataset: DatasetDict, problem_set: dict[str, ProblemTestState]):
    print("=" * 80)
    print("BENCHMARK STATISTICS")
    print("=" * 80)

    split_difficulty_stats = {}

    for split_name, split in dataset.items():
        split_difficulty_stats[split_name] = {}
        
        for row in split:
            problem_id = row["problem_id"]
            difficulty = row.get("difficulty", "unknown")

            if problem_id in problem_set:
                judge_result = problem_set[problem_id].judge_result
            else:
                judge_result = "Not Tested"

            if difficulty not in split_difficulty_stats[split_name]:
                split_difficulty_stats[split_name][difficulty] = {
                    "total": 0, 
                    "accepted": 0, 
                    "judge_results": {}
                }
            
            split_difficulty_stats[split_name][difficulty]["total"] += 1
            if judge_result == "Accepted":
                split_difficulty_stats[split_name][difficulty]["accepted"] += 1

            if judge_result not in split_difficulty_stats[split_name][difficulty]["judge_results"]:
                split_difficulty_stats[split_name][difficulty]["judge_results"][judge_result] = []
            split_difficulty_stats[split_name][difficulty]["judge_results"][judge_result].append(problem_id)

    for split_name in split_difficulty_stats:
        print(f"\n[SPLIT: {split_name.upper()}]")
        print("-" * 60)
        
        total_problems_in_split = 0
        total_accepted_in_split = 0
        
        for difficulty, stats in sorted(split_difficulty_stats[split_name].items()):
            total = stats["total"]
            accepted = stats["accepted"]
            accuracy = (accepted / total * 100) if total > 0 else 0.0
            
            print(f"\n{difficulty.upper()} Difficulty: {accepted}/{total} ({accuracy:.1f}%)")

            for judge_result, problem_ids in sorted(stats["judge_results"].items()):
                count = len(problem_ids)
                percentage = (count / total * 100) if total > 0 else 0.0
                print(f"  {judge_result:20s}: {count:3d} ({percentage:5.1f}%) - {', '.join(sorted(problem_ids))}")
            
            total_problems_in_split += total
            total_accepted_in_split += accepted

        overall_accuracy = (total_accepted_in_split / total_problems_in_split * 100) if total_problems_in_split > 0 else 0.0
        print(f"\nOVERALL for {split_name}: {total_accepted_in_split}/{total_problems_in_split} ({overall_accuracy:.1f}%)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    dataset = load_dataset("QAQAQAQAQ/LiveCodeBench-Pro")
    problem_set = get_problem_set(dataset)

    easy_problems = {
        pid: info
        for pid, info in problem_set.items()
        if dict(info).get("difficulty") == "easy"
    }
    medium_problems = {
        pid: info
        for pid, info in problem_set.items()
        if dict(info).get("difficulty") == "medium"
    }
    hard_problems = {
        pid: info
        for pid, info in problem_set.items()
        if dict(info).get("difficulty") == "hard"
    }

    random.seed(2139)

    chosen_easy = random.sample(list(easy_problems.keys()), k=2)
    chosen_medium = random.sample(list(medium_problems.keys()), k=2)
    chosen_hard = random.sample(list(hard_problems.keys()), k=2)

    selected_ids = chosen_easy + chosen_medium + chosen_hard
    selected_problems = [problem_set[pid] for pid in selected_ids]

    # -------------------------------------------------------
    # Display the selected problems to the human
    # todo: figure out how to display this
    # -------------------------------------------------------
    print("\n=== SELECTED PROBLEMS ===")
    for p in selected_problems:
        print("\n----------------------------------------------")
        print(f"ID: {p.problem_id}")
        print(f"Title: {p.problem_title}")
        print(f"Difficulty: {p.difficulty}")
        print(f"Statement:\n{p.problem_statement}")

    # -------------------------------------------------------
    # Collect a human response (placeholder for now)
    # todo: collect a human answer
    # -------------------------------------------------------
    # Example smoke test: known-wrong solution
    human_code = """#include <iostream>

        int main() {
            std::cout << "Hello World!" << std::endl;
            return 0;
        }"""

    # -------------------------------------------------------
    # Judging
    # todo: pass in a human answer
    # -------------------------------------------------------
    with LightCPVerifierJudge(worker=2) as judge:
        problem_id = selected_problems[0].problem_id
        sid = judge.submit(problem_id, SupportedLanguage.CPP, human_code)

        # poll until result is ready
        while True:
            result = judge.get_result(sid)
            print(f"Current result: {result}")
            if result != "Judging":
                break
            time.sleep(1)
        
        print(f"Final result: {result}")
