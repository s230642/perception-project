import argparse
from detect_robot import detect_sequence

TASKS = {
    "detect": detect_sequence,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one pipeline stage on one sequence.")
    parser.add_argument("--dataset", required=True, help="e.g. data/1_no_obstacle-2")
    parser.add_argument("--task", required=True, choices=TASKS.keys())
    args = parser.parse_args()

    TASKS[args.task](args.dataset)