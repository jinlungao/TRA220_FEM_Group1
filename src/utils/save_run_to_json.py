import json
import os

def save_run_to_json(metrics, filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))      # src/utils
    root_dir = os.path.abspath(os.path.join(base_dir, "..", ".."))  # root
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, filename)

    # if file not exist→ creat new one
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # add new metrics
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        data = [data]
    data.append(metrics)

    # rewrite
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def save_cpu_run_to_json(metrics, filename="cpu_fem_run.json"):
    save_run_to_json(metrics, filename)


def save_gpu_run_to_json(metrics, filename="gpu_fem_run.json"):
    save_run_to_json(metrics, filename)