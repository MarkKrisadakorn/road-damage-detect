import os
import numpy as np
from ultralytics import YOLO
import time
import json
import gc
import torch
from datetime import datetime

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

base_dir = "path_to/RDD_Kfold"
num_folds = 5
model_types = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
epochs = 100
imgsz = 640
device = 0
batch_size = 16

timestamp = datetime.now().strftime("%Y%m%d__%H_%M_%S")

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory cleared. Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        return allocated < 0.1
    return True

def optimize_memory_usage():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

    gc.collect()

    torch.backends.cudnn.benchmark = True

    torch.use_deterministic_algorithms(True, warn_only=True)

    torch.autograd.set_detect_anomaly(False)

    print("Applied memory optimization techniques without changing batch size")

def train_fold(model_type, fold_num, project_dir):
    print(f"\n{'='*50}")
    print(f"Training {model_type} on Fold {fold_num}/{num_folds} with batch size {batch_size}")
    print(f"{'='*50}")

    fold_dir = os.path.join(base_dir, f"fold{fold_num}")
    data_yaml = os.path.join(fold_dir, "data.yaml")

    clear_gpu_memory()

    optimize_memory_usage()

    model = YOLO(model_type)

    args = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "device": device,
        "name": f"{model_type.split('.')[0]}_fold{fold_num}",
        "project": project_dir,
        "exist_ok": True,
        "batch": batch_size,
        "workers": 2,
        "cache": True,
        "amp": True,
        "lr0": 0.01,
        "momentum": 0.937,
        "optimizer": "SGD",
        "weight_decay": 0.00055,
        "close_mosaic": 10
    }

    start_time = time.time()

    training_success = False
    e_msg = None
    results = None

    try:
        results = model.train(**args)
        training_success = True
    except Exception as e:
        print(f"Error training {model_type} on fold {fold_num}: {e}")
        e_msg = str(e)
        clear_gpu_memory()

    training_time = time.time() - start_time

    del model
    clear_gpu_memory()

    if training_success:
        return {
            "model": model_type,
            "fold": fold_num,
            "batch_size": batch_size,
            "training_time": training_time,
            "metrics": results.metrics if hasattr(results, 'metrics') else None,
            "results": results,
            "success": True
        }
    else:
        return {
            "model": model_type,
            "fold": fold_num,
            "batch_size": batch_size,
            "training_time": training_time,
            "metrics": None,
            "results": None,
            "success": False,
            "error": e_msg if e_msg else "Unknown error"
        }

def setup_distributed_training(model_type):
    if torch.cuda.device_count() > 1:
        print(f"Setting up distributed training across {torch.cuda.device_count()} GPUs.")
        return True
    return False

def safe_extract_metrics(model_metrics):
    try:
        if model_metrics:
            sample_metric = model_metrics[0]
            print("Available metrics attributes on sample:", dir(sample_metric))
            if hasattr(sample_metric, 'box'):
                print("Available box metrics attributes on sample:", dir(sample_metric.box))

        precision = []
        recall = []
        map50 = []
        map50_95 = []

        for m in model_metrics:
            if hasattr(m, 'box'):
                if hasattr(m.box, 'map'):
                    precision.append(float(m.box.map))
                elif hasattr(m.box, 'precision'):
                    precision.append(float(m.box.precision))
                else:
                    precision.append(0.0)

                if hasattr(m.box, 'r'):
                    recall.append(float(m.box.r))
                elif hasattr(m.box, 'recall'):
                    recall.append(float(m.box.recall))
                else:
                    recall.append(0.0)

                if hasattr(m.box, 'map50'):
                    map50.append(float(m.box.map50))
                else:
                    map50.append(0.0)

                if hasattr(m.box, 'map50_95'):
                    map50_95.append(float(m.box.map50_95))
                else:
                    map50_95.append(0.0)
            else:
                print("Warning: No 'box' attribute found in metrics, trying direct access.")
                precision.append(float(getattr(m, 'map', 0.0)))
                recall.append(float(getattr(m, 'recall', 0.0)))
                map50.append(float(getattr(m, 'map50', 0.0)))
                map50_95.append(float(getattr(m, 'map50_95', 0.0)))

        metrics_summary = {
            "precision": {
                "mean": float(np.mean(precision)),
                "std": float(np.std(precision))
            },
            "recall": {
                "mean": float(np.mean(recall)),
                "std": float(np.std(recall))
            },
            "mAP50": {
                "mean": float(np.mean(map50)),
                "std": float(np.std(map50))
            },
            "mAP50-95": {
                "mean": float(np.mean(map50_95)),
                "std": float(np.std(map50_95))
            }
        }

        return metrics_summary, precision, recall, map50, map50_95, True

    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return None, [], [], [], [], False

if __name__ == "__main__":
    main_results_dir = os.path.join(base_dir, f"results_{timestamp}")
    os.makedirs(main_results_dir, exist_ok=True)

    all_models_results = {}

    for model_idx, model_type in enumerate(model_types):
        print(f"\n{'#'*60}")
        print(f"# Starting training for {model_type} ({model_idx+1}/{len(model_types)})")
        print(f"{'#'*60}")

        clear_gpu_memory()

        model_name = model_type.split('.')[0]
        model_results_dir = os.path.join(main_results_dir, model_name)
        os.makedirs(model_results_dir, exist_ok=True)

        model_metrics_per_fold = []
        model_raw_results_per_fold = []

        use_distributed = setup_distributed_training(model_type)

        for fold in range(1, num_folds + 1):
            memory_cleared = clear_gpu_memory()
            if not memory_cleared:
                print(f"Warning: Could not clear memory sufficiently before fold {fold}. Attempting to continue anyway.")

            result = train_fold(model_type, fold, model_results_dir)

            if not result["success"]:
                error_msg = result.get("error", "Unknown error")
                print(f"Failed to train {model_type} on fold {fold}.")
                print(f"Logged error: {error_msg}")

                failure_record = {
                    "model": model_type,
                    "fold": fold,
                    "batch_size": batch_size,
                    "error": error_msg,
                    "recommendation": "Consider using a more powerful GPU "
                }

                with open(os.path.join(model_results_dir, f"fold{fold}_failure.json"), "w") as f:
                    json.dump(failure_record, f, indent=4)

                for _ in range(3):
                    clear_gpu_memory()
                    time.sleep(1)

            model_raw_results_per_fold.append(result)
            if result["success"] and result["metrics"] is not None:
                model_metrics_per_fold.append(result["metrics"])

            for _ in range(2):
                clear_gpu_memory()
                time.sleep(1)

        metrics_summary, precision, recall, map50, map50_95, metrics_extracted_success = \
            safe_extract_metrics(model_metrics_per_fold)

        all_models_results[model_name] = {
            "raw_results_per_fold": model_raw_results_per_fold,
            "aggregated_metrics": metrics_summary,
            "precision_per_fold": precision,
            "recall_per_fold": recall,
            "map50_per_fold": map50,
            "map50_95_per_fold": map50_95,
            "metrics_extracted_success": metrics_extracted_success
        }

        with open(os.path.join(model_results_dir, f"{model_name}_aggregated_results.json"), "w") as f:
            serializable_results = []
            for res in model_raw_results_per_fold:
                temp_res = res.copy()
                if 'results' in temp_res and temp_res['results'] is not None:
                    temp_res['results'] = "Ultralytics Results object (not serialized)"
                serializable_results.append(temp_res)

            json.dump({
                "raw_results_per_fold": serializable_results,
                "aggregated_metrics": metrics_summary,
                "precision_per_fold": precision,
                "recall_per_fold": recall,
                "map50_per_fold": map50,
                "map50_95_per_fold": map50_95,
                "metrics_extracted_success": metrics_extracted_success
            }, f, indent=4)

    print(f"\nAll training complete! Results saved to {main_results_dir}")
