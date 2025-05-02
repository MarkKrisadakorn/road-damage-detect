import os
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

# Configuration
base_dir = "path/RDD_Kfold"
num_folds = 5
model_types = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"] 
epochs = 3
imgsz = 640
batch_size = 8
patience = 20  # Early stopping patience
device = 0  

# Create timestamp for unique run identifier
timestamp = datetime.now().strftime("%Y%m%d__%H_%M_%S")

# Function to train model on a specific fold
def train_fold(model_type, fold_num, project_dir):
    print(f"\n{'='*50}")
    print(f"Training {model_type} on Fold {fold_num}/{num_folds}")
    print(f"{'='*50}")
    
    # Set paths for this fold
    fold_dir = os.path.join(base_dir, f"fold{fold_num}")
    data_yaml = os.path.join(fold_dir, "data.yaml")
    
    # Initialize model - will download if not present locally
    model = YOLO(model_type)
    
    # Set training arguments - using same batch size for all models
    args = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,  # Using the same batch size for all models
        "patience": patience,
        "device": device,
        "name": f"{model_type.split('.')[0]}_fold{fold_num}",
        "project": project_dir,
        "exist_ok": True
    }
    
    # Start timer
    start_time = time.time()
    
    # Train model
    try:
        results = model.train(**args)
        training_success = True
    except Exception as e:
        print(f"Error training {model_type} on fold {fold_num}: {e}")
        results = None
        training_success = False
        e_msg = str(e)
    
    # End timer
    training_time = time.time() - start_time
    
    # Return validation metrics and training time
    if training_success:
        metrics = model.metrics
        return {
            "model": model_type,
            "fold": fold_num,
            "training_time": training_time,
            "metrics": metrics,
            "results": results,
            "success": True
        }
    else:
        return {
            "model": model_type,
            "fold": fold_num,
            "training_time": training_time,
            "metrics": None,
            "results": None,
            "success": False,
            "error": e_msg if 'e_msg' in locals() else "Unknown error"
        }

# Main execution
if __name__ == "__main__":
    # Create main results directory
    main_results_dir = os.path.join(base_dir, f"results_{timestamp}")
    os.makedirs(main_results_dir, exist_ok=True)
    
    # Dictionary to store results for all models
    all_models_results = {}
    
    # Train each model on all folds
    for model_type in model_types:
        print(f"\n{'#'*60}")
        print(f"# Starting training for {model_type}")
        print(f"{'#'*60}")
        
        # Create directory for this model's results
        model_results_dir = os.path.join(main_results_dir, model_type.split('.')[0])
        os.makedirs(model_results_dir, exist_ok=True)
        
        # Store results for this model
        model_metrics = []
        model_results = []
        
        # Train on each fold
        for fold in range(1, num_folds + 1):
            result = train_fold(model_type, fold, model_results_dir)
            model_results.append(result)
            if result["success"]:
                model_metrics.append(result["metrics"])
        
        # Calculate average metrics across folds if we have successful runs
        successful_runs = [r for r in model_results if r["success"]]
        if successful_runs:
            print(f"\n{'-'*50}")
            print(f"Results Summary for {model_type}")
            print(f"{'-'*50}")
            
            # First, check available attributes in the metrics object
            # This is safer than assuming specific attributes
            if model_metrics:
                sample_metric = model_metrics[0]
                # Print available attributes for debugging
                print("Available metrics attributes:", dir(sample_metric))
                print("Available box metrics attributes:", dir(sample_metric.box) if hasattr(sample_metric, 'box') else "No box attribute")
                
                # Extract key metrics safely with proper attribute verification
                try:
                    # Try to extract precision (mAP) - use getattr with default to avoid errors
                    precision = [getattr(m.box, 'map', 0.0) for m in model_metrics]
                    
                    # For recall, it might be stored differently in newer versions - try possible naming conventions
                    # In newer YOLOv8 versions, it might just be 'r' or some other abbreviation
                    if hasattr(model_metrics[0].box, 'r'):
                        recall = [m.box.r for m in model_metrics]
                    elif hasattr(model_metrics[0].box, 'recall'):
                        recall = [m.box.recall for m in model_metrics]
                    else:
                        # Default to zeros if not found, but log this issue
                        print("Warning: Recall metric not found with expected attribute names")
                        recall = [0.0] * len(model_metrics)
                    
                    # mAP50 and mAP50-95 are usually available
                    map50 = [m.box.map50 for m in model_metrics]
                    map50_95 = [m.box.map50_95 for m in model_metrics]
                    
                    # Calculate means and standard deviations
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
                    
                    # Print summary results
                    print(f"Precision: {metrics_summary['precision']['mean']:.4f} ± {metrics_summary['precision']['std']:.4f}")
                    print(f"Recall: {metrics_summary['recall']['mean']:.4f} ± {metrics_summary['recall']['std']:.4f}")
                    print(f"mAP50: {metrics_summary['mAP50']['mean']:.4f} ± {metrics_summary['mAP50']['std']:.4f}")
                    print(f"mAP50-95: {metrics_summary['mAP50-95']['mean']:.4f} ± {metrics_summary['mAP50-95']['std']:.4f}")
                    
                    # Save metrics as JSON
                    with open(os.path.join(model_results_dir, "cross_validation_metrics.json"), "w") as f:
                        json.dump(metrics_summary, f, indent=4)
                    
                    # Plot metrics for each fold
                    plt.figure(figsize=(12, 10))
                    
                    # Plot mAP50
                    plt.subplot(2, 2, 1)
                    plt.bar(range(1, len(successful_runs) + 1), map50)
                    plt.axhline(y=metrics_summary['mAP50']['mean'], color='r', linestyle='-')
                    plt.title(f"mAP50 by Fold (Avg: {metrics_summary['mAP50']['mean']:.4f})")
                    plt.xlabel("Fold")
                    plt.ylabel("mAP50")
                    
                    # Plot mAP50-95
                    plt.subplot(2, 2, 2)
                    plt.bar(range(1, len(successful_runs) + 1), map50_95)
                    plt.axhline(y=metrics_summary['mAP50-95']['mean'], color='r', linestyle='-')
                    plt.title(f"mAP50-95 by Fold (Avg: {metrics_summary['mAP50-95']['mean']:.4f})")
                    plt.xlabel("Fold")
                    plt.ylabel("mAP50-95")
                    
                    # Plot Precision
                    plt.subplot(2, 2, 3)
                    plt.bar(range(1, len(successful_runs) + 1), precision)
                    plt.axhline(y=metrics_summary['precision']['mean'], color='r', linestyle='-')
                    plt.title(f"Precision by Fold (Avg: {metrics_summary['precision']['mean']:.4f})")
                    plt.xlabel("Fold")
                    plt.ylabel("Precision")
                    
                    # Plot Recall
                    plt.subplot(2, 2, 4)
                    plt.bar(range(1, len(successful_runs) + 1), recall)
                    plt.axhline(y=metrics_summary['recall']['mean'], color='r', linestyle='-')
                    plt.title(f"Recall by Fold (Avg: {metrics_summary['recall']['mean']:.4f})")
                    plt.xlabel("Fold")
                    plt.ylabel("Recall")
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(model_results_dir, "cross_validation_metrics.png"))
                    
                    # Store summary for this model
                    all_models_results[model_type] = metrics_summary
                
                except Exception as e:
                    print(f"Error processing metrics for {model_type}: {e}")
                    # Save available metrics data for debugging
                    if model_metrics:
                        with open(os.path.join(model_results_dir, "raw_metrics_debug.json"), "w") as f:
                            # Convert metrics to dictionary for JSON serialization
                            debug_data = {
                                f"fold_{i+1}": {
                                    "available_attrs": dir(m),
                                    "box_attrs": dir(m.box) if hasattr(m, 'box') else "No box attribute"
                                }
                                for i, m in enumerate(model_metrics)
                            }
                            json.dump(debug_data, f, indent=4)
                    all_models_results[model_type] = {"error": f"Metrics processing error: {str(e)}"}
            else:
                print(f"No metrics data available for {model_type}")
                all_models_results[model_type] = {"error": "No metrics data available"}
        else:
            print(f"No successful runs for {model_type}")
            all_models_results[model_type] = {"error": "No successful runs"}
    
    # Create comparison between models
    if all_models_results:
        # Extract model names and key metrics
        model_names = []
        map50_means = []
        map50_stds = []
        map50_95_means = []
        map50_95_stds = []
        
        for model_name, metrics in all_models_results.items():
            if "error" not in metrics:
                model_names.append(model_name.split('.')[0])
                map50_means.append(metrics["mAP50"]["mean"])
                map50_stds.append(metrics["mAP50"]["std"])
                map50_95_means.append(metrics["mAP50-95"]["mean"])
                map50_95_stds.append(metrics["mAP50-95"]["std"])
        
        if model_names:
            # Create comparison plot
            plt.figure(figsize=(14, 6))
            
            # Plot mAP50 comparison
            plt.subplot(1, 2, 1)
            x = np.arange(len(model_names))
            plt.bar(x, map50_means, yerr=map50_stds, alpha=0.7, capsize=5)
            plt.xticks(x, model_names, rotation=45)
            plt.title("mAP50 Comparison Across Models")
            plt.ylabel("mAP50")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Plot mAP50-95 comparison
            plt.subplot(1, 2, 2)
            plt.bar(x, map50_95_means, yerr=map50_95_stds, alpha=0.7, capsize=5)
            plt.xticks(x, model_names, rotation=45)
            plt.title("mAP50-95 Comparison Across Models")
            plt.ylabel("mAP50-95")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(main_results_dir, "model_comparison.png"))
            
            # Save overall comparison to JSON
            comparison = {
                model: {
                    "mAP50": {"mean": mean, "std": std},
                    "mAP50-95": {"mean": mean_95, "std": std_95}
                }
                for model, mean, std, mean_95, std_95 in zip(model_names, map50_means, map50_stds, map50_95_means, map50_95_stds)
            }
            
            with open(os.path.join(main_results_dir, "models_comparison.json"), "w") as f:
                json.dump(comparison, f, indent=4)
    
    print(f"\nAll training complete! Results saved to {main_results_dir}")