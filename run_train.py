import os
import numpy as np
from ultralytics import YOLO
import time
import json
import gc
import torch
from datetime import datetime

# Set environment variable to help with memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Configuration
base_dir = "path_to/RDD_Kfold"
num_folds = 5
model_types = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
epochs = 50
imgsz = 640
device = 0  
batch_size = 1

# Create timestamp for unique run identifier
timestamp = datetime.now().strftime("%Y%m%d__%H_%M_%S")

# Function to clear GPU memory thoroughly
def clear_gpu_memory():
    """Clean up GPU memory more aggressively"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        
        # Report memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory cleared. Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        
        return allocated < 0.1
    return True

# Function to handle memory optimizations while maintaining consistent batch size
def optimize_memory_usage():
    """Apply memory optimization techniques without changing batch size"""
    # Set environment variables for PyTorch memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    # Force garbage collection
    gc.collect()
    
    # Additional memory optimization techniques
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    
    # Set to deterministic mode (slightly slower but more memory efficient)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Disable gradient synchronization for unused parameters
    torch.autograd.set_detect_anomaly(False) 
    
    print("Applied memory optimization techniques without changing batch size")

# Function to train model on a specific fold
def train_fold(model_type, fold_num, project_dir):
    print(f"\n{'='*50}")
    print(f"Training {model_type} on Fold {fold_num}/{num_folds} with batch size {batch_size}")
    print(f"{'='*50}")
    
    # Set paths for this fold
    fold_dir = os.path.join(base_dir, f"fold{fold_num}")
    data_yaml = os.path.join(fold_dir, "data.yaml")
    
    # First clear GPU memory
    clear_gpu_memory()
    
    # Apply memory optimizations before model initialization
    optimize_memory_usage()
    
    # Initialize model - will download if not present locally
    model = YOLO(model_type)
    
    # Set training arguments
    args = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "device": device,
        "name": f"{model_type.split('.')[0]}_fold{fold_num}",
        "project": project_dir,
        "exist_ok": True,
        "batch": batch_size,
        "workers": 2,  # Reduce worker threads
        "cache": True,  # Enable caching to improve memory efficiency
        "amp": True     # Use mixed precision training to save memory
    }
    
    # Use the same settings for all models to ensure fair comparison
    # No special handling for different model types
        
    # Start timer
    start_time = time.time()
    
    # Train model
    try:
        results = model.train(**args)
        training_success = True
        e_msg = None
    except Exception as e:
        print(f"Error training {model_type} on fold {fold_num}: {e}")
        results = None
        training_success = False
        e_msg = str(e)
        
        # Extra cleanup after exception
        clear_gpu_memory()
    
    # End timer
    training_time = time.time() - start_time
    
    # Force cleanup
    del model
    clear_gpu_memory()
    
    # Return validation metrics and training time
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

# Function to handle distributed training across multiple GPUs
def setup_distributed_training(model_type):
    """Set up distributed training if multiple GPUs are available"""
    if torch.cuda.device_count() > 1:
        print(f"Setting up distributed training across {torch.cuda.device_count()} GPUs")
        return True
    return False

# Function to safely extract metrics with fallbacks
def safe_extract_metrics(model_metrics):
    """Safely extract metrics with error handling and fallbacks"""
    try:
        # Print available attributes for debugging
        sample_metric = model_metrics[0]
        print("Available metrics attributes:", dir(sample_metric))
        
        # Initialize lists for metrics
        precision = []
        recall = []
        map50 = []
        map50_95 = []
        
        # Extract metrics from each fold with proper error handling
        for m in model_metrics:
            if hasattr(m, 'box'):
                # Extract map (precision)
                if hasattr(m.box, 'map'):
                    precision.append(float(m.box.map))
                elif hasattr(m.box, 'precision'):
                    precision.append(float(m.box.precision))
                else:
                    precision.append(0.0)
                    
                # Extract recall
                if hasattr(m.box, 'r'):
                    recall.append(float(m.box.r))
                elif hasattr(m.box, 'recall'):
                    recall.append(float(m.box.recall))
                else:
                    recall.append(0.0)
                    
                # Extract mAP50
                if hasattr(m.box, 'map50'):
                    map50.append(float(m.box.map50))
                else:
                    map50.append(0.0)
                    
                # Extract mAP50-95
                if hasattr(m.box, 'map50_95'):
                    map50_95.append(float(m.box.map50_95))
                else:
                    map50.append(0.0)
            else:
                # Fall back to direct attribute access if box isn't present
                print("Warning: No 'box' attribute found, trying direct access")
                precision.append(float(getattr(m, 'map', 0.0)))
                recall.append(float(getattr(m, 'recall', 0.0)))
                map50.append(float(getattr(m, 'map50', 0.0)))
                map50_95.append(float(getattr(m, 'map50_95', 0.0)))
        
        # Calculate metrics summary
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

# Main execution
if __name__ == "__main__":
    # Create main results directory
    main_results_dir = os.path.join(base_dir, f"results_{timestamp}")
    os.makedirs(main_results_dir, exist_ok=True)
    
    # Dictionary to store results for all models
    all_models_results = {}
    
    # Train each model on all folds
    for model_idx, model_type in enumerate(model_types):
        print(f"\n{'#'*60}")
        print(f"# Starting training for {model_type} ({model_idx+1}/{len(model_types)})")
        print(f"{'#'*60}")
        
        # Clear memory before starting a new model
        clear_gpu_memory()
        
        # Create directory for this model's results
        model_name = model_type.split('.')[0]
        model_results_dir = os.path.join(main_results_dir, model_name)
        os.makedirs(model_results_dir, exist_ok=True)
        
        # Store results for this model
        model_metrics = []
        model_results = []
        
        # Check if we should use distributed training
        use_distributed = setup_distributed_training(model_type)
        
        # Train on each fold
        for fold in range(1, num_folds + 1):
            # Ensure we start with clean memory
            memory_cleared = clear_gpu_memory()
            if not memory_cleared:
                print(f"Warning: Could not clear memory sufficiently before fold {fold}. Attempting to continue anyway.")
            
            # Train with professional batch size
            result = train_fold(model_type, fold, model_results_dir)
            
            # If training fails, document the issue properly
            if not result["success"]:
                error_msg = result.get("error", "Unknown error")
                print(f"Failed to train {model_type} on fold {fold}.")
                print(f"Logged error: {error_msg}")
                
                # Create a record of the failure for documentation
                failure_record = {
                    "model": model_type,
                    "fold": fold,
                    "batch_size": batch_size,
                    "error": error_msg,
                    "recommendation": "Consider using a more powerful GPU or distributed training setup for larger models."
                }
                
                # Save failure record
                with open(os.path.join(model_results_dir, f"fold{fold}_failure.json"), "w") as f:
                    json.dump(failure_record, f, indent=4)
                
                # Force thorough cleanup
                for _ in range(3):
                    clear_gpu_memory()
                    time.sleep(1)
            
            # Store results
            model_results.append(result)
            if result["success"] and result["metrics"] is not None:
                model_metrics.append(result["metrics"])
                
            # Force cleanup between folds
            for _ in range(2):
                clear_gpu_memory()
                time.sleep(1)
        

    print(f"\nAll training complete! Results saved to {main_results_dir}")