# helper functions (logging, metrics, etc.)
import yaml
import itertools
import os
import sys
import time
from datetime import datetime
import functools
import torch

def load_config(path="./config.yaml"):
    data = {}
    with open(path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data


def permute_hyperparams(hyperparams):
    # generating all combinations of hyperparameters from config
    lr_values = hyperparams.get("learning_rate", [])
    batch_values = hyperparams.get("batch_size", [])
    epoch_values = hyperparams.get("num_epochs", [])
    optimizer_values = hyperparams.get("optimizer", [])
    decay_values = hyperparams.get("weight_decay", [])
    
    permutations = []
    for lr, batch, epochs, optimizer, decay in itertools.product(
        lr_values, batch_values, epoch_values, optimizer_values, decay_values
    ):
        permutations.append({
            "learning_rate": lr,
            "batch_size": batch,
            "num_epochs": epochs,
            "optimizer": optimizer,
            "weight_decay": decay,
        })
    
    return permutations

# ============================
# LOGGING SECTION
# ============================


class LogConfig:
    error_path = "err.txt"
    time_path = "time_log.txt"
    gpu_path = "gpu_log.txt"
    log_path = "log.txt"


def send_log(path, message):
    if not path:
        return
    # ensure directory exists
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    timestamp = datetime.now().isoformat()
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | {message}\n")


def gpu_logger(path=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_mem = 0
            if torch.cuda.is_available():
                start_mem = torch.cuda.memory_allocated()
            result = func(*args, **kwargs)
            end_mem = 0
            if torch.cuda.is_available():
                end_mem = torch.cuda.memory_allocated()
            usage = end_mem - start_mem
            timestamp = datetime.now().isoformat()
            msg = f"{timestamp} | [GPU] {func.__name__}: {usage}"
            send_log(path, msg)
            return result
        return wrapper
    return decorator


def time_logger(path=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            t1 = time.time()
            elapsed = t1 - t0
            timestamp = datetime.now().isoformat()
            msg = f"{timestamp} | [TIME] {func.__name__}: {elapsed:.6f}s"
            send_log(path, msg)
            return result
        return wrapper
    return decorator


def error_logger(path=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                timestamp = datetime.now().isoformat()
                msg = f"{timestamp} | [ERROR] {func.__name__}: {type(e).__name__}: {e}"
                send_log(path, msg)
                # terminate runtime as requested
                sys.exit(1)
        return wrapper
    return decorator


def basic_logger(log_config=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            time_path = log_config.time_path
            gpu_path = log_config.gpu_path
            error_path = log_config.error_path
            path = log_config.log_path

            try:
                t0 = time.time()
                start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                result = func(*args, **kwargs)
                t1 = time.time()
                end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

                timestamp = datetime.now().isoformat()
                # time entry
                time_msg = f"{timestamp} | [TIME] {func.__name__}: {t1 - t0:.6f}s"
                # gpu entry
                gpu_msg = f"{timestamp} | [GPU] {func.__name__}: {end_mem - start_mem}"

                # write
                if time_path:
                    send_log(time_path, time_msg)
                else:
                    send_log(path, time_msg)

                if gpu_path:
                    send_log(gpu_path, gpu_msg)
                else:
                    send_log(path, gpu_msg)

                return result
            except Exception as e:
                timestamp = datetime.now().isoformat()
                err_msg = f"{timestamp} | [ERROR] {func.__name__}: {type(e).__name__}: {e}"
                if error_path:
                    send_log(error_path, err_msg)
                else:
                    send_log(path, err_msg)
                sys.exit(1)
        return wrapper
    return decorator

global_log = LogConfig()