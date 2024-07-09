import os
import os.path
import sys
import logging
import time
import torch
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from methods.prompt2guard import Prompt2Guard
import numpy as np


def train(args):
    logfilename = "logs/{}/{}".format(
        args["run_name"].replace("_", "/"),
        time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
    )
    os.makedirs(logfilename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + "/info.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args)
    _set_device(args)
    print_args(args)
    
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    args["class_order"] = data_manager._class_order
    args["filename"] = os.path.join(logfilename, "task")
    model = Prompt2Guard(args)

    acc_matrix = {
        "top1": np.zeros((data_manager.nb_tasks, data_manager.nb_tasks)),
        "mean": np.zeros((data_manager.nb_tasks, data_manager.nb_tasks)),
        "mix_top_mean": np.zeros((data_manager.nb_tasks, data_manager.nb_tasks)),
    }
    label_history = []

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model.network)))
        logging.info("Trainable params: {}".format(count_parameters(model.network, True)))

        model.incremental_train(data_manager)
        record_task_accuracy(task, model.eval_task(), acc_matrix, label_history)
        model.after_task(data_manager.nb_tasks)
        model.save_checkpoint()

    compute_forgetting(model, acc_matrix)
    model.wandb_logger.finish()


def _compute_AF(matrix):
    total_bwt = 0
    N = matrix.shape[0]
    for i in range(N - 1):  # Iterate through each task except the last one
        bwt_i = 0
        for j in range(i + 1, N):  # Iterate from task i+1 to N to calculate BWT_i
            bwt_i += matrix[j, i] - matrix[i, i]
        bwt_i /= N - i - 1  # Normalize by the number of tasks considered for this BWT_i
        total_bwt += bwt_i
    af = total_bwt / (N - 1)  # Calculate the average of all BWT_i
    return af


def compute_forgetting(model: Prompt2Guard, acc_matrix):
    for k in acc_matrix.keys():
        forgetting = _compute_AF(acc_matrix[k])
        logging.info("Avg Forgetting of {}: {:.4f}".format(k, forgetting))
        model.wandb_logger.log({**{f"AF/{k}": forgetting}})


def record_task_accuracy(
    current_task, current_task_acc: dict, matrix_dict: dict, label_history: list
):
    label_history.append(
        "{}-{}".format(
            str(current_task * 2).zfill(2), str(current_task * 2 + 1).zfill(2)
        )
    )
    for logit_ops in current_task_acc.keys():
        dict_subset = {
            k: current_task_acc[logit_ops][k]
            for k in label_history
            if k in current_task_acc[logit_ops]
        }
        for idx_label, label_task in enumerate(dict_subset):
            matrix_dict[logit_ops][current_task][idx_label] = current_task_acc[
                logit_ops
            ][label_task]

    for key, value in current_task_acc.items():
        logging.info(f"Performance Task {current_task} for {key}: {value}")


def _set_device(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info("Device: " + device.type)
    args["device"] = device


def _set_random(args):
    torch.manual_seed(args["torch_seed"])
    torch.cuda.manual_seed(args["torch_seed"])
    torch.cuda.manual_seed_all(args["torch_seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
