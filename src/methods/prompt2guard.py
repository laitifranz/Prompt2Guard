import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import os

from utils.toolkit import tensor2numpy, accuracy_domain
from models.slinet import SliNet
from utils.lr_scheduler import build_lr_scheduler
from utils.data_manager import DataManager
from eval import compute_predictions

import wandb


class Prompt2Guard:

    def __init__(self, args: dict):
        # Network and device settings
        self.network = SliNet(args)
        self.device = args["device"]
        self.class_num = self.network.class_num

        # Task and class settings
        self.cur_task = -1
        self.n_clusters = 5
        self.n_cluster_one = 1
        self.known_classes = 0
        self.total_classes = 0

        # Key settings, different clusters tested
        self.all_keys = []              # consider n_clusters image prototypes for each domain
        self.all_keys_one_vector = []   # consider 1 image prototype for each domain
        self.real_keys_one_vector = []  # only real images considered to build the prototype
        self.fake_keys_one_vector = []  # only fake images considered to build the prototype

        # Learning parameters
        self.EPSILON = args["EPSILON"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.warmup_epoch = args["warmup_epoch"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.batch_size_eval = args["batch_size_eval"]
        self.weight_decay = args["weight_decay"]
        self.label_smoothing = args["label_smoothing"]
        self.enable_prev_prompt = args["enable_prev_prompt"]

        # System settings
        self.num_workers = int(
            os.environ.get("SLURM_CPUS_ON_NODE", args["num_workers"])
        )
        self.filename = args["filename"]

        # Other settings
        self.args = args

        # wandb setup
        slurm_job_name = os.environ.get("SLURM_JOB_NAME", 'prompt2guard')
        if slurm_job_name == "bash":
            slurm_job_name += "/localtest"

        self.wandb_logger = wandb.init(
            project=slurm_job_name.split("/")[0],
            entity="YOUR_USERNAME",
            name=slurm_job_name.split("/")[1],
            mode="disabled" if not args["wandb"] else "online",
            config=args,
        )
        if self.wandb_logger is None:
            raise ValueError("Failed to initialize wandb logger")

        self.wandb_logger.define_metric("epoch")
        self.wandb_logger.define_metric("task")
        self.wandb_logger.define_metric("condition")
        self.wandb_logger.define_metric("task_*", step_metric="epoch")
        self.wandb_logger.define_metric("eval_trainer/*", step_metric="task")
        self.wandb_logger.define_metric("inference_*", step_metric="condition")

    def after_task(self, nb_tasks):
        self.known_classes = self.total_classes
        if self.enable_prev_prompt and self.network.numtask < nb_tasks:
            with torch.no_grad():
                self.network.prompt_learner[self.network.numtask].load_state_dict(
                    self.network.prompt_learner[self.network.numtask - 1].state_dict()
                )

    def incremental_train(self, data_manager: DataManager):
        self.cur_task += 1
        self.total_classes = self.known_classes + data_manager.get_task_size(
            self.cur_task
        )
        self.network.update_fc()

        logging.info("Learning on {}-{}".format(self.known_classes, self.total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self.known_classes, self.total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self.total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self._train(self.train_loader, self.test_loader)
        self.clustering(self.train_loader)

    def _train(self, train_loader, test_loader):
        self.network.to(self.device)
        for name, param in self.network.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" + "." + str(self.network.numtask - 1) in name:
                param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logging.info(f"Parameters to be updated: {enabled}")

        if self.cur_task == 0:
            optimizer = optim.SGD(
                self.network.parameters(),
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.init_weight_decay,
            )
            scheduler = build_lr_scheduler(
                optimizer,
                lr_scheduler="cosine",
                warmup_epoch=self.warmup_epoch,
                warmup_type="constant",
                warmup_cons_lr=1e-5,
                max_epoch=self.epochs,
            )
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self.network.parameters(),
                momentum=0.9,
                lr=self.lrate,
                weight_decay=self.weight_decay,
            )
            scheduler = build_lr_scheduler(
                optimizer,
                lr_scheduler="cosine",
                warmup_epoch=self.warmup_epoch,
                warmup_type="constant",
                warmup_cons_lr=1e-5,
                max_epoch=self.epochs,
            )
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0
            for i, (object_name, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self.known_classes

                logits = self.network(inputs, object_name)["logits"]
                loss = F.cross_entropy(
                    logits, targets, label_smoothing=self.label_smoothing
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy_domain(self.network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self.cur_task,
                epoch + 1,
                self.run_epoch,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)
            self.wandb_logger.log(
                {
                    "task_{}/train_loss".format(self.cur_task): losses
                    / len(train_loader),
                    "task_{}/train_acc".format(self.cur_task): train_acc,
                    "task_{}/test_acc".format(self.cur_task): test_acc,
                    "epoch": epoch + 1,
                }
            )

        logging.info(info)

    def clustering(self, dataloader):
        def run_kmeans(n_clusters, fts):
            clustering = KMeans(
                n_clusters=n_clusters, random_state=0, n_init="auto"
            ).fit(fts)
            return torch.tensor(clustering.cluster_centers_).to(self.device)

        all_fts = []
        real_fts = []
        fake_fts = []
        for _, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            index_reals = (targets == self.known_classes).nonzero().view(-1)  # 0 real
            index_fakes = ((targets == self.known_classes + 1).nonzero().view(-1))  # 1 fake
            with torch.no_grad():
                feature = self.network.extract_vector(inputs)  # only img fts
            all_fts.append(feature)
            real_fts.append(torch.index_select(feature, 0, index_reals))
            fake_fts.append(torch.index_select(feature, 0, index_fakes))
        all_fts = torch.cat(all_fts, 0).cpu().detach().numpy()
        real_fts = torch.cat(real_fts, 0).cpu().detach().numpy()
        fake_fts = torch.cat(fake_fts, 0).cpu().detach().numpy()

        self.all_keys.append(run_kmeans(self.n_clusters, all_fts))
        self.all_keys_one_vector.append(run_kmeans(self.n_cluster_one, all_fts))
        self.real_keys_one_vector.append(run_kmeans(self.n_cluster_one, real_fts))
        self.fake_keys_one_vector.append(run_kmeans(self.n_cluster_one, fake_fts))

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (object_labels, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = model(inputs, object_labels)["logits"]

            predicts = torch.max(outputs, dim=1)[1]
            correct += (
                (predicts % self.class_num).cpu() == (targets % self.class_num)
            ).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def save_checkpoint(self):
        self.network.cpu()

        layers_to_save = ["prompt_learner"]
        model_state_dict = {
            name: param
            for name, param in self.network.named_parameters()
            if any(layer in name for layer in layers_to_save)
        }

        keys_dict = {
            "all_keys": torch.stack(self.all_keys).squeeze().to(dtype=torch.float16),
            "all_keys_one_cluster": torch.stack(self.all_keys_one_vector)
            .squeeze()
            .to(dtype=torch.float16),
            "real_keys_one_cluster": torch.stack(self.real_keys_one_vector)
            .squeeze()
            .to(dtype=torch.float16),
            "fake_keys_one_cluster": torch.stack(self.fake_keys_one_vector)
            .squeeze()
            .to(dtype=torch.float16),
        }

        ensembling_flags = [
            self.network.ensemble_token_embedding,
            self.network.ensemble_before_cosine_sim,
            self.network.ensemble_after_cosine_sim,
            self.network.confidence_score_enable,
        ]

        save_dict = {
            "tasks": self.cur_task,
            "model_state_dict": model_state_dict,
            "keys": keys_dict,
            "K": self.network.K,
            "run_name": os.environ["SLURM_JOB_NAME"],
            "topk_classes": self.network.topk_classes,
            "ensembling_flags": ensembling_flags,
        }
        torch.save(save_dict, "{}_{}.tar".format(self.filename, self.cur_task))

    def eval_task(self):
        y_pred, y_true = self._eval(self.test_loader)
        metrics = {}
        for logit_key in y_pred.keys():
            metrics[logit_key] = accuracy_domain(
                y_pred[logit_key], y_true, self.known_classes, class_num=self.class_num
            )
            self.wandb_logger.log(
                {
                    **{
                        f"eval_{logit_key}/{key}": value
                        for key, value in metrics[logit_key].items()
                    },
                    "task": self.cur_task,
                }
            )
        return metrics

    def prepare_tensor(self, tensor, unsqueeze=False):
        tensor = torch.stack(tensor).squeeze().to(dtype=torch.float16)
        if unsqueeze:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _eval(self, loader):
        self.network.eval()
        unsqueeze = self.network.numtask == 1

        dummy_key_dict = {
            "all_keys": self.prepare_tensor(self.all_keys),
            "all_keys_one_cluster": self.prepare_tensor(
                self.all_keys_one_vector, unsqueeze
            ),
            "real_keys_one_cluster": self.prepare_tensor(
                self.real_keys_one_vector, unsqueeze
            ),
            "fake_keys_one_cluster": self.prepare_tensor(
                self.fake_keys_one_vector, unsqueeze
            ),
            "upperbound": self.prepare_tensor(self.fake_keys_one_vector, unsqueeze),
            "prototype": "fake",
        }

        softmax = False
        total_tasks = self.network.numtask
        y_pred, y_true = {}, []
        for _, (object_name, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                outputs = self.network.interface(inputs, object_name, total_tasks, dummy_key_dict)  # * [B, T, P]
            if softmax:
                outputs = torch.nn.functional.softmax(outputs, dim=-1)
            predicts = compute_predictions(outputs)
            for key in predicts.keys():
                if key not in y_pred:
                    y_pred[key] = []
                y_pred[key].append(predicts[key].cpu().numpy())
            y_true.append(targets.cpu().numpy())
        y_true = np.concatenate(y_true)

        for key in y_pred.keys():
            y_pred[key] = np.concatenate(y_pred[key])

        return y_pred, y_true
