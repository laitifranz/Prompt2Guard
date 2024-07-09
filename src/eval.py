import argparse
import json
import os
from tqdm import tqdm
import io
import pickle
import copy
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from einops import reduce, rearrange

from models.slinet import SliNet


class DummyDataset(Dataset):
    def __init__(self, data_path, data_type, data_scenario, data_compression):
        self.do_compress = [
            data_compression[0],
            data_compression[1],
        ]  # enable/disable compression from flag - jpeg quality
        self.trsf = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        images = []
        labels = []

        # print(f'--- Data compression: {data_compression} ---')

        if data_type == "cddb":
            if data_scenario == "cddb_hard":
                subsets = [
                    "gaugan",
                    "biggan",
                    "wild",
                    "whichfaceisreal",
                    "san",
                ]  # <- CDDB Hard
                multiclass = [0, 0, 0, 0, 0]
            elif data_scenario == "ood":
                subsets = ["deepfake", "glow", "stargan_gf"]  # <- OOD experiments
                multiclass = [0, 1, 1]
            else:
                raise RuntimeError(
                    f"Unexpected data_scenario value: {data_scenario}. Expected 'cddb_hard' or 'ood'."
                )
            print(f"--- Test on {subsets} with {data_scenario} scenario ---")
            for id, name in enumerate(subsets):
                root_ = os.path.join(data_path, name, "val")
                # sub_classes = ['']
                sub_classes = os.listdir(root_) if multiclass[id] else [""]
                for cls in sub_classes:
                    for imgname in os.listdir(os.path.join(root_, cls, "0_real")):
                        images.append(os.path.join(root_, cls, "0_real", imgname))
                        labels.append(0 + 2 * id)

                    for imgname in os.listdir(os.path.join(root_, cls, "1_fake")):
                        images.append(os.path.join(root_, cls, "1_fake", imgname))
                        labels.append(1 + 2 * id)
        else:
            pass

        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.dataset_path = data_path

        with open("./src/utils/classes.pkl", "rb") as f:
            self.object_labels = pickle.load(f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.images[idx])
        image = self.trsf(
            self.pil_loader(img_path, self.do_compress[0], self.do_compress[1])
        )
        label = self.labels[idx]
        object_label = self.object_labels[img_path.replace(self.dataset_path, "")][0:5]
        return object_label, image, label

    def pil_loader(self, path, do_compress, quality):
        with open(path, "rb") as f:
            if do_compress:
                f = self.compress_image_to_memory(path, quality=quality)
            img = Image.open(f)
            return img.convert("RGB")

    def compress_image_to_memory(self, path, quality):
        with Image.open(path) as img:
            output = io.BytesIO()
            img.save(output, "JPEG", quality=quality)
            output.seek(0)
            return output


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce of multiple continual learning algorithms."
    )
    parser.add_argument(
        "--scenario", type=str, default="cddb_hard", help="scenario to test"
    )
    parser.add_argument("--resume", type=str, default="", help="resume model")
    parser.add_argument(
        "--random_select", action="store_true", help="use random select"
    )
    parser.add_argument(
        "--upperbound", action="store_true", help="use groundtruth task identification"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cddb_inference.json",
        help="Json file of settings.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/francesco.laiti/datasets/CDDB/",
        help="data path",
    )
    parser.add_argument("--datatype", type=str, default="deepfake", help="data type")
    parser.add_argument(
        "--compression", type=bool, default=False, help="test on compressed data"
    )
    parser.add_argument(
        "--c_quality",
        type=int,
        default=100,
        help="quality of JPEG compressed (100, 90, 50...)",
    )
    return parser


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def load_configuration():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args_dict = vars(args)
    args_dict.update(param)
    return args_dict


def compute_predictions(outputs):
    predictions = {}

    # Top1
    outputs_top1 = rearrange(outputs, "b t p -> b (t p)")
    _, predicts_top1 = outputs_top1.max(dim=1)
    predictions["top1"] = predicts_top1 % 2

    # Mean
    outputs_mean = reduce(outputs, "b t p -> b p", "mean")
    predictions["mean"] = torch.argmax(outputs_mean, dim=-1)

    # Mixture of experts (top & mean)
    r_f_tensor = rearrange(outputs, "b t p -> b p t")
    r_f_max, _ = torch.max(r_f_tensor, dim=-1)
    r_f_mean = reduce(r_f_tensor, "b p t -> b p", "mean")
    diff_max = torch.abs(r_f_max[:, 0] - r_f_max[:, 1])
    diff_mean = torch.abs(r_f_mean[:, 0] - r_f_mean[:, 1])
    conditions = diff_mean > diff_max
    predicts_based_on_mean = torch.where(
        r_f_mean[:, 0] > r_f_mean[:, 1],
        torch.zeros_like(conditions),
        torch.ones_like(conditions),
    )
    predicts_based_on_max = torch.where(
        r_f_max[:, 0] > r_f_max[:, 1],
        torch.zeros_like(conditions),
        torch.ones_like(conditions),
    )
    predictions["mix_top_mean"] = torch.where(
        conditions, predicts_based_on_mean, predicts_based_on_max
    )

    return predictions


def accuracy_binary(y_pred, y_true, increment=2):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = float(
        "{:.2f}".format((y_pred % 2 == y_true % 2).sum() * 100 / len(y_true))
    )  # * Task-agnostic AA *

    task_acc = []
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        acc = ((y_pred[idxes] % 2) == (y_true[idxes] % 2)).sum() * 100 / len(idxes)
        all_acc[label] = float("{:.2f}".format(acc))
        task_acc.append(acc)
    all_acc["task_wise"] = float(
        "{:.2f}".format(sum(task_acc) / len(task_acc))
    )  # * Average Accuracy (AA) or Task-wise AA *
    return all_acc


def prepare_model(args):
    checkpoint = torch.load(args["resume"], map_location=args["device"])

    # update config args
    args["K"] = checkpoint["K"]
    args["topk_classes"] = checkpoint["topk_classes"]
    args["ensembling"] = checkpoint["ensembling_flags"]

    # load all prototypes
    keys_dict = {
        "all_keys": checkpoint["keys"]["all_keys"],  # * [Task, N_cluster = 5, 512]
        "all_keys_one_cluster": checkpoint["keys"]["all_keys_one_cluster"],  # * [Task, 512]
        "real_keys_one_cluster": checkpoint["keys"]["real_keys_one_cluster"],  # * [Task, 512]
        "fake_keys_one_cluster": checkpoint["keys"]["fake_keys_one_cluster"],  # * [Task, 512]
    }

    args["num_tasks"] = checkpoint["tasks"] + 1
    args["task_name"] = range(args["num_tasks"])

    # build and load model
    model = SliNet(args)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    model = model.to(args["device"])

    print(f"--- Run: {checkpoint.get('run_name', 'not available')} ---")

    return model, keys_dict


def prepare_data_loader(args):
    test_dataset = DummyDataset(
        args["data_path"],
        args["dataset"],
        args["scenario"],
        [args["compression"], args["c_quality"]],
    )
    return DataLoader(
        test_dataset,
        batch_size=args["batch_size_eval"],
        shuffle=False,
        num_workers=int(os.environ.get("SLURM_CPUS_ON_NODE", 2)),
    )


@torch.no_grad
def inference_step(args, model: SliNet, test_loader, keys_dict):
    def upperbound_selection(targets):
        domain_indices = torch.div(targets, 2, rounding_mode="floor")
        domain_prob = torch.zeros(
            (len(targets), total_tasks), dtype=torch.float16, device=args["device"]
        )
        domain_prob[torch.arange(len(targets)), domain_indices] = 1.0
        return domain_prob

    def process_batch(inputs, targets, object_name):
        keys_dict["upperbound"] = upperbound_selection(targets)
        if args["upperbound"]:
            keys_dict["prototype"] = "upperbound"

        outputs = model.interface(inputs, object_name, total_tasks, keys_dict)

        if args["softmax"]:
            outputs = torch.nn.functional.softmax(outputs, dim=-1)
        return compute_predictions(outputs)

    total_tasks = args["num_tasks"]
    y_pred = {key: [] for key in ["top1", "mean", "mix_top_mean"]}
    y_true = []

    for _, (object_name, inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets = inputs.to(args["device"]), targets.to(args["device"])
        predictions = process_batch(inputs, targets, object_name)

        for key, pred in predictions.items():
            y_pred[key].append(pred.cpu().numpy())
        y_true.append(targets.cpu().numpy())

    y_true = np.concatenate(y_true)

    accuracies = {
        key: accuracy_binary(np.concatenate(pred), y_true)
        for key, pred in y_pred.items()
    }

    return accuracies


def pretty_print(data):
    return json.dumps(data, indent=4, sort_keys=True)


if __name__ == "__main__":
    args = load_configuration()
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    scenarios = copy.deepcopy(args["scenario"])
    model, keys_dict = prepare_model(args)
    keys_dict["prototype"] = args["prototype"]

    for s in scenarios:
        args["scenario"] = s
        test_loader = prepare_data_loader(args)

        print(pretty_print(inference_step(args, model, test_loader, keys_dict)))
        if args["upperbound"] and s != "ood":
            print(pretty_print(inference_step(args, model, test_loader, keys_dict)))
        if args["random_select"]:
            print(pretty_print(inference_step(args, model, test_loader, keys_dict)))
