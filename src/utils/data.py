import os
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class CDDB_benchmark(object):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = args["class_order"]
        self.class_order = class_order

    def download_data(self):

        train_dataset = []
        test_dataset = []
        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, "train")
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else [""]
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, "0_real")):
                    train_dataset.append(
                        (os.path.join(root_, cls, "0_real", imgname), 0 + 2 * id)
                    )
                for imgname in os.listdir(os.path.join(root_, cls, "1_fake")):
                    train_dataset.append(
                        (os.path.join(root_, cls, "1_fake", imgname), 1 + 2 * id)
                    )

        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, "val")
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else [""]
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, "0_real")):
                    test_dataset.append(
                        (os.path.join(root_, cls, "0_real", imgname), 0 + 2 * id)
                    )
                for imgname in os.listdir(os.path.join(root_, cls, "1_fake")):
                    test_dataset.append(
                        (os.path.join(root_, cls, "1_fake", imgname), 1 + 2 * id)
                    )

        self.train_data, self.train_targets = split_images_labels(train_dataset)
        self.test_data, self.test_targets = split_images_labels(test_dataset)
