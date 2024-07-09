import sys
import os
import torch
from PIL import Image
from class_names import imagenet1k_classnames, facedataset_classnames
import json
from tqdm import tqdm
import pickle

sys.path.append("../")
from models.clip import clip


def zeroshot_CLIP_batch(model, preprocess, device, text_inputs, class_names, image_paths, topk_indexes=5):
    batch = torch.stack([preprocess(Image.open(path)) for path in image_paths]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(batch)
        text_features = model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    results = []
    for i in range(similarity.size(0)):
        values, indices = similarity[i].topk(topk_indexes)
        outputs = [
            [class_names[index.item()], round(100 * value.item(), 4)]
            for value, index in zip(values, indices)
        ]
        results.append(outputs)
    return results


def process_images_in_batches(
    model,
    preprocess,
    device,
    text_inputs,
    dataset_dir,
    class_names,
    image_paths,
    batch_size,
    topk_indexes,
    class_label=None,
):
    results = {}
    for i in tqdm(
        range(0, len(image_paths), batch_size),
        desc=f"Processing batch of size {batch_size}",
    ):
        batch_paths = image_paths[i : i + batch_size]
        batch_results = zeroshot_CLIP_batch(
            model,
            preprocess,
            device,
            text_inputs,
            class_names,
            batch_paths,
            topk_indexes,
        )
        for path, result in zip(batch_paths, batch_results):
            if class_label:
                result.append([class_label, -1])
            results[path.replace(dataset_dir, "")] = result
    return results


def prepare_text_inputs(data_type):
    if data_type == "CDDB":
        dataset_structure = [
            "whichfaceisreal",
            "stylegan",
            "crn",
            "imle",
            "cyclegan",
            "wild",
            "glow",
            "deepfake",
            "san",
            "stargan_gf",
            "biggan",
            "gaugan",
        ]
        multiclass = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
        humans_inside = [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0]
        subsets = ["train", "val"]
        classes = ["0_real", "1_fake"]

        return dataset_structure, multiclass, humans_inside, subsets, classes
    else:
        raise ValueError(f"{data_type} not valid.")


def zeroshot_dataset_batch(dataset_dir, data_type, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device)

    dataset_structure, multiclass, humans_inside, subsets, classes = (
        prepare_text_inputs(data_type)
    )

    results = {}
    for index, folder in enumerate(tqdm(dataset_structure, desc="Processing datasets")):
        if humans_inside[index] == 0:
            text_inputs = torch.cat(
                [
                    clip.tokenize(f"a photo of a {c}")
                    for c in imagenet1k_classnames.values()
                ]
            ).to(device)
            class_names = imagenet1k_classnames
        else:
            text_inputs = torch.cat(
                [
                    clip.tokenize(f"a photo of a {c}")
                    for c in facedataset_classnames.values()
                ]
            ).to(device)
            class_names = facedataset_classnames

        for subset in subsets:
            subset_path = os.path.join(dataset_dir, folder, subset)
            if multiclass[index] == 1:
                class_labels = os.listdir(subset_path)
                for class_label in class_labels:
                    class_path = os.path.join(subset_path, class_label)
                    for binary_label in classes:
                        image_paths = [
                            os.path.join(class_path, binary_label, img)
                            for img in os.listdir(
                                os.path.join(class_path, binary_label)
                            )
                        ]
                        batch_results = process_images_in_batches(
                            model,
                            preprocess,
                            device,
                            text_inputs,
                            dataset_dir,
                            class_names,
                            image_paths,
                            batch_size,
                            5,
                            class_label,
                        )
                        results.update(batch_results)
            else:
                for binary_label in classes:
                    image_paths = [
                        os.path.join(subset_path, binary_label, img)
                        for img in os.listdir(os.path.join(subset_path, binary_label))
                    ]
                    batch_results = process_images_in_batches(
                        model,
                        preprocess,
                        device,
                        text_inputs,
                        dataset_dir,
                        class_names,
                        image_paths,
                        batch_size,
                        5,
                    )
                    results.update(batch_results)

    with open("./DEBUG_classes.json", "w") as f:  # only for fast debug
        json.dump(results, f, indent=4)
    with open("./classes.pkl", "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_JSON_dataset_batch():
    dataroot = "/home/francesco.laiti/datasets/CDDB/"
    datatype = "CDDB"
    batch_size = 2048
    zeroshot_dataset_batch(dataroot, datatype, batch_size)


if __name__ == "__main__":
    get_JSON_dataset_batch()
