from datasets import load_dataset
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np
import torch
from PIL import Image
import os
import pandas as pd
import datasets

np.random.seed(42)
torch.manual_seed(42)

DATASET_BASE = Path("synthetic")
DATASET_IMG_BASE = DATASET_BASE / "image"
DATASET_CODE_BASE = DATASET_BASE / "code"
MAX_LEN = 512


def split_paths_train_val_test(
    sample_paths, train_size=0.8, val_size=0.1, test_size=0.1
):
    assert train_size + val_size + test_size == 1
    train_size = int(train_size * len(sample_paths))
    val_size = int(val_size * len(sample_paths))
    test_size = int(test_size * len(sample_paths))
    train_paths, val_paths, test_paths = torch.utils.data.random_split(
        sample_paths, [train_size, val_size, test_size]
    )
    return list(train_paths), (val_paths), (test_paths)


def get_image_path(id):
    return str(DATASET_IMG_BASE / f"{id}.png")


def preprocess_function(examples):
    # Resize images to 224x224
    img_paths = examples["img_path"]
    images = []
    for img_path in img_paths:
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = img.convert("RGB")
        images.append(img)

    examples["image"] = images
    return examples


def tokenization_fn(tokenizer, captions, max_target_length):
    """Run tokenization on captions."""
    labels = tokenizer(
        captions, padding="max_length", truncation=True, max_length=max_target_length
    ).input_ids

    return labels


def feature_extraction_fn(feature_extractor, image_paths, check_image=True):

    if check_image:
        images = []
        to_keep = []
        for image_file in image_paths:
            try:
                img = Image.open(image_file)
                images.append(img)
                to_keep.append(True)
            except Exception:
                to_keep.append(False)
    else:
        images = [Image.open(image_file) for image_file in image_paths]
    images = [img.resize((224, 224)) for img in images]
    images = [img.convert("RGB") for img in images]
    encoder_inputs = feature_extractor(images=images, return_tensors="np")

    return encoder_inputs.pixel_values


def preprocess_fn(examples, max_target_length, check_image=True):
    """Run tokenization + image feature extraction"""
    image_paths = examples["img_path"]
    captions = examples["code"]

    model_inputs = {}
    # This contains image path column
    model_inputs["labels"] = tokenization_fn(captions, max_target_length)
    model_inputs["pixel_values"] = feature_extraction_fn(
        image_paths, check_image=check_image
    )

    return model_inputs


def get_dataset(model, paths, tokenizer=None, feature_extractor=None):
    dataset = load_dataset("json", data_files=paths)
    dataset = dataset.map(lambda x: {"img_path": get_image_path(x["id"])})

    if tokenizer is None and feature_extractor is None:
        return dataset["train"]
    else:
        raise NotImplementedError


def get_train_val_test_paths(
    train_path="train.csv", val_path="val.csv", test_path="test.csv"
):
    if (
        not os.path.exists(train_path)
        or not os.path.exists(val_path)
        or not os.path.exists(test_path)
    ):
        sample_paths = list(DATASET_CODE_BASE.glob("*.json"))
        train_paths, val_paths, test_paths = split_paths_train_val_test(sample_paths)
        train_paths = [str(path) for path in train_paths]
        val_paths = [str(path) for path in val_paths]
        test_paths = [str(path) for path in test_paths]
        pd.DataFrame(train_paths).to_csv(train_path, header=False, index=False)
        pd.DataFrame(val_paths).to_csv(val_path, header=False, index=False)
        pd.DataFrame(test_paths).to_csv(test_path, header=False, index=False)

    train_paths = pd.read_csv(train_path, header=None)[0].tolist()
    val_paths = pd.read_csv(val_path, header=None)[0].tolist()
    test_paths = pd.read_csv(test_path, header=None)[0].tolist()
    return train_paths, val_paths, test_paths


def get_train_val_test_datasets(
    tokenizer,
    feature_extractor,
    model,
    train_path="train.csv",
    val_path="val.csv",
    test_path="test.csv",
    debug_amount=None,
):
    train_paths, val_paths, test_paths = get_train_val_test_paths(
        train_path, val_path, test_path
    )
    if debug_amount is not None:
        train_paths = train_paths[:debug_amount]
        val_paths = val_paths[:debug_amount]
        test_paths = test_paths[:debug_amount]
    train_dataset = get_dataset(tokenizer, feature_extractor, model, train_paths)
    val_dataset = get_dataset(tokenizer, feature_extractor, model, val_paths)
    test_dataset = get_dataset(tokenizer, feature_extractor, model, test_paths)
    return train_dataset, val_dataset, test_dataset


def get_train_val_test_datasets_processed(
    tokenizer,
    feature_extractor,
    model,
    max_length,
    train_path="train.csv",
    val_path="val.csv",
    test_path="test.csv",
    debug_amount=None,
):
    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
        tokenizer,
        feature_extractor,
        model,
        train_path,
        val_path,
        test_path,
        debug_amount,
    )
    dataset = datasets.DatasetDict()
    dataset["train"] = train_dataset
    dataset["validation"] = val_dataset
    dataset["test"] = test_dataset
    processed_dataset = dataset.map(
        function=preprocess_fn,
        batched=True,
        fn_kwargs={"max_target_length": max_length},
        remove_columns=dataset["train"].column_names,
    )
    return processed_dataset
