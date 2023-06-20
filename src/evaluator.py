# %%
#
from dataset_creator import generate_screenshot
from pathlib import Path
from dataloader import get_train_base_dataset
import os
from tqdm import tqdm
import asyncio
import cv2
import numpy as np
import matplotlib.pyplot as plt
import nltk
import json
from HTMLBLEU.htmlBLEU import htmlbleu
import numpy as np
import json
from itertools import combinations


# %%


def rank_data(data):
    return {value: rank + 1 for rank, value in enumerate(sorted(set(data)))}


def calculate_spearman(x1, x2):
    if len(x1) != len(x2):
        raise ValueError("Input lists must have the same length")

    n = len(x1)

    x1_rank = rank_data(x1)
    x2_rank = rank_data(x2)

    x1_ranks = [x1_rank[val] for val in x1]
    x2_ranks = [x2_rank[val] for val in x2]

    diff_sq_sum = sum((x1_ranks[i] - x2_ranks[i]) ** 2 for i in range(n))

    spearman_corr = 1 - (6 * diff_sq_sum) / (n * (n**2 - 1))
    return spearman_corr


def generate_images_if_needed(
    test_dataset, test_outputs_images_path, test_outputs_codes_path
):
    for sample in tqdm(test_dataset, total=len(test_dataset)):
        predicted_code_path = test_outputs_codes_path / (sample["id"] + ".html")
        # print(os.path.exists(predicted_code_path))
        if not os.path.exists(predicted_code_path):
            continue
        predicted_code = open(predicted_code_path, "r").read()
        predicted_img_path = test_outputs_images_path / (sample["id"] + ".png")
        # print(os.path.exists(predicted_img_path))
        if not os.path.exists(predicted_img_path):
            asyncio.get_event_loop().run_until_complete(
                generate_screenshot(predicted_code, predicted_img_path)
            )
            # print("generated image for {} at path {}".format(sample['id'], predicted_img_path))


def calculate_IOU(ground_img, predicted_img):
    ground_img = np.all(ground_img != [255, 255, 255], axis=-1)
    predicted_img = np.all(predicted_img != [255, 255, 255], axis=-1)
    intersection = np.logical_and(ground_img, predicted_img)
    union = np.logical_or(ground_img, predicted_img)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def calculate_MSE_colorblind(ground_img, predicted_img):
    ground_img = np.all(ground_img != [255, 255, 255], axis=-1).astype(int)
    predicted_img = np.all(predicted_img != [255, 255, 255], axis=-1).astype(int)
    mse = np.mean((ground_img - predicted_img) ** 2)
    return mse


def calculate_MSE(ground_img, predicted_img):
    mse = np.mean((ground_img - predicted_img) ** 2)
    return mse


def calculate_divs_matching(ground_code, predicted_code):

    # check number of divs in both codes
    n_divs_ground = ground_code.count("<div")
    n_divs_predicted = predicted_code.count("<div")
    if n_divs_ground != n_divs_predicted:
        return 0
    return 1


def calculate_BLEU(ground_code: str, predicted_code: str):

    ground_code = ground_code.split()
    predicted_code = predicted_code.split()
    bleu_score = nltk.translate.bleu_score.sentence_bleu([ground_code], predicted_code)
    return bleu_score


def calculate_HTMLBLEU(ground_code: str, predicted_code: str):
    return htmlbleu(ground_code, predicted_code)


def extract_element_types(code):
    """
    Extract number of elements given each code contains title in the format of
    <title>2 Circles, 3 Blocks</title>
    """
    title_start = code.find("<title>")
    title_end = code.find("</title>")
    title = code[title_start + 7 : title_end]

    n_circles = 0
    n_blocks = 0

    for element in title.split(","):
        if "Circle" in element:
            n_circles = int(element.split()[0])
        if "Block" in element:
            n_blocks = int(element.split()[0])

    return {"n_circles": n_circles, "n_blocks": n_blocks}


def number_of_divs(ground_code):
    n_divs_ground = ground_code.count("<div")
    ground_elements = extract_element_types(ground_code)
    n_divs_title = sum(ground_elements.values())
    assert n_divs_ground == n_divs_title
    return n_divs_ground


def calculate_element_matching(ground_code, predicted_code):
    ground_elements = extract_element_types(ground_code)
    predicted_elements = extract_element_types(predicted_code)
    if (
        ground_elements["n_circles"] == predicted_elements["n_circles"]
        and ground_elements["n_blocks"] == predicted_elements["n_blocks"]
    ):
        return 1
    return 0


def visualize_images(ground_img, predicted_img):

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(ground_img)
    ax[1].imshow(predicted_img)
    plt.show()
