# %%
import pickle
from evaluator import extract_element_types
import matplotlib.pyplot as plt
import seaborn as sns

with open("all_metrics.pickle", "rb") as handle:
    all_metrics = pickle.load(handle)

# %%

import pandas as pd

iou_score_scores = {}
iou_score_stds = {}
mse_score_scores = {}
for model in all_metrics:
    if "overfit" in model:
        continue
    print(model)
    df = pd.DataFrame.from_dict(all_metrics[model], orient="index")
    # group number_of_divs_count
    means = df.groupby("number_of_divs_count")[["iou_score", "mse_score"]].mean()
    stds = df.groupby("number_of_divs_count")[["iou_score", "mse_score"]].std()
    # sort means in ascending order
    means.sort_index(inplace=True)
    stds.sort_index(inplace=True)
    iou_score_scores[model] = means["iou_score"].tolist()
    mse_score_scores[model] = means["mse_score"].tolist()
    iou_score_stds[model] = stds["iou_score"].tolist()
# %%
# plot iou_score_scores lineplot
modelname_to_label = {"code_generation_dit": "DiT-GPT2", "code_generation": "ViT-GPT2"}
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 7})
plt.figure(figsize=(10, 7))
for model in iou_score_scores:
    plt.plot(iou_score_scores[model], label=modelname_to_label[model])
plt.legend()
plt.xlabel("Number of <div> elements")
plt.ylabel("IoU score")
plt.savefig("plots/iou_score_scores.pdf")
plt.show()
