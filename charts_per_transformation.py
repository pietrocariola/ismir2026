import numpy as np
import json
import matplotlib.pyplot as plt
import os
import transformations as tf
from pathlib import Path

INPUT_FOLDER = "/home/gpu3/visfma/projs/"
OUTPUT_FOLDER = "/home/gpu3/visfma/charts_per_transformation/"

MODELS = ['audioclip', 'clap', 'wav2clip']
TRANSFS = [
    "pitchshift",
    "timestretch",
    "highpass",
    "lowpass",
    "clipper",
    "noiseadder",
    "bitcrush",
    "gain",
    ]

DATASET = 'fmx'

GENRES = ["electronic", "experimental", "folk", "hip-hop", "instrumental", "pop", "rock"]

genres = ""
for genre in GENRES:
    genres += f"{genre[:2]}{genre[-1]}"    

for transf in TRANSFS:

    transf_full_name = transf
    transf_id = tf.tf_dict_ids[transf_full_name]
    transf = f"{transf[:2]}{transf[-1]}"

    ### AUDIO CLIP ###
    path = os.path.join(INPUT_FOLDER, f"x2d_{DATASET}_{genres}_aup_{transf}")
    x2d_aup = np.load(path+".npy")
    path = os.path.join(INPUT_FOLDER, f"labels_{DATASET}_{genres}_aup_{transf}")
    with open(path+".json", "r") as f:
        labels_aup = json.load(f)
    labels_aup = [label if label!='audioclip' else transf_id for label in labels_aup]

    ### CLAP ###
    path = os.path.join(INPUT_FOLDER, f"x2d_{DATASET}_{genres}_clp_{transf}")
    x2d_clp = np.load(path+".npy")
    path = os.path.join(INPUT_FOLDER, f"labels_{DATASET}_{genres}_clp_{transf}")
    with open(path+".json", "r") as f:
        labels_clp = json.load(f)
    labels_clp = [label if label!='clap' else transf_id for label in labels_clp]    

    ### WAV2CLIP ###
    path = os.path.join(INPUT_FOLDER, f"x2d_{DATASET}_{genres}_wap_{transf}")
    x2d_wap = np.load(path+".npy")
    path = os.path.join(INPUT_FOLDER, f"labels_{DATASET}_{genres}_wap_{transf}")
    with open(path+".json", "r") as f:
        labels_wap = json.load(f)
    labels_wap = [label if label!='wav2clip' else transf_id for label in labels_wap]

    fig, axes = plt.subplots(1, 3, figsize=(8, 3), constrained_layout=True)

    fig.suptitle(f"{transf_full_name[0].upper()}{transf_full_name[1:]}", fontsize=16)

    titles = ["AudioCLIP", "CLAP", "WAV2CLIP"]
    
    # for i, label in enumerate(labels_clp):
    #     if label == 'clap':
    #         plt.scatter(x2d_clp[i,0], x2d_clp[i,1], c='white', edgecolors='black')
    #         labels_clp[i] = transf_id


    sc = axes[0].scatter(x2d_aup[:, 0], x2d_aup[:, 1], c=labels_aup, s=7, alpha=0.15)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title(titles[0])

    sc = axes[1].scatter(x2d_clp[:, 0], x2d_clp[:, 1], c=labels_clp, s=7, alpha=0.15)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title(titles[1])    

    sc = axes[2].scatter(x2d_wap[:, 0], x2d_wap[:, 1], c=labels_wap, s=7, alpha=0.15)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title(titles[2])

    # single colorbar, aligned to all subplots
    cbar = fig.colorbar(
        sc,
        ax=axes,
        location="right",
        shrink=1,
        pad=0.03
    )
    cbar.solids.set_alpha(0.6)

    ticks = tf.tf_dict_ticks[transf_full_name]
    cbar.set_ticks(ticks)
    scale = tf.tf_dict_scale[transf_full_name]
    cbar.set_label(scale)

    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    path = os.path.join(OUTPUT_FOLDER, f"chart_{DATASET}_{genres}_{transf}")
    plt.savefig(path+".png")