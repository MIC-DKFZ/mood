from collections import defaultdict
import json
import os

import numpy as np


def generate_from_folders(base_path, mode="sample", avg_cases=True):

    sub_fldrs = [el for el in sorted(os.listdir(base_path), reverse=True) if os.path.isdir(os.path.join(base_path, el))]

    submission_dict_brain = dict()
    submission_dict_colon = dict()

    for fldr in sub_fldrs:
        fldr_path = os.path.join(base_path, fldr)

        if (
            not os.path.exists(os.path.join(fldr_path, f"brain_{mode}_scores.json"))
            or not os.path.exists(os.path.join(fldr_path, f"colon_{mode}_scores.json"))
            or not os.path.exists(os.path.join(fldr_path, f"config.json"))
        ):
            continue

        with open(os.path.join(fldr_path, "config.json"), "r") as fp_:
            username = json.load(fp_)["username"]

        if username in submission_dict_brain:
            continue

        with open(os.path.join(fldr_path, f"brain_{mode}_scores.json"), "r") as fp_:
            brain_scores = json.load(fp_)
        with open(os.path.join(fldr_path, f"colon_{mode}_scores.json"), "r") as fp_:
            colon_scores = json.load(fp_)

        if avg_cases:
            brain_scores = [np.mean(brain_scores)]
            colon_scores = [np.mean(colon_scores)]

        submission_dict_brain[username] = brain_scores
        submission_dict_colon[username] = colon_scores

        print(username, fldr)

    ranking_csv = os.path.join(base_path, f"{mode}_ranking.csv")
    if not avg_cases:
        ranking_csv = os.path.join(base_path, f"{mode}_ranking_noavg.csv")
    with open(ranking_csv, "w") as fp_:
        fp_.write('"task","alg_name","value","case"\n')

    with open(ranking_csv, "a") as fp_:
        for u_name, subs in submission_dict_brain.items():
            for i, val in enumerate(subs):
                fp_.write(f'"brain","{u_name}",{val},"{i}"\n')
        for u_name, subs in submission_dict_colon.items():
            for i, val in enumerate(subs):
                fp_.write(f'"colon","{u_name}",{val},"{i}"\n')

    print("Done")

