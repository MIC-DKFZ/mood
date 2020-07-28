import json
import os
import random
import traceback

import nibabel as nib
import numpy as np
from sklearn import metrics


class InvalidPredictionException(Exception):
    pass


class CouldNotProcessException(Exception):
    pass


def process_file_pixelwise(pred_path, label_path):

    pred_list, label_list = [], []
    label_appended, pred_appended = False, False
    try:
        label_nimg = nib.load(label_path)
        label_array = np.rint(label_nimg.get_fdata()).astype(np.int)

        # should already be in that interval but just be sure
        label_array = np.clip(label_array, a_max=1, a_min=0)
        label_array = label_array == 1

        label_list = label_array.flatten()
        label_appended = True

        if os.path.exists(pred_path):

            pred_nimg = nib.load(pred_path)
            pred_array = pred_nimg.get_fdata(dtype=np.float16)

            if pred_array.shape != label_array.shape:
                raise InvalidPredictionException("Array shapes do not match", pred_path)

            # well predicitions should also be in [0,1]
            pred_array = np.clip(pred_array, a_max=1.0, a_min=0.0)

            pred_list = pred_array.flatten()
            pred_appended = True

        else:
            raise InvalidPredictionException("Prediction file not found", pred_path)

    except InvalidPredictionException:
        pred_array = np.zeros_like(label_array)
        pred_list = pred_array.flatten()
    except Exception:
        if label_appended and not pred_appended:
            pred_array = np.zeros_like(label_array)
            pred_list = pred_array.flatten()
        else:
            raise CouldNotProcessException("CouldNotProcessException")

    return pred_list, label_list


def process_file_samplewise(pred_path, label_path):

    label_appended, pred_appended = False, False
    try:

        with open(label_path, "r") as val_fl:
            val_str = val_fl.readline()
        label = int(val_str)

        label_appended = True

        if os.path.exists(pred_path):

            with open(pred_path, "r") as pred_fl:
                pred_str = pred_fl.readline()
            pred = float(pred_str)

            # predicitions should also be in [0,1]
            pred = np.clip(pred, a_max=1.0, a_min=0.0)

            pred_appended = True

        else:
            raise InvalidPredictionException("Prediction file not found", pred_path)

    except InvalidPredictionException:
        pred = 0.0
    except Exception:
        if label_appended and not pred_appended:
            pred = 0.0
        else:
            traceback.print_exc()
            raise CouldNotProcessException("CouldNotProcessException")

    return [pred], [label]


def eval_list(pred_file_list, label_file_list, mode="pixel"):

    label_vals = []
    pred_vals = []

    for pred_path, label_path in zip(pred_file_list, label_file_list):
        try:
            if mode == "pixel":
                pred_list, label_list = process_file_pixelwise(pred_path, label_path)
            elif mode == "sample":
                pred_list, label_list = process_file_samplewise(pred_path, label_path)
            else:
                pred_list, label_list = []
            pred_vals.append(pred_list)
            label_vals.append(label_list)
        except Exception:
            print(f"Smth went fundamentally wrong with {pred_path}")

    label_vals = np.concatenate(label_vals, axis=0)
    pred_vals = np.concatenate(pred_vals, axis=0)

    return metrics.average_precision_score(label_vals, pred_vals)


def eval_dir(pred_dir, label_dir, mode="pixel", save_file=None):

    pred_file_list = []
    label_file_list = []

    for f_name in sorted(os.listdir(label_dir)):

        pred_file_path = os.path.join(pred_dir, f_name)
        label_file_path = os.path.join(label_dir, f_name)

        pred_file_list.append(pred_file_path)
        label_file_list.append(label_file_path)

    score = eval_list(pred_file_list, label_file_list, mode=mode)

    if save_file is not None:
        with open(save_file, "w") as outfile:
            json.dump(score, outfile)

    return score


def bootstrap_dir(
    pred_dir, label_dir, splits_file=None, n_runs=10, n_files=2, save_dir=None, seed=123, mode="pixel",
):

    random.seed(seed)

    all_preds_file_list = []
    all_labels_file_list = []
    for f_name in sorted(os.listdir(label_dir)):

        pred_file_path = os.path.join(pred_dir, f_name)
        label_file_path = os.path.join(label_dir, f_name)

        all_preds_file_list.append(pred_file_path)
        all_labels_file_list.append(label_file_path)

    all_preds_file_list = np.array(all_preds_file_list)
    all_labels_file_list = np.array(all_labels_file_list)

    scores = []
    if splits_file is not None:
        with open(splits_file, "r") as json_file:
            split_list = json.load(json_file)

    else:
        split_list = []
        idx_list = list(range(len(all_labels_file_list)))
        split_list = [random.sample(idx_list, k=n_files) for r in range(n_runs)]

    for idx_sub_list in split_list:
        scores.append(eval_list(all_preds_file_list[idx_sub_list], all_labels_file_list[idx_sub_list], mode=mode,))

    if save_dir is not None:
        with open(os.path.join(save_dir, "splits.json"), "w") as outfile:
            json.dump(split_list, outfile)
        with open(os.path.join(save_dir, "scores.json"), "w") as outfile:
            json.dump(scores, outfile)

    return np.mean(scores)


def bootstrap_list(
    eval_lists, save_file=None, mode="pixel", base_pred_dir=None, base_label_dir=None,
):

    scores = []

    for pl_list in eval_lists:

        pred_lists, label_lists = zip(*pl_list)
        if base_pred_dir is not None:
            if mode == "pixel":
                pred_lists = [os.path.join(base_pred_dir, el) for el in pred_lists]
            if mode == "sample":
                pred_lists = [os.path.join(base_pred_dir, el + ".txt") for el in pred_lists]
        if base_label_dir is not None:
            label_lists = [os.path.join(base_label_dir, el) for el in label_lists]

        score = eval_list(pred_lists, label_lists, mode=mode,)

        if not np.isfinite(score):
            score = 0

        scores.append(score)

    if save_file is not None:
        with open(save_file, "w") as outfile:
            json.dump(scores, outfile)

    return np.mean(scores)
