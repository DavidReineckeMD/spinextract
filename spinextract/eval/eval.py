"""Evaluation modules and script.

Copyright (c) 2022 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import os
import logging
from shutil import copy2
from functools import partial
from typing import List, Union, Dict, Any

import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import torch
from torchvision.transforms import Compose

import pytorch_lightning as pl
from torchmetrics import AveragePrecision, Accuracy

from spinextract.datasets.srh_dataset import SRHClassificationDataset
from spinextract.datasets.improc import get_srh_base_aug, get_srh_vit_base_aug
from spinextract.train.common import (parse_args, get_exp_name, config_loggers,
                                  get_num_worker)
from spinextract.train.train_ce import CESystem



def get_embeddings(cf: Dict[str, Any],
                   exp_root: str) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Run forward pass on the dataset, and generate embeddings and logits"""
    # get model
    if cf["model"]["backbone"] == "resnet50":
        aug_func = get_srh_base_aug
    elif cf["model"]["backbone"] == "vit":
        aug_func = get_srh_vit_base_aug
    else:
        raise NotImplementedError()

    # get dataset / loader
    val_dset = SRHClassificationDataset(data_root=cf["data"]["db_root"],
                                        studies="val",
                                        transform=Compose(aug_func()),
                                        balance_patch_per_class=False)

    val_loader = torch.utils.data.DataLoader(val_dset,
                                             batch_size=32,
                                             drop_last=False,
                                             pin_memory=True,
                                             num_workers=get_num_worker(),
                                             persistent_workers=True)

    # create lightning module
    if cf["eval"].get("imagenet_checkpoint", None):
        # load resnet imagenet checkpoint in pytorch format
        ckpt_dict = torch.load(cf["eval"]["imagenet_checkpoint"],
                               map_location="cpu")
        del ckpt_dict["fc.weight"]
        del ckpt_dict["fc.bias"]
        model = CESystem(cf=cf, num_it_per_ep=0, nc=2)
        model.model.bb.load_state_dict(ckpt_dict)
    else:
        # load lightning checkpoint
        ckpt_path = os.path.join(cf["infra"]["log_dir"],
                                 cf["infra"]["exp_name"],
                                 cf["eval"]["ckpt_path"])
        model = CESystem.load_from_checkpoint(ckpt_path,
                                              cf=cf,
                                              num_it_per_ep=0,
                                              max_epochs=-1,
                                              nc=2)

    # create trainer
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         max_epochs=-1,
                         default_root_dir=exp_root,
                         enable_checkpointing=False,
                         logger=False)

    # generate predictions
    predictions = trainer.predict(model, dataloaders=val_loader)

    # aggregate predictions
    pred = {}
    for k in predictions[0].keys():
        if k == "path":
            pred[k] = [pk for p in predictions for pk in p[k][0]]
        else:
            pred[k] = torch.cat([p[k] for p in predictions])

    return pred


def make_specs(predictions: Dict[str, Union[torch.Tensor, List[str]]]) -> None:
    """Compute all specs for an experiment"""

    # aggregate prediction into a dataframe
    pred = pd.DataFrame.from_dict({
        "path":
        predictions["path"],
        "labels": [l.item() for l in list(predictions["label"])],
        "logits": [l.tolist() for l in list(predictions["logits"])]
    })
    pred["logits"] = pred["logits"].apply(
        lambda x: torch.nn.functional.softmax(torch.tensor(x), dim=0))

    # add patient and slide info from patch paths
    pred["patient"] = pred["path"].apply(lambda x: x.split("/")[-4])
    pred["slide"] = pred["path"].apply(lambda x: "/".join(
        [x.split("/")[-4], x.split("/")[-3]]))

    # aggregate logits
    get_agged_logits = lambda pred, mode: pd.DataFrame(
        pred.groupby(by=[mode, "labels"])["logits"].apply(
            lambda x: [sum(y) for y in zip(*x)])).reset_index()

    slides = get_agged_logits(pred, "slide")
    patients = get_agged_logits(pred, "patient")

    normalize_f = lambda x: torch.nn.functional.normalize(x, dim=1, p=1)
    patch_logits = normalize_f(torch.tensor(np.vstack(pred["logits"])))
    slides_logits = normalize_f(torch.tensor(np.vstack(slides["logits"])))
    patient_logits = normalize_f(torch.tensor(np.vstack(patients["logits"])))

    patch_label = torch.tensor(pred["labels"])
    slides_label = torch.tensor(slides["labels"])
    patient_label = torch.tensor(patients["labels"])

    # generate metrics
    def get_all_metrics(logits, label):
        map = AveragePrecision(num_classes=7)
        acc = Accuracy(num_classes=7)
        t2 = Accuracy(num_classes=7, top_k=2)
        t3 = Accuracy(num_classes=7, top_k=3)
        mca = Accuracy(num_classes=7, average="macro")

        acc_val = acc(logits, label)
        t2_val = t2(logits, label)
        t3_val = t3(logits, label)
        mca_val = mca(logits, label)
        map_val = map(logits, label)

        fn = (logits.argmax(dim=1) == 4) & (label != 4)
        fnr = fn.sum() / len(fn)

        return torch.stack((acc_val, t2_val, t3_val, mca_val, map_val, fnr))

    all_metrics = torch.vstack((get_all_metrics(patch_logits, patch_label),
                                get_all_metrics(slides_logits, slides_label),
                                get_all_metrics(patient_logits,
                                                patient_label)))
    all_metrics = pd.DataFrame(
        all_metrics,
        columns=["acc", "t2", "t3", "mca", "map", "fnr"],
        index=["patch", "slide", "patient"])

    # generate confusion matrices
    patch_conf = confusion_matrix(y_true=patch_label,
                                  y_pred=patch_logits.argmax(dim=1))

    slide_conf = confusion_matrix(y_true=slides_label,
                                  y_pred=slides_logits.argmax(dim=1))

    patient_conf = confusion_matrix(y_true=patient_label,
                                    y_pred=patient_logits.argmax(dim=1))

    print("\nmetrics")
    print(all_metrics)
    print("\npatch confusion matrix")
    print(patch_conf)
    print("\nslide confusion matrix")
    print(slide_conf)
    print("\npatient confusion matrix")
    print(patient_conf)

    return


def setup_eval_paths(cf, get_exp_name, cmt_append):
    """Get name of the ouput dirs and create them in the file system."""
    log_root = cf["infra"]["log_dir"]
    exp_name = cf["infra"]["exp_name"]
    instance_name = cf["eval"]["ckpt_path"].split("/")[0]
    eval_instance_name = "_".join([get_exp_name(cf), cmt_append])
    exp_root = os.path.join(log_root, exp_name, instance_name, "evals",
                            eval_instance_name)

    # generate needed folders, evals will be embedded in experiment folders
    pred_dir = os.path.join(exp_root, 'predictions')
    config_dir = os.path.join(exp_root, 'config')
    for dir_name in [pred_dir, config_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    # if there is a previously generated prediction, also return the
    # prediction filename so we don't have to predict again
    if cf["eval"].get("eval_predictions", None):
        other_eval_instance_name = cf["eval"]["eval_predictions"]
        pred_fname = os.path.join(log_root, exp_name, instance_name, "evals",
                                  other_eval_instance_name, "predictions",
                                  "predictions.pt")
    else:
        pred_fname = None

    return exp_root, pred_dir, partial(copy2, dst=config_dir), pred_fname

CONF_WEIGHT_PMAX = 0.7
CONF_WEIGHT_AGREE = 0.3
CONF_HIGH_THRESHOLD = 0.9

def _conf_score(P: np.ndarray, pred_idx: int, w_pmax: float = CONF_WEIGHT_PMAX, w_agree: float = CONF_WEIGHT_AGREE) -> float:
    if P.size == 0:
        return 0.0
    p_max_mean = float(P[:, pred_idx].mean())
    agreement = float((P.argmax(axis=1) == pred_idx).mean())
    s = w_pmax * p_max_mean + w_agree * agreement
    return max(0.0, min(1.0, s))

def _extract_patient_slide(p: str):
    parts = p.split("/")
    patient = parts[-4]
    slide = "/".join([parts[-4], parts[-3]])
    return patient, slide

def _to_probs(logits_tensor: Union[torch.Tensor, List[List[float]]]) -> np.ndarray:
    if isinstance(logits_tensor, torch.Tensor):
        return torch.softmax(logits_tensor, dim=1).detach().cpu().numpy()
    x = np.asarray(logits_tensor, dtype=np.float32)
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)

def save_pred_csvs(predictions: Dict[str, Union[torch.Tensor, List[str]]], out_dir: str,
                   conf_high_threshold: float = CONF_HIGH_THRESHOLD,
                   w_pmax: float = CONF_WEIGHT_PMAX, w_agree: float = CONF_WEIGHT_AGREE):
    paths: List[str] = predictions["path"]
    labels: List[int] = [int(l.item()) if hasattr(l, "item") else int(l) for l in list(predictions["label"])]
    logits = predictions["logits"]
    probs = _to_probs(logits)
    preds = probs.argmax(axis=1).astype(int)
    n_classes = probs.shape[1]
    pts = [_extract_patient_slide(p) for p in paths]
    patients = [t[0] for t in pts]
    slides = [t[1] for t in pts]
    df_cols = {
        "path": paths,
        "patient": patients,
        "slide": slides,
        "true_label": labels,
        "pred_label": preds,
        "confidence_score": probs.max(axis=1)
    }
    for i in range(n_classes):
        df_cols[f"prob_{i}"] = probs[:, i]
    patch_df = pd.DataFrame(df_cols)
    patch_df["high_confidence"] = patch_df["confidence_score"] >= conf_high_threshold
    patch_df.to_csv(os.path.join(out_dir, "predictions_patch.csv"), index=False)
    def _agg(df: pd.DataFrame, by: str, conf_colname: str, out_name: str):
        g = df.groupby(by)
        rows = []
        for key, grp in g:
            P = grp[[f"prob_{i}" for i in range(n_classes)]].to_numpy()
            avg_probs = P.mean(axis=0)
            pred_idx = int(avg_probs.argmax())
            row = {
                by: key,
                "true_label": int(grp["true_label"].iloc[0]),
                "pred_label": pred_idx,
                "patch_count": int(len(grp)),
                conf_colname: float(_conf_score(P, pred_idx, w_pmax, w_agree)),
                "high_confidence": False
            }
            for i in range(n_classes):
                row[f"avg_prob_{i}"] = float(avg_probs[i])
            row["high_confidence"] = bool(row[conf_colname] >= conf_high_threshold)
            rows.append(row)
        out_df = pd.DataFrame(rows)
        out_df.to_csv(os.path.join(out_dir, out_name), index=False)
        return out_df
    slide_df = _agg(patch_df, "slide", "confidence_score", "predictions_slide.csv")
    patient_df = _agg(patch_df, "patient", "confidence_score", "predictions_patient.csv")
    return patch_df, slide_df, patient_df


def main():
    """Driver script for evaluation pipeline."""
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    exp_root, pred_dir, cp_config, pred_fname = setup_eval_paths(
        cf, get_exp_name, "")
    pl.seed_everything(cf["infra"]["seed"])

    # logging and copying config files
    cp_config(cf_fd.name)
    config_loggers(exp_root)

    # get predictions
    if not cf["eval"].get("eval_predictions", None):
        logging.info("generating predictions")
        predictions = get_embeddings(cf, exp_root)
        torch.save(predictions, os.path.join(pred_dir, "predictions.pt"))
    else:
        logging.info("loading predictions")
        predictions = torch.load(pred_fname)

    # generate specs
    make_specs(predictions)
    # generate confidence scores for slide and patient-level
    patch_df, slide_df, patient_df = save_pred_csvs(predictions, pred_dir)



if __name__ == "__main__":
    main()
