import gc
import copy
import time
import typing as t
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from utils import transform_features_with_tree
from model import GCN, GCN_inductive

ADJ = None
FEAT_MULTIPLIER = 1
GLOBAL_NEIGHBORS_DICT = {}
GLOBAL_FEATS = None
FEAT_LEN = None
SAINT_DATASETS = ["flickr", "ogbn-arxiv", "reddit"]


def log(x: str) -> None:
    pass

def train_backend_inductive(
        model,
        nepochs,
        data,
        data_syn,
        splits,
        writer
):
    opt = optim.Adam(model.parameters())
    loss_fn = F.nll_loss

    loop = tqdm(
        range(nepochs),
        ascii=False,
        ncols=120,
        desc="Training"
    )

    best_acc_val = 0

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    d = data
    d = d.to(device, "x", "adj", "y")

    d_syn = data_syn
    d_syn = d_syn.to(device, "x", "adj", "y", "target")

    test = splits["test"]
    val = splits["val"]

    for epoch in loop:
        model.train()

        out = model(d_syn.x, d_syn.adj)

        loss = loss_fn(
            out[d_syn.target],
            d_syn.y[d_syn.target]
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar("loss/train", loss.item(), epoch)

        model.eval()

        with torch.no_grad():
            out = model(d.x, d.adj)

            loss = loss_fn(out[val], d.y[val])

            writer.add_scalar("loss/val", loss.item(), epoch)

            preds = out[val].max(1)[1].cpu().numpy()

            acc = accuracy_score(
                d.y[val].cpu().numpy(),
                preds
            )

            writer.add_scalar("acc/val", acc, epoch)

            if acc > best_acc_val:
                best_acc_val = acc
                weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(weights)

    with torch.no_grad():
        out = model(d.x, d.adj)

        preds = out[test].max(1)[1].cpu().numpy()

        acc = accuracy_score(
            d.y[test].cpu().numpy(),
            preds
        )

    writer.add_scalar("test_acc/test", acc)

    return acc
