r"""This is the implementation of Hybrid Bonsai:
Combines WL-based rKNN selection (Bonsai) with Density Peak-based selection (LeadingTree).
Includes Rank-Based Normalization, Detailed Logging, and FGW Node Addition Evaluation.
"""

import gc
import copy
import time
import typing as t
import argparse
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

import networkx as nx
import numpy as np
from scipy.sparse import coo_array

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from tensorboardX import SummaryWriter

# === 导入原项目依赖 ===
from utils import (
    wl2rknn,
    select_max_coverage_rknn_celf,
)
from utils import transform_features_with_tree
from model import GCN, GCN_inductive
from WL_Distance2 import compute_wl_representations

# === 导入新增的 FGW 评估模块 ===
from fgw_selector import fgw_budget_select

# === 导入你独立出来的 LeadingTree 模块 ===
# 请确保你的文件命名为 leading_tree.py
from leading_tree import LeadingTree


# 辅助计算 dc (局部截断距离) 的函数
def get_dc(dist_matrix, percent=2.0):
    temp = dist_matrix.reshape(-1)
    temp = temp[temp > 1e-6]
    if len(temp) == 0: return 0.1
    k = max(int(len(temp) * (percent / 100)), 1)
    try:
        dc = np.partition(temp, k)[k]
    except:
        dc = 0.1
    return dc


ADJ = None
FEAT_MULTIPLIER = 1
GLOBAL_NEIGHBORS_DICT = {}
GLOBAL_FEATS = None
FEAT_LEN = None
SAINT_DATASETS = ["flickr", "ogbn-arxiv", "reddit"]

def log(x: str) -> None:
    pass

def load_dataset(dataset_name: str, root: t.Union[str, Path]) -> dict:
    if dataset_name in SAINT_DATASETS:
        from sklearn.preprocessing import StandardScaler
        from load_saint_dataset import load_saint_dataset
        data = load_saint_dataset(dataset_name, root=root)
        feat_full = data.x.cpu().numpy()
        scaler = StandardScaler()
        scaler.fit(feat_full)
        feat_full = torch.tensor(scaler.transform(feat_full))
        data.x_normed = feat_full
        return {"data": data, "scaler": lambda feat: scaler.transform(feat)}
    from torch_geometric.datasets import Planetoid
    data = Planetoid(root=root, name=dataset_name)._data
    def nop_scaler(x):
        return x
    return {"data": data, "scaler": nop_scaler}

def dist2rknn_sorting(WL_dist: np.ndarray, sampled_nodes: np.ndarray, k: int) -> t.List[int]:
    rknn_result = wl2rknn(WL_dist, sampled_nodes=sampled_nodes, k=k)
    sorted_nodes = select_max_coverage_rknn_celf(rknn_result["rknn"])
    return sorted_nodes

def train_backend_inductive(model, nepochs, data, data_syn, splits, writer):
    opt = optim.Adam(model.parameters())
    loss_fn = F.nll_loss
    loop = tqdm(range(nepochs), ascii=False, ncols=120, desc="Training")
    best_acc_val = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        loss = loss_fn(out[d_syn.target], d_syn.y[d_syn.target])
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
            acc = accuracy_score(d.y[val].cpu().numpy(), preds)
            writer.add_scalar("acc/val", acc, epoch)
            if acc > best_acc_val:
                best_acc_val = acc
                weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(weights)
    with torch.no_grad():
        out = model(d.x, d.adj)
        preds = out[test].max(1)[1].cpu().numpy()
        acc = accuracy_score(d.y[test].cpu().numpy(), preds)
    writer.add_scalar("test_acc/test", acc)
    return acc

def train_backend_pyg(model, nepochs, data, data_syn, splits, writer):
    opt = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    loop = tqdm(range(nepochs), ascii=False, ncols=120, desc="Training")
    best_acc_val = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    d = data
    d = d.to(device, "x", "edge_index", "y")
    d_syn = data_syn
    d_syn = d_syn.to(device, "x", "edge_index", "y", "target")
    test = splits["test"]
    val = splits["val"]

    try:
        from my_profiling import profile
        profiler = profile(False)
    except ImportError:
        class DummyProfiler:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        profiler = DummyProfiler()

    with profiler:
        for epoch in loop:
            model.train()
            out = model(d_syn.x, d_syn.edge_index)
            loss = loss_fn(out[d_syn.target], d_syn.y[d_syn.target])
            opt.zero_grad()
            loss.backward()
            opt.step()
            writer.add_scalar("loss/train", loss.item(), epoch)
            model.eval()
            with torch.no_grad():
                out = model(d.x, d.edge_index)
                loss = loss_fn(out[val], d.y[val])
                writer.add_scalar("loss/val", loss.item(), epoch)
                preds = out[val].max(1)[1].cpu().numpy()
                acc = accuracy_score(d.y[val].cpu().numpy(), preds)
                writer.add_scalar("acc/val", acc, epoch)
                if acc > best_acc_val:
                    best_acc_val = acc
                    weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(weights)
    with torch.no_grad():
        out = model(d.x, d.edge_index)
        preds = out[test].max(1)[1].cpu().numpy()
        acc = accuracy_score(d.y[test].cpu().numpy(), preds)
    writer.add_scalar("test_acc/test", acc)
    return acc

def train_model(model_type, model, nepochs, data, data_syn, splits, writer):
    if model_type == "GCN_inductive":
        return train_backend_inductive(model, nepochs, data, data_syn, splits, writer)
    return train_backend_pyg(model, nepochs, data, data_syn, splits, writer)

def size(nnodes, nedges, feats, dtype="int"):
    mx = 1 if dtype == "int" else 2 if dtype == "float" else None
    return (nnodes * feats * mx + nedges * 2) * 2

def build_neighborhood_dict_sparse():
    neighbors = defaultdict()
    for node in tqdm(
            range(ADJ.shape[0]),
            ascii=False,
            total=ADJ.shape[0],
            ncols=120,
            desc="build neighbor",
    ):
        node_nbrs = set(ADJ[[node]].tocoo().col)
        node_nbrs.add(node)
        neighbors[node] = node_nbrs
    return neighbors

def repr_to_dist(degree_weighted_repr, frac_to_sample=1):
    if frac_to_sample == 1:
        distance_matrix = pairwise_distances(
            np.array(degree_weighted_repr), n_jobs=20
        )
        nodes = np.asarray(list(range(len(degree_weighted_repr))))
    else:
        n = len(degree_weighted_repr)
        m = min(max(int(frac_to_sample * n), 1), n)
        nodes = np.random.choice(range(n), (m,), replace=False)
        node_repr = [degree_weighted_repr[node] for node in nodes]
        distance_matrix = pairwise_distances(
            np.array(node_repr), np.array(degree_weighted_repr), n_jobs=20
        )
    return distance_matrix, nodes

def match_distribution(merged_graph, data, train, rknn_ranked_nodes):
    rknn_ranked_nodes_real = [train[node] for node in rknn_ranked_nodes]

    num_nodes = sum(1 for _, v in merged_graph.nodes(data=True) if v["target"])
    labels_train = data.y[train]
    class_counts = torch.bincount(labels_train).float()
    total_train_nodes = len(train)
    class_distribution_scaled = (class_counts / total_train_nodes) * num_nodes
    class_distribution_scaled = class_distribution_scaled.long()

    class_dict = {i: [] for i in range(len(class_distribution_scaled))}
    for node in rknn_ranked_nodes_real:
        node_class = data.y[node].item()
        class_dict[node_class].append(node)

    all_nodes_to_add = set()
    for class_label, scaled_count in enumerate(class_distribution_scaled):
        merged_class_nodes = [
            node
            for node, node_attrs in merged_graph.nodes(data=True)
            if data.y[node].item() == class_label and node_attrs["target"]
        ]

        lower_bound = 0.99 * scaled_count.item()
        upper_bound = 1.01 * scaled_count.item()
        current_count = len(merged_class_nodes)

        if current_count < lower_bound:
            nodes_to_add = class_dict[class_label]
            cnt = 0
            for node in nodes_to_add:
                if node not in merged_graph:
                    merged_graph.add_node(node)
                    all_nodes_to_add.add(node)
                    cnt += 1
                if cnt >= lower_bound - current_count:
                    break

        elif current_count > upper_bound:
            def sort_key(node):
                try:
                    return rknn_ranked_nodes_real.index(node)
                except ValueError:
                    return float('inf')

            merged_class_nodes_sorted = sorted(
                merged_class_nodes, key=lambda node: sort_key(node)
            )
            nodes_to_remove = merged_class_nodes_sorted[
                              -(current_count - int(upper_bound)) :
                              ]
            for node in nodes_to_remove:
                if node in merged_graph:
                    merged_graph.remove_node(node)

    P1 = lambda u, v: u in all_nodes_to_add and v in all_nodes_to_add
    P2 = lambda u, v: u in all_nodes_to_add and v in merged_graph.nodes
    P3 = lambda u, v: u in merged_graph.nodes and v in all_nodes_to_add

    for i in range(data.edge_index.shape[1]):
        u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        if P1(u, v) or P2(u, v) or P3(u, v):
            if not merged_graph.has_edge(u, v):
                merged_graph.add_edge(u, v)
    return merged_graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_size_frac", required=True, type=float)
    parser.add_argument("--nepochs", default=100, type=int)
    parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--dataset", default="cora", choices=["cora", "citeseer", "pubmed", "ogbn-products", "flickr", "ogbn-arxiv", "reddit"])
    parser.add_argument("--alpha", default=0.5, type=float, help="Weight for Bonsai (0~1).")
    parser.add_argument("--use_fgw", action="store_true", default=False, help="Enable FGW dynamic node evaluation.")
    parser.add_argument("--search_window", default=3, type=int, help="FGW candidate window size.")
    parser.add_argument("--fgw_alpha", default=0.5, type=float, help="FGW loss alpha parameter.")
    parser.add_argument("--lt_num", default=None, type=int, help="Number of LeadingTree roots.")

    args = parser.parse_args()

    dataset = load_dataset(args.dataset, root="datasets")
    data = dataset["data"]
    scaler = dataset["scaler"]

    nnodes = data.x.shape[0]
    nfeats = data.x.shape[1]
    nedges = data.edge_index.shape[1]

    row, col = data.edge_index
    weights = np.ones(len(row))
    row, col = row.numpy(), col.numpy()
    adj = coo_array((weights, (row, col)), shape=(nnodes, nnodes))
    adj = adj.tocsr()

    global ADJ, GLOBAL_NEIGHBORS_DICT, GLOBAL_FEATS, FEAT_LEN, FEAT_MULTIPLIER
    ADJ = adj
    dtype = ("int" if args.dataset in ["cora", "citeseer", "flickr"] else "float")
    size_full = size(nnodes, nedges, nfeats, dtype)
    target_size = float(f"{args.target_size_frac * size_full:.2f}")
    nclasses = len(set(data.y.reshape(-1).tolist()))

    train, test = train_test_split(range(nnodes), test_size=0.2, random_state=42)
    train = np.array(train)

    rng = np.random.RandomState(seed=42)
    idx_train = rng.choice(train, size=int(0.7 * len(train)), replace=False)
    idx_val = list(set(range(nnodes)) - set(idx_train).union(set(test)))
    splits = {"train": idx_train, "val": idx_val, "test": test}
    train = idx_train
    log("split done")

    GLOBAL_FEATS = data.x
    GLOBAL_NEIGHBORS_DICT = build_neighborhood_dict_sparse()

    degree_weighted_repr = compute_wl_representations(data.x, adj)
    degree_weighted_repr = degree_weighted_repr[train]

    log("begin dtree")
    if args.dataset != 'reddit':
        degree_weighted_repr, features_used = transform_features_with_tree(
            data, degree_weighted_repr, train
        )
        data.x = data.x[:, features_used]
    else:
        features_used = list(range(data.x.shape[1]))

    GLOBAL_FEATS = data.x

    if args.dataset not in ["ogbn-arxiv", "reddit", "flickr", "PubMed"]:
        FEAT_LEN = []
        FEAT_MULTIPLIER = 1
        nfeats = len(features_used)
        for x in range(data.x.shape[0]):
            FEAT_LEN.append(data.x[x].sum().item())
    elif args.dataset in ["PubMed", "flickr"]:
        FEAT_MULTIPLIER = 2 if args.dataset == "flickr" else 3
        FEAT_LEN = []
        nfeats = len(features_used)
        for x in range(data.x.shape[0]):
            nnz = torch.where(data.x[x] == 0, 0, 1).sum().item()
            FEAT_LEN.append(nnz)
    else:
        FEAT_LEN = []
        FEAT_MULTIPLIER = 2
        nfeats = len(features_used)
        FEAT_LEN = defaultdict(lambda: nfeats)

    log("Creating WL now")
    start_time = time.perf_counter()

    if args.dataset in ["reddit", "ogbn-arxiv", "ogbn-products"]:
        WL_dist, sampled_nodes = repr_to_dist(degree_weighted_repr, frac_to_sample=0.2)
    else:
        WL_dist, sampled_nodes = repr_to_dist(degree_weighted_repr)

    end_time = time.perf_counter()
    log("Created WL now")

    # ================= [计算混合排名 (Bonsai + LT)] =================
    print(f"--- Calculating Hybrid Base Scores (Alpha={args.alpha}) ---")

    bonsai_sorted_indices = dist2rknn_sorting(WL_dist, sampled_nodes, args.k)

    dc_val = get_dc(WL_dist, percent=2.0)
    lt_num = args.lt_num if args.lt_num is not None else max(nclasses * 3, 40)

    # ==== 调用外部的 LeadingTree ====
    lt = LeadingTree(dc=dc_val, lt_num=lt_num, D=WL_dist)
    lt.fit()

    # 从类实例中获取对应的值
    node_gamma = lt.gamma
    node_layer = lt.layer
    num_sampled = len(sampled_nodes)

    bonsai_rank_map = {idx: rank for rank, idx in enumerate(bonsai_sorted_indices)}
    bonsai_norm_map = {}
    for idx in range(num_sampled):
        rank = bonsai_rank_map.get(idx, num_sampled)
        score = 1.0 - (rank / num_sampled)
        bonsai_norm_map[idx] = score

    lt_raw_scores = []
    for i in range(num_sampled):
        layer_val = node_layer[i] if node_layer[i] > 0 else 1.0
        raw_val = node_gamma[i] * (1.0 / np.sqrt(layer_val))
        lt_raw_scores.append((i, raw_val))

    lt_raw_scores.sort(key=lambda x: x[1], reverse=True)
    lt_norm_map = {}
    for rank, (idx, val) in enumerate(lt_raw_scores):
        score = 1.0 - (rank / num_sampled)
        lt_norm_map[idx] = score

    alpha = args.alpha
    final_scores = []

    for idx in range(num_sampled):
        s_bonsai = bonsai_norm_map.get(idx, 0.0)
        s_lt = lt_norm_map.get(idx, 0.0)
        final_s = alpha * s_bonsai + (1 - alpha) * s_lt
        final_scores.append((idx, final_s))

    # 这是 FGW 介入前的“原始静态排序”
    final_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_nodes = [x[0] for x in final_scores]
    static_rank_map = {node_idx: rank for rank, node_idx in enumerate(sorted_nodes)}

    accs = {}
    del WL_dist
    gc.collect()

    # --- 调用 FGW 选择模块 ---
    def compute_ogsize_nodes():
        original_nodes, _ = fgw_budget_select(
            sorted_nodes, train, target_size,
            GLOBAL_NEIGHBORS_DICT, GLOBAL_FEATS, FEAT_LEN, FEAT_MULTIPLIER,
            use_fgw=args.use_fgw, search_window=args.search_window, fgw_alpha=args.fgw_alpha
        )
        merged_nodes = set()
        for org_id_node in original_nodes:
            train_id_node = train[org_id_node]
            nbr_nodes = GLOBAL_NEIGHBORS_DICT[train_id_node]
            for nbrs in nbr_nodes:
                merged_nodes.add(nbrs)
            merged_nodes.add(train_id_node)
        return len(merged_nodes)

    ogsize = compute_ogsize_nodes()

    for m in [0.9]:
        log(f"{m = }")
        upscale = 1 + m / (1 - m)

        # --- 第二次真实调用 FGW 模块，获取最终加点顺序和 FGW 距离 ---
        size_selected_nodes, fgw_metrics = fgw_budget_select(
            sorted_nodes, train, target_size * upscale,
            GLOBAL_NEIGHBORS_DICT, GLOBAL_FEATS, FEAT_LEN, FEAT_MULTIPLIER,
            use_fgw=args.use_fgw, search_window=args.search_window, fgw_alpha=args.fgw_alpha
        )

        # ================= [打印最终带 FGW 距离的动态 Top-20] =================
        print("\n" + "="*115)
        print(f"🚀 FGW Dynamic Selection Top 40 (Window={3}, Alpha={alpha})")
        print(f"{'Add Step':<10} | {'NodeID':<8} | {'Orig Rank':<10} | {'Bonsai Norm':<12} | {'LT Norm':<10} | {'FGW Dist (Loss)':<15}")
        print("-" * 115)

        for step, (idx, fgw_dist) in enumerate(fgw_metrics[:40]):
            real_id = train[sampled_nodes[idx]]
            orig_rank = static_rank_map.get(idx, -1)
            b_norm = bonsai_norm_map.get(idx, 0.0)
            l_norm = lt_norm_map.get(idx, 0.0)

            fgw_str = f"{fgw_dist:.6f}" if fgw_dist > 0 else "0.0 (No FGW)"

            print(f"{step:<10} | {real_id:<8} | {orig_rank:<10} | {b_norm:<12.4f} | {l_norm:<10.4f} | {fgw_str:<15}")
        print("="*115 + "\n")
        # ======================================================================

        log("selected nodes")
        merged_graph = nx.Graph()

        for org_id_node in size_selected_nodes:
            train_id_node = train[org_id_node]
            nbr_nodes = GLOBAL_NEIGHBORS_DICT[train_id_node]
            merged_graph.add_node(train_id_node)
            row_center = adj[[train_id_node]]
            for nbr in nbr_nodes:
                merged_graph.add_node(nbr)
                row_nbr = adj[[nbr]]
                ego = row_center.multiply(row_nbr)
                nbrnbrs = ego.tocoo().col
                for nbrnbr in nbrnbrs:
                    edge = (nbr, nbrnbr)
                    edge = sorted(edge)
                    merged_graph.add_edge(*edge)
                edge = (train_id_node, nbr)
                edge = sorted(edge)
                merged_graph.add_edge(*edge)

        for node in merged_graph.nodes:
            merged_graph.nodes[node]["target"] = (node in train)
            merged_graph.nodes[node]["x"] = data.x[node].numpy().tolist()
            merged_graph.nodes[node]["y"] = data.y[node]

        personalization = {node: 0 for node in merged_graph.nodes()}
        l = 1.0 / len(size_selected_nodes)
        log("before personlisation")
        for node in size_selected_nodes:
            personalization[train[node]] = l
        ppr = sorted(
            nx.pagerank(merged_graph, personalization=personalization).items(),
            key=lambda x: x[1],
        )
        todel = len(ppr) - ogsize
        if todel < 0: todel = 0
        nodes_todel = []
        log(f"{todel=}")
        for node, _ in ppr:
            nodes_todel.append(node)
            if len(nodes_todel) >= todel:
                break
        for node in nodes_todel:
            merged_graph.remove_node(node)

        final_graph = match_distribution(merged_graph, data, train, sorted_nodes)
        merged_graph = final_graph
        for node in merged_graph.nodes:
            merged_graph.nodes[node]["target"] = (node in train)
            merged_graph.nodes[node]["x"] = data.x[node].numpy().tolist()
            merged_graph.nodes[node]["y"] = data.y[node]
        merged_data = from_networkx(merged_graph, group_node_attrs=["x"])
        if args.save:
            save_root = Path("saved_ours")
            save_dir = save_root / f"{args.dataset}-{args.target_size_frac}"
            save_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
            save_file = save_dir / f"data_m_{m}.pt"
            torch.save(
                {"data": merged_data, "features_used": features_used}, save_file
            )
        model_type = "GCN" if args.dataset not in SAINT_DATASETS else "GCN_inductive"
        hidim = 128 if args.dataset not in SAINT_DATASETS else 1024
        model = globals()[model_type](nfeats, nclasses, hidim=hidim)
        writer = SummaryWriter(
            f"tensorboard_logs/{args.dataset}_{model_type}_{args.target_size_frac}"
        )
        if args.dataset in SAINT_DATASETS:
            merged_data.x = torch.tensor(scaler(merged_data.x.cpu().numpy()))
            import scipy.sparse as sp
            from torch_sparse import SparseTensor

            d = np.ones(merged_data.edge_index.shape[1])
            r = merged_data.edge_index[0].cpu().numpy()
            c = merged_data.edge_index[1].cpu().numpy()
            n = merged_data.x.shape[0]
            adj_m = sp.csr_matrix((d, (r, c)), shape=(n, n))
            adj_m = adj_m.tolil()
            adj_m = adj_m + sp.eye(adj_m.shape[0])
            rowsum = np.array(adj_m.sum(1))
            r_inv = np.power(rowsum, -1 / 2).flatten()
            r_inv[np.isinf(r_inv)] = 0.0
            r_mat_inv = sp.diags(r_inv)
            adj_m = r_mat_inv.dot(adj_m)
            adj_m = adj_m.dot(r_mat_inv)
            adj_m = adj_m.tocoo().astype(np.float32)
            sparserow = torch.LongTensor(adj_m.row).unsqueeze(1)
            sparsecol = torch.LongTensor(adj_m.col).unsqueeze(1)
            sparseconcat = torch.cat((sparserow, sparsecol), 1)
            sparsedata = torch.FloatTensor(adj_m.data)
            adj_m = torch.sparse.FloatTensor(
                sparseconcat.t(), sparsedata, torch.Size(adj_m.shape)
            )
            adj_m = SparseTensor(
                row=adj_m._indices()[0],
                col=adj_m._indices()[1],
                value=adj_m._values(),
                sparse_sizes=adj_m.size(),
            )
            merged_data.adj = adj_m
        data.x = (
            data.x_normed if hasattr(data, "x_normed") else data.x
        )
        kwargs = {
            "model": model,
            "model_type": model_type,
            "data_syn": merged_data,
            "nepochs": args.nepochs,
            "data": data,
            "splits": splits,
        }
        acc = 0
        var = 0
        nruns = 5
        for run_num in range(1, nruns + 1):
            from timing import Timer

            writer = SummaryWriter(
                f"logs/bonsai_{args.dataset}_{model_type}_{args.target_size_frac}_{run_num}"
            )
            with Timer(f"Training Bonsai for {args.nepochs} epochs") as timer:
                run_acc = train_model(**kwargs, writer=writer)
            writer.add_scalar("experiment/time", timer.dur)
            delta = run_acc - acc
            acc += (run_acc / run_num) - (acc / run_num)
            delta2 = run_acc - acc
            var += delta * delta2
        std = np.sqrt(var / nruns)
        accs[m] = rf"{acc*100:.2f}\pm {std*100:.2f}"
    for _, v in accs.items():
        print(v)

if __name__ == "__main__":
    main()