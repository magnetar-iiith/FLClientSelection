# ============================================================
# Federated Learning – Full Baseline Suite (GPU Safe)
# ============================================================

import os
import copy
import math
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# 2. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 3. Model
# ============================================================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ============================================================
# 4. Dataset
# ============================================================

transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

test_loader = DataLoader(testset, batch_size=512, shuffle=False)

# ============================================================
# 5. Non-IID Clients
# ============================================================

# def dirichlet_partition(dataset, n_clients, alpha):
#     # labels = np.array(dataset.targets)
#     labels = dataset.targets.numpy()
#     n_classes = len(np.unique(labels))
#     idx_by_class = [np.where(labels == i)[0] for i in range(n_classes)]

#     client_indices = [[] for _ in range(n_clients)]

#     for c in range(n_classes):
#         proportions = np.random.dirichlet(alpha * np.ones(n_clients))
#         proportions = (np.cumsum(proportions) * len(idx_by_class[c])).astype(int)[:-1]
#         splits = np.split(idx_by_class[c], proportions)

#         for k in range(n_clients):
#             client_indices[k].extend(splits[k])

#     return client_indices

def dirichlet_partition(dataset, n_clients, alpha, min_size=10):
    labels = dataset.targets.numpy()
    n_classes = len(np.unique(labels))
    idx_by_class = [np.where(labels == i)[0] for i in range(n_classes)]

    while True:
        client_indices = [[] for _ in range(n_clients)]

        for c in range(n_classes):
            proportions = np.random.dirichlet(alpha * np.ones(n_clients))
            proportions = (np.cumsum(proportions) * len(idx_by_class[c])).astype(int)[:-1]
            splits = np.split(idx_by_class[c], proportions)

            for k in range(n_clients):
                client_indices[k].extend(splits[k])

        sizes = [len(ci) for ci in client_indices]

        # ensure no empty clients
        if min(sizes) >= min_size:
            break

    return client_indices

def create_non_iid_clients(K=20, samples_per_client=2000,
                           classes_per_client=2, good_fraction=0.3):

    label_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(trainset):
        label_indices[label].append(idx)

    clients = []
    num_good = int(K * good_fraction)

    for k in range(K):
        chosen = random.sample(range(10), classes_per_client)
        selected = []

        per_label = samples_per_client // classes_per_client
        for lbl in chosen:
            selected.extend(random.sample(label_indices[lbl], per_label))

        subset = Subset(trainset, selected)

        if k < num_good:
            clients.append({
                "dataset": subset,
                "epochs": 2,
                "lr": 0.03,
                "compute_time": 100,
                "quality": 2.0
            })
        else:
            clients.append({
                "dataset": subset,
                "epochs": 1,
                "lr": 0.05,
                "compute_time": 10,
                "quality": 0.5
            })

    return clients

# ============================================================
# 6. Utilities
# ============================================================

def copy_model(model):
    new_model = CNN().to(device)
    new_model.load_state_dict(model.state_dict())
    return new_model

def model_diff(local, global_model):
    with torch.no_grad():
        return {
            k: (local.state_dict()[k] - global_model.state_dict()[k]).cpu()
            for k in global_model.state_dict()
        }

def apply_update(global_model, delta, lr):
    with torch.no_grad():
        for k, v in global_model.state_dict().items():
            v += lr * delta[k].to(device)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# def evaluate_train(model, clients):
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for c in clients:
#             loader = DataLoader(c["dataset"], batch_size=256)
#             for x, y in loader:
#                 x, y = x.to(device), y.to(device)
#                 pred = model(x).argmax(1)
#                 correct += (pred == y).sum().item()
#                 total += y.size(0)
#     return correct / total
def evaluate_train(model, clients):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for c in clients:
            loader = DataLoader(
                Subset(trainset, c["indices"]),
                batch_size=256
            )
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

    return correct / total
    #return np.random.rand()

# def local_train(global_model, client):
#     model = copy_model(global_model)
#     model.train()
#     loader = DataLoader(client["dataset"], batch_size=64, shuffle=True)
#     opt = torch.optim.SGD(model.parameters(), lr=client["lr"])

#     for _ in range(client["epochs"]):
#         for x, y in loader:
#             x, y = x.to(device), y.to(device)
#             opt.zero_grad()
#             loss = F.cross_entropy(model(x), y)
#             loss.backward()
#             opt.step()

#     return model
def local_train(global_model, client):
    model = copy_model(global_model)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=client["lr"]
    )

    loader = DataLoader(
        Subset(trainset, client["indices"]),
        batch_size=64,
        shuffle=True
    )

    for _ in range(client["local_epochs"]):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

    return model

# def sample_arrivals(clients, prob=0.05):
#     selected = []
#     for k, c in enumerate(clients):
#         if random.random() < prob:
#             selected.append(k)
#     return selected

def sample_arrivals(clients):
    selected = []
    for k, c in enumerate(clients):
        # fast clients arrive frequently
        if c["compute_time"] <= 10:
            prob = 0.2#0.2
        else:
            prob = 0.05#0.01   # slow clients are RARE
        if random.random() < prob:
            selected.append(k)
    return selected
# ============================================================
# 7. Algorithms
# ============================================================

def synchronous_fedavg(clients, MAX_TIME, LOG_INTERVAL, eta=1.0):
    W = CNN().to(device)
    wall, test_log, train_log = [], [], []

    current_time = 0
    last_log = 0
    next_log = LOG_INTERVAL

    # for r in range(R):
    while current_time < MAX_TIME:
        # sample subset (fairness fix)
        active_clients = random.sample(clients, min(5, len(clients)))
        deltas = []
        # for c in active_clients:#clients:
        for c in clients:
            local = local_train(W, c)
            deltas.append(model_diff(local, W))
            del local

        round_time = max(c["compute_time"] for c in clients)

        for delta in deltas:
            apply_update(W, delta, eta / len(deltas))

        for _ in range(round_time):
            if current_time >= MAX_TIME:
                break
            current_time += 1
            wall.append(current_time)
            test_log.append(evaluate(W, test_loader))
            train_log.append(evaluate_train(W, clients))

        torch.cuda.empty_cache()

    return wall, test_log, train_log


def power_of_choice(clients, MAX_TIME, LOG_INTERVAL, m=5,topK=2, eta=0.2):
    W = CNN().to(device)
    wall, test_log, train_log = [], [], []

    current_time = 0
    last_log = 0
    next_log = LOG_INTERVAL
    # for r in range(R):
    while current_time < MAX_TIME:

        # safety fix
        m_eff = min(m, len(clients))

        candidates = random.sample(clients, m_eff)

        losses = []
        W.eval()
        with torch.no_grad():
            for c in candidates:
                loader = DataLoader(Subset(trainset, c["indices"]), batch_size=256)
                total_loss, total = 0, 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    loss = F.cross_entropy(W(x), y, reduction="sum")
                    total_loss += loss.item()
                    total += y.size(0)
                losses.append(total_loss / total)

        losses = np.array(losses)
        order = np.argsort(-losses, kind="stable")
        topk_indices = order[:topK]#order[:k]

        for i in topk_indices:
            local = local_train(W, candidates[i])
            delta = model_diff(local, W)
            apply_update(W, delta, eta)

        # step_time = best_client["compute_time"]
        step_time = max(client["compute_time"] for client in clients)

        
        # === 4. Simulate time passing ===
        # for _ in range(step_time):
        #     current_time += 1

        #     if current_time >= next_log:
        #         wall.append(current_time)
        #         test_log.append(evaluate(W, test_loader))
        #         train_log.append(evaluate_train(W, clients))
        #         next_log += LOG_INTERVAL
                 
                # === 4. Simulate time passing ===
        

        for _ in range(step_time):
            if current_time >= MAX_TIME:
                break            
            wall.append(current_time)
            test_log.append(evaluate(W, test_loader))
            train_log.append(evaluate_train(W, clients))
            current_time += 1

        torch.cuda.empty_cache()

    return wall, test_log, train_log


def async_fl(clients, MAX_TIME, LOG_INTERVAL, eta=0.1, lam=0.01, topK=2):
    W = CNN().to(device)
    arrivals = defaultdict(list)
    wall, test_log, train_log = [], [], []
    current_time = 0
    last_log = 0
    next_log = LOG_INTERVAL


    # for t in range(T):
    while current_time < MAX_TIME:
        # same arrival process as FLANP
        # selected = sample_arrivals(clients, prob=0.05)
        selected = sample_arrivals(clients)

        for k in selected:
            # if random.random() < 0.05:
        # if random.random() < 0.3: #if random.random() < 0.1: for faster run
            # k = random.randrange(len(clients))
            c = clients[k]
            local = local_train(W, c)
            delta = model_diff(local, W)
            finish = current_time + c["compute_time"]
            arrivals[finish].append((k, delta, current_time))
            # KEY IDEA: keep only best updates
            arrivals[finish] = sorted(
                arrivals[finish],
                key=lambda x: clients[x[0]]["quality"],
                reverse=True
            )[:topK]
            del local

        if current_time in arrivals:
            updates = arrivals.pop(current_time)

            weights, deltas = [], []
            for k, delta, t0 in updates:
                staleness = current_time - t0
                w = clients[k]["quality"] * math.exp(-lam * staleness)
                weights.append(w)
                deltas.append(delta)

            # weights = torch.tensor(weights)
            # alphas = weights / weights.sum()
            weights = torch.tensor(weights)
            if weights.sum() > 0:
                alphas = weights / weights.sum()
            else:
                alphas = torch.ones_like(weights) / len(weights)

            for delta, alpha in zip(deltas, alphas):
                apply_update(W, delta, eta * alpha.item())

        wall.append(current_time)
        test_log.append(evaluate(W, test_loader))
        train_log.append(evaluate_train(W, clients))
        current_time += 1

        torch.cuda.empty_cache()

    return wall, test_log, train_log


# def flanp(clients, MAX_TIME, LOG_INTERVAL, eta=0.05, lam=0.01):
#     W = CNN().to(device)
#     arrivals = defaultdict(list)

#     wall, test_log, train_log = [0.0], [0.0], [0.0]
#     current_time, last_log = 0, 0
#     next_log = LOG_INTERVAL

#     while current_time < MAX_TIME:

#         # selected = sample_arrivals(clients, prob=0.05)
#         selected = sample_arrivals(clients)

#         for k in selected:
#             c = clients[k]
#             local = local_train(W, c)
#             delta = model_diff(local, W)

#             finish = current_time + c["compute_time"]
#             arrivals[finish].append((k, delta, current_time))
#             del local

#         if current_time in arrivals:
#             updates = arrivals.pop(current_time)

#             weights = torch.tensor([
#                 math.exp(-lam * (current_time - t0))
#                 for (_, _, t0) in updates
#             ])

#             # alphas = weights / weights.sum()
#             if weights.sum() > 0:
#                 alphas = weights / weights.sum()
#             else:
#                 alphas = torch.ones_like(weights) / len(weights)

#             for (_, delta, _), alpha in zip(updates, alphas):
#                apply_update(W, delta, eta * alpha.item())

#         wall.append(current_time)
#         test_log.append(evaluate(W, test_loader))
#         train_log.append(evaluate_train(W, clients))
#         current_time += 1

#     return wall, test_log, train_log

###########################################
# This below version of FLANP runs slower on CPU
#########################################3
# def flanp(clients, MAX_TIME, LOG_INTERVAL, eta=0.05, lam=0.01,
#           mu=0.1, init_m=2, max_m=20):

#     W = CNN().to(device)
#     arrivals = defaultdict(list)

#     wall, test_log, train_log = [], [], []
#     current_time = 0
#     next_log = LOG_INTERVAL

#     m_t = init_m  # adaptive number of clients

#     while current_time < MAX_TIME:

#         # =========================
#         # 1. Sample candidate clients
#         # =========================
#         m_eff = min(m_t, len(clients))
#         selected = random.sample(range(len(clients)), m_eff)

#         grad_norm_sq = 0.0
#         loss_list = []
#         updates_buffer = []

#         W.eval()

#         # =========================
#         # 2. Compute stats for condition
#         # =========================
#         with torch.no_grad():
#             for k in selected:
#                 c = clients[k]

#                 loader = DataLoader(
#                     Subset(trainset, c["indices"]),
#                     batch_size=128,
#                     shuffle=True
#                 )

#                 batch_losses = []

#                 for x, y in loader:
#                     x, y = x.to(device), y.to(device)
#                     logits = W(x)
#                     loss = F.cross_entropy(logits, y, reduction="none")
#                     batch_losses.extend(loss.detach().cpu().numpy())

#                 batch_losses = np.array(batch_losses)

#                 loss_list.extend(batch_losses.tolist())

#                 # local update for gradient proxy
#                 local = local_train(W, c)
#                 delta = model_diff(local, W)

#                 # compute squared norm
#                 norm_sq = sum((v**2).sum().item() for v in delta.values())
#                 grad_norm_sq += norm_sq

#                 updates_buffer.append((k, delta, current_time))
#                 del local

#         # =========================
#         # 3. Estimate variance
#         # =========================
#         if len(loss_list) > 1:
#             V_ns = np.var(loss_list)
#         else:
#             V_ns = 0.0

#         # =========================
#         # 4. FLANP condition
#         # =========================
#         if grad_norm_sq <= 2 * mu * V_ns:
#             # NOT enough signal -> increase clients
#             m_t = min(2 * m_t, max_m)
#             continue
#         else:
#             # good signal -> reset
#             m_t = init_m

#         # =========================
#         # 5. Schedule async updates
#         # =========================
#         for (k, delta, t0) in updates_buffer:
#             finish = current_time + clients[k]["compute_time"]
#             arrivals[finish].append((k, delta, t0))

#         # =========================
#         # 6. Apply arrived updates
#         # =========================
#         if current_time in arrivals:
#             updates = arrivals.pop(current_time)

#             weights = torch.tensor([
#                 math.exp(-lam * (current_time - t0))
#                 for (_, _, t0) in updates
#             ])

#             if weights.sum() > 0:
#                 alphas = weights / weights.sum()
#             else:
#                 alphas = torch.ones_like(weights) / len(weights)

#             for (_, delta, _), alpha in zip(updates, alphas):
#                 apply_update(W, delta, eta * alpha.item())

#         # =========================
#         # 7. Logging (FIXED LENGTH)
#         # =========================
#         if current_time >= next_log:
#             wall.append(current_time)
#             test_log.append(evaluate(W, test_loader))
#             train_log.append(evaluate_train(W, clients))
#             next_log += LOG_INTERVAL

#         current_time += 1
#         torch.cuda.empty_cache()

#     return wall, test_log, train_log

################################################
######The below version is a GPU efficient version of FLANP
#################################################
# def flanp(clients, MAX_TIME, LOG_INTERVAL,
#           eta=0.05, lam=0.01,
#           mu=0.1, init_m=2, max_m=20):

#     W = CNN().to(device)
#     arrivals = defaultdict(list)

#     wall, test_log, train_log = [], [], []
#     current_time = 0
#     next_log = LOG_INTERVAL

#     m_t = init_m

#     # =========================================================
#     # Pre-create loaders (CRITICAL SPEED FIX)
#     # =========================================================
#     for c in clients:
#         # c["loader"] = DataLoader(
#         #     Subset(trainset, c["indices"]),
#         #     batch_size=128,
#         #     shuffle=True,
#         #     pin_memory=True,
#         #     drop_last=True   # improves stability of variance estimate
#         # )
#         effective_bs = min(128, len(c["indices"]))

#         c["loader"] = DataLoader(
#             Subset(trainset, c["indices"]),
#             batch_size=effective_bs,
#             shuffle=True,
#             pin_memory=True,
#             drop_last=False
#         )
#         c["iter"] = iter(c["loader"])

#     # =========================================================
#     # MAIN LOOP
#     # =========================================================
#     while current_time < MAX_TIME:

#         # ---------------------------------
#         # 1. Sample clients
#         # ---------------------------------
#         ##### Earlier Version
        
#         # m_eff = min(m_t, len(clients))
#         # selected_ids = random.sample(range(len(clients)), m_eff)

#         ##### New Version with sample arrivals
#         available = sample_arrivals(clients)

#         if len(available) == 0:
#             current_time += 1
#             continue

#         m_eff = min(m_t, len(available))
#         selected_ids = random.sample(available, m_eff)

#         grad_norm_sq = 0.0
#         loss_tensors = []
#         updates_buffer = []

#         # ---------------------------------
#         # 2. FAST stats estimation
#         # ---------------------------------
#         for k in selected_ids:
#             c = clients[k]

#             # === Efficient batch fetch ===
#             try:
#                 x, y = next(c["iter"])
#             except StopIteration:
#                 c["iter"] = iter(c["loader"])
#                 x, y = next(c["iter"])

#             x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

#             # === Forward + backward ONCE ===
#             local_model = copy_model(W)
#             local_model.train()
#             local_model.zero_grad()

#             logits = local_model(x)
#             loss = F.cross_entropy(logits, y)
#             loss.backward()

#             # === Gradient norm ===
#             grad_norm_sq += sum(
#                 (p.grad.detach()**2).sum().item()
#                 for p in local_model.parameters() if p.grad is not None
#             )

#             # === Loss variance sample ===
#             # loss_tensors.append(loss.detach())
#             loss = F.cross_entropy(logits, y, reduction="none")
#             loss_tensors.append(loss.detach())

#             # === Local training (for update) ===
#             local = local_train(W, c)

#             # KEEP ON GPU
#             delta = {
#                 key: (local.state_dict()[key] - W.state_dict()[key])
#                 for key in W.state_dict()
#             }

#             updates_buffer.append((k, delta, current_time))
#             del local

#         # ---------------------------------
#         # 3. Variance estimate (GPU)
#         # ---------------------------------
#         if len(loss_tensors) > 1:
#             V_ns = torch.var(torch.cat(loss_tensors)).item()
#         else:
#             V_ns = 1e-6

#         V_ns = max(V_ns, 1e-6)

#         # ---------------------------------
#         # 4. FLANP condition
#         # ---------------------------------
#         if grad_norm_sq <= 2 * mu * V_ns:
#             # Not enough signal → increase clients
#             m_t = min(2 * m_t, max_m)

#         else:
#             # Good signal → reset
#             m_t = init_m

#             # schedule updates ONLY when condition passes
#             for (k, delta, t0) in updates_buffer:
#                 finish = current_time + clients[k]["compute_time"]
#                 arrivals[finish].append((k, delta, t0))

#         # ---------------------------------
#         # 5. Apply async updates
#         # ---------------------------------
#         if current_time in arrivals:
#             updates = arrivals.pop(current_time)

#             weights = torch.tensor([
#                 math.exp(-lam * (current_time - t0))
#                 for (_, _, t0) in updates
#             ], device=device)

#             if weights.sum() > 0:
#                 alphas = weights / weights.sum()
#             else:
#                 alphas = torch.ones_like(weights) / len(weights)

#             for (_, delta, _), alpha in zip(updates, alphas):
#                 for key in W.state_dict():
#                     W.state_dict()[key].add_(eta * alpha.item() * delta[key])

#         # ---------------------------------
#         # 6. Logging (FIXED LENGTH)
#         # ---------------------------------
#         # if current_time >= next_log:
#         wall.append(current_time)
#         test_log.append(evaluate(W, test_loader))
#         train_log.append(evaluate_train(W, clients))
#         # next_log += LOG_INTERVAL

#         # ---------------------------------
#         # 7. Advance time
#         # ---------------------------------
#         current_time += 1

#     return wall, test_log, train_log

# Reorganizing the code to fix the issue of unequal logging which creates an issue when computing mean at the end.
def flanp(clients, MAX_TIME, LOG_INTERVAL,
          eta=0.05, lam=0.01,
          mu=0.1, init_m=2, max_m=20):

    W = CNN().to(device)
    arrivals = defaultdict(list)

    wall, test_log, train_log = [], [], []

    current_time = 0
    m_t = init_m

    # =====================================================
    # Pre-create loaders
    # =====================================================
    for c in clients:

        effective_bs = min(128, len(c["indices"]))

        c["loader"] = DataLoader(
            Subset(trainset, c["indices"]),
            batch_size=effective_bs,
            shuffle=True,
            pin_memory=True,
            drop_last=False
        )

        c["iter"] = iter(c["loader"])

    # =====================================================
    # Main loop
    # =====================================================
    while current_time < MAX_TIME:

        # =================================================
        # 1. Apply updates arriving NOW
        # =================================================
        if current_time in arrivals:

            updates = arrivals.pop(current_time)

            weights = torch.tensor([
                math.exp(-lam * (current_time - t0))
                for (_, _, t0) in updates
            ], device=device)

            if weights.sum() > 0:
                alphas = weights / weights.sum()
            else:
                alphas = torch.ones_like(weights) / len(weights)

            for (_, delta, _), alpha in zip(updates, alphas):
                apply_update(W, delta, eta * alpha.item())

        # =================================================
        # 2. Determine available clients
        # =================================================
        available = sample_arrivals(clients)

        # =================================================
        # 3. No clients available
        # =================================================
        if len(available) == 0:

            wall.append(current_time)
            test_log.append(evaluate(W, test_loader))
            train_log.append(evaluate_train(W, clients))

            current_time += 1
            continue

        # =================================================
        # 4. Select m_t available clients
        # =================================================
        m_eff = min(m_t, len(available))
        selected_ids = random.sample(available, m_eff)

        grad_norm_sq = 0.0
        loss_tensors = []
        updates_buffer = []

        # =================================================
        # 5. Estimate FLANP statistics
        # =================================================
        for k in selected_ids:

            c = clients[k]

            try:
                x, y = next(c["iter"])

            except StopIteration:
                c["iter"] = iter(c["loader"])
                x, y = next(c["iter"])

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # -----------------------------------------
            # Gradient estimate
            # -----------------------------------------
            local_model = copy_model(W)

            local_model.train()
            local_model.zero_grad()

            logits = local_model(x)

            loss = F.cross_entropy(logits, y)

            loss.backward()

            grad_norm_sq += sum(
                (p.grad.detach() ** 2).sum().item()
                for p in local_model.parameters()
                if p.grad is not None
            )

            # -----------------------------------------
            # Variance estimate
            # -----------------------------------------
            sample_losses = F.cross_entropy(
                logits,
                y,
                reduction="none"
            )

            loss_tensors.append(sample_losses.detach())

            # -----------------------------------------
            # Local training
            # -----------------------------------------
            local = local_train(W, c)

            delta = {
                key: local.state_dict()[key] - W.state_dict()[key]
                for key in W.state_dict()
            }

            updates_buffer.append(
                (k, delta, current_time)
            )

            del local
            del local_model

        # =================================================
        # 6. Estimate V_ns
        # =================================================
        if len(loss_tensors) > 1:
            V_ns = torch.var(
                torch.cat(loss_tensors)
            ).item()
        else:
            V_ns = 1e-6

        V_ns = max(V_ns, 1e-6)

        # =================================================
        # 7. FLANP condition
        # =================================================
        if grad_norm_sq <= 2 * mu * V_ns:

            # increase sample size
            m_t = min(2 * m_t, max_m)

        else:

            # reset sample size
            m_t = init_m

            # schedule accepted updates
            for (k, delta, t0) in updates_buffer:

                finish_time = (
                    current_time
                    + clients[k]["compute_time"]
                )

                arrivals[finish_time].append(
                    (k, delta, t0)
                )

        # =================================================
        # 8. Logging EVERY timestep
        # =================================================
        wall.append(current_time)

        test_log.append(
            evaluate(W, test_loader)
        )

        train_log.append(
            evaluate_train(W, clients)
        )

        current_time += 1

    return wall, test_log, train_log

#########Another version of FLANP with updates to ensure that the delays in FLANP computation are taken into account, Uncomment when sending the updated code!
# def flanp(
#     clients,
#     MAX_TIME,
#     LOG_INTERVAL,
#     eta=0.03,
#     lam=0.05,
#     mu=0.1,
#     init_m=2,
#     max_m=20
# ):

#     W = CNN().to(device)

#     arrivals = defaultdict(list)

#     wall = []
#     test_log = []
#     train_log = []

#     current_time = 0

#     # =====================================================
#     # Adaptive client count
#     # =====================================================
#     m_t = init_m

#     # =====================================================
#     # Persistent loaders
#     # =====================================================
#     for c in clients:

#         c["loader"] = DataLoader(
#             Subset(trainset, c["indices"]),
#             batch_size=64,
#             shuffle=True,
#             pin_memory=True
#         )

#         c["iter"] = iter(c["loader"])

#     # =====================================================
#     # MAIN LOOP
#     # =====================================================
#     while current_time < MAX_TIME:

#         # =================================================
#         # 1. Sample clients
#         # =================================================
#         m_eff = min(m_t, len(clients))

#         selected_ids = random.sample(
#             range(len(clients)),
#             m_eff
#         )

#         updates_buffer = []

#         grad_norm_sq = 0.0

#         all_losses = []

#         # =================================================
#         # 2. Client-side estimation
#         # =================================================
#         for k in selected_ids:

#             c = clients[k]

#             # ---------------------------------------------
#             # SAFE batch fetch
#             # ---------------------------------------------
#             fetched = False

#             while not fetched:

#                 try:
#                     x, y = next(c["iter"])
#                     fetched = True

#                 except StopIteration:
#                     c["iter"] = iter(c["loader"])

#             x = x.to(device, non_blocking=True)
#             y = y.to(device, non_blocking=True)

#             # ---------------------------------------------
#             # Per-sample losses
#             # ---------------------------------------------
#             W.eval()

#             with torch.no_grad():

#                 logits = W(x)

#                 losses = F.cross_entropy(
#                     logits,
#                     y,
#                     reduction="none"
#                 )

#                 all_losses.append(losses)

#             # ---------------------------------------------
#             # Local training
#             # ---------------------------------------------
#             local = local_train(W, c)

#             delta = {}

#             local_norm_sq = 0.0

#             with torch.no_grad():

#                 for key in W.state_dict():

#                     d = (
#                         local.state_dict()[key]
#                         - W.state_dict()[key]
#                     )

#                     delta[key] = d

#                     local_norm_sq += (
#                         d.float().pow(2).sum().item()
#                     )

#             grad_norm_sq += local_norm_sq

#             finish = current_time + c["compute_time"]

#             updates_buffer.append(
#                 (k, delta, current_time, finish)
#             )

#             del local

#         # =================================================
#         # 3. Estimate V_ns
#         # =================================================
#         if len(all_losses) > 0:

#             all_losses = torch.cat(all_losses)

#             V_ns = torch.var(all_losses).item()

#         else:
#             V_ns = 1e-8

#         V_ns = max(V_ns, 1e-8)

#         # =================================================
#         # 4. FLANP condition
#         #
#         # ||grad||^2 > 2 mu V_ns
#         # =================================================
#         if grad_norm_sq > 2 * mu * V_ns:

#             # =============================================
#             # GOOD signal:
#             # accept updates
#             # =============================================
#             for item in updates_buffer:

#                 k, delta, t0, finish = item

#                 arrivals[finish].append(
#                     (k, delta, t0)
#                 )

#             # reset client count
#             m_t = init_m

#         else:

#             # =============================================
#             # BAD signal:
#             # double clients
#             # =============================================
#             m_t = min(2 * m_t, max_m)

#         # =================================================
#         # 5. Apply async updates
#         # =================================================
#         if current_time in arrivals:

#             updates = arrivals.pop(current_time)

#             weights = []

#             for (k, _, t0) in updates:

#                 staleness = current_time - t0

#                 w = math.exp(-lam * staleness)

#                 weights.append(w)

#             weights = torch.tensor(
#                 weights,
#                 device=device
#             )

#             weights = weights / (
#                 weights.sum() + 1e-12
#             )

#             for alpha, (_, delta, _) in zip(
#                 weights,
#                 updates
#             ):

#                 with torch.no_grad():

#                     for key in W.state_dict():

#                         W.state_dict()[key].add_(
#                             eta
#                             * alpha.item()
#                             * delta[key]
#                         )

#         # =================================================
#         # 6. Logging
#         # =================================================
#         wall.append(current_time)

#         test_log.append(
#             evaluate(W, test_loader)
#         )

#         train_log.append(
#             evaluate_train(W, clients)
#         )

#         # =================================================
#         # 7. Advance time
#         # =================================================
#         current_time += 1

#         torch.cuda.empty_cache()

#     return wall, test_log, train_log

############################################
##############End of FLANP #################
############################################

def unified(clients, MAX_TIME, LOG_INTERVAL, eta=0.05, lam=0.05, m=5):
    # Unified = PoC selection + async + quality weighting
    W = CNN().to(device)
    arrivals = defaultdict(list)
    wall, test_log, train_log = [0.0], [0.0], [0.0]
    current_time = 0
    last_log = 0
    next_log = LOG_INTERVAL

    # for t in range(T):
    while current_time < MAX_TIME:

        if random.random() < 0.05:
            candidates = random.sample(clients, m)

            losses = []
            W.eval()
            with torch.no_grad():
                for c in candidates:
                    loader = DataLoader(Subset(trainset, c["indices"]), batch_size=256)
                    total_loss, total = 0, 0
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        loss = F.cross_entropy(W(x), y, reduction="sum")
                        total_loss += loss.item()
                        total += y.size(0)
                    losses.append(total_loss / total)

            best = candidates[int(np.argmax(losses))]
            k = clients.index(best)

            local = local_train(W, best)
            delta = model_diff(local, W)
            finish = current_time + best["compute_time"]
            arrivals[finish].append((k, delta, current_time))
            del local

        if current_time in arrivals:
            updates = arrivals.pop(current_time)

            weights, deltas = [], []
            for k, delta, t0 in updates:
                staleness = current_time - t0
                w = clients[k]["quality"] * math.exp(-lam * staleness)
                weights.append(w)
                deltas.append(delta)

            # weights = torch.tensor(weights)
            # alphas = weights / weights.sum()
            weights = torch.tensor(weights)
            if weights.sum() > 0:
                alphas = weights / weights.sum()
            else:
                alphas = torch.ones_like(weights) / len(weights)            

            for delta, alpha in zip(deltas, alphas):
                apply_update(W, delta, eta * alpha.item())
            # current_time = t

        wall.append(current_time)
        test_log.append(evaluate(W, test_loader))
        train_log.append(evaluate_train(W, clients))
        current_time += 1

        torch.cuda.empty_cache()

    return wall, test_log, train_log

# ============================================================
# 8. Multi-Seed Experiment
# ============================================================

NUM_RUNS = 2

async_runs_test = []
sync_runs_test = []
async_runs_train = []
sync_runs_train = []
poc_runs_test = []
flanp_runs_test = []
poc_runs_train = []
flanp_runs_train = []
unified_runs_test = []
unified_runs_train = []


for seed in range(NUM_RUNS):
    print(f"\n==== Seed {seed} ====")
    set_seed(seed)

    # clients = create_non_iid_clients()
    client_indices = dirichlet_partition(trainset, 20, 0.05, min_size=10) # dirichlet_partition(trainset, 20, 0.3)  #Try dirichlet_partition(trainset, 10, 0.5) for faster run and reduce R=50 and T=800
    
    # client_ids = list(range(20))
    # random.shuffle(client_ids)

    # fast_bad = client_ids[:15]
    # slow_good = client_ids[15:]
    
    clients = []
    for k in range(20):

        # Strong compute heterogeneity
        if k < 15:#20 // 2:
            compute_time = np.random.randint(1, 3)#np.random.randint(2)      # fast clients
            quality = 0.5+0.5*np.random.rand()
            clients.append({
            "indices": client_indices[k],
            "compute_time": compute_time,
            # "quality": 0.3,       # noisy
            "lr": 0.05, # 0.02,           # small LR
            "local_epochs": 1,    # VERY IMPORTANT
            "quality": quality # 0.2        # remove bias advantage
            })
        else:
            compute_time = 8+np.random.randint(6)    # slow clients
            quality = 6+np.random.randint(9)
            clients.append({
            "indices": client_indices[k],
            "compute_time": compute_time,
            "lr": 0.01,           # small LR
            "local_epochs": 3,    # VERY IMPORTANT
            "quality": quality # 4.0        # remove bias advantage
            })


    MAX_TIME = 60#20000
    LOG_INTERVAL = 1#500

    lam = 0.05
    eta_quaad = 0.1
    eta_flanp = 0.03
    topK = 2

    wall_async, test_async, train_async = async_fl(clients, MAX_TIME, LOG_INTERVAL, eta=eta_quaad, lam=lam, topK=topK)
    wall_sync, test_sync, train_sync = synchronous_fedavg(clients, MAX_TIME, LOG_INTERVAL)
    wall_poc, test_poc, train_poc = power_of_choice(clients, MAX_TIME, LOG_INTERVAL)
    wall_flanp, test_flanp, train_flanp = flanp(clients, MAX_TIME, LOG_INTERVAL, eta=eta_flanp, lam=lam, mu=0.1, init_m=2, max_m=20)
    wall_unified, test_unified, train_unified = unified(clients, MAX_TIME, LOG_INTERVAL)

    async_runs_test.append(test_async)
    sync_runs_test.append(test_sync)
    async_runs_train.append(train_async)
    sync_runs_train.append(train_sync)
    poc_runs_test.append(test_poc)
    flanp_runs_test.append(test_flanp)
    poc_runs_train.append(train_poc)
    flanp_runs_train.append(train_flanp)
    unified_runs_test.append(test_unified)
    unified_runs_train.append(train_unified)

    os.makedirs("plots", exist_ok=True)

    plt.figure()
    plt.plot(wall_async, test_async, label="QUAAD")
    plt.plot(wall_sync, test_sync, label="DelayHetSampling")
    plt.plot(wall_poc, test_poc, label="Power-of-Choice")
    plt.plot(wall_flanp, test_flanp, label="FLANP")
    plt.plot(wall_unified, test_unified, label="Unified")
    plt.xlabel("Wall Clock")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/test_accuracy_{seed}.png", dpi=300)
    plt.show()

    plt.figure()
    plt.plot(wall_async, train_async, label="QUAAD")
    plt.plot(wall_sync, train_sync, label="DelayHetSampling")
    plt.plot(wall_poc, train_poc, label="Power-of-Choice")
    plt.plot(wall_flanp, train_flanp, label="FLANP")
    plt.plot(wall_unified, train_unified, label="Unified")
    plt.xlabel("Wall Clock")
    plt.ylabel("Train Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/train_accuracy_{seed}.png", dpi=300)
    plt.show()
    async_runs_test.append(test_async)
    sync_runs_test.append(test_sync)
    async_runs_train.append(train_async)
    sync_runs_train.append(train_sync)
    poc_runs_test.append(test_poc)
    flanp_runs_test.append(test_flanp)
    poc_runs_train.append(train_poc)
    flanp_runs_train.append(train_flanp)
    unified_runs_test.append(test_unified)
    unified_runs_train.append(train_unified)
    
    np.savez(f"results_mnist_fl_gpu_{seed}.npz",
    wall_async=wall_async,
    wall_sync=wall_sync,
    async_test=test_async,
    sync_test=test_sync,
    async_train=train_async,
    sync_train=train_sync,
    wall_poc=wall_poc,
    wall_flanp=wall_flanp,
    poc_test=test_poc,
    flanp_test=test_flanp,
    poc_train=train_poc,
    flanp_train=train_flanp,
    wall_unified=wall_unified,
    unified_test=test_unified,
    unified_train=train_unified)

    #np.savetxt("output.csv", arr, delimiter=",")

    

# Convert to numpy
async_mean_test = np.mean(async_runs_test, axis=0)
sync_mean_test = np.mean(sync_runs_test, axis=0)
async_mean_train = np.mean(async_runs_train, axis=0)
sync_mean_train = np.mean(sync_runs_train, axis=0)
poc_mean_test = np.mean(poc_runs_test, axis=0)
flanp_mean_test = np.mean(flanp_runs_test, axis=0)
poc_mean_train = np.mean(poc_runs_train, axis=0)
flanp_mean_train = np.mean(flanp_runs_train, axis=0)
unified_mean_test = np.mean(unified_runs_test, axis=0)
unified_mean_train = np.mean(unified_runs_train, axis=0)


# ============================================================
# 9. Plot + Save
# ============================================================

os.makedirs("plots", exist_ok=True)

plt.figure()
plt.plot(wall_async, async_mean_test, label="QUAAD")
plt.plot(wall_sync, sync_mean_test, label="DelayHetSampling")
plt.plot(wall_poc, poc_mean_test, label="Power-of-Choice")
plt.plot(wall_flanp, flanp_mean_test, label="FLANP")
plt.plot(wall_unified, unified_mean_test, label="Unified")
# plt.plot(wall_sync, sync_mean_test, label="Sync")
plt.xlabel("Wall Clock")
plt.ylabel("Test Accuracy")
plt.legend()
plt.grid()
plt.savefig("plots/test_accuracy.png", dpi=300)
plt.show()

plt.figure()
plt.plot(wall_async, async_mean_train, label="QUAAD")
plt.plot(wall_sync, sync_mean_train, label="DelayHetSampling")
plt.plot(wall_poc, poc_mean_train, label="Power-of-Choice")
plt.plot(wall_flanp, flanp_mean_train, label="FLANP")
plt.plot(wall_unified, unified_mean_train, label="Unified")
plt.xlabel("Wall Clock")
plt.ylabel("Train Accuracy")
plt.legend()
plt.grid()
plt.savefig("plots/train_accuracy.png", dpi=300)
plt.show()

np.savez(
    "results_mnist_fl_gpu.npz",
    wall_async=wall_async,
    wall_sync=wall_sync,
    async_test=async_mean_test,
    sync_test=sync_mean_test,
    async_train=async_mean_train,
    sync_train=sync_mean_train,
    wall_poc=wall_poc,
    wall_flanp=wall_flanp,
    poc_test=poc_mean_test,
    flanp_test=flanp_mean_test,
    poc_train=poc_mean_train,
    flanp_train=flanp_mean_train,
    wall_unified=wall_unified,
    unified_test=unified_mean_test,
    unified_train=unified_mean_train
)

print("\n✅ Experiment Complete – GPU Safe – Results Saved")