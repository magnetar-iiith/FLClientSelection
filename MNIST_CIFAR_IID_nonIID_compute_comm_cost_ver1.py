# ============================================================
# Federated Learning – Full Baseline Suite (GPU Safe)
# ============================================================

import os
import copy
import math
import random
import numpy as np
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# 2. Model
# ============================================================

class CNN(nn.Module):
    """
    Same conv architecture as before, but now adapts to whatever
    dataset it's handed: in_channels=1, input_size=28 for MNIST,
    in_channels=3, input_size=32 for CIFAR10.

    The flattened feature dimension (1024 for MNIST, 1600 for CIFAR10)
    is computed automatically with a dummy forward pass instead of
    being hardcoded, so we don't have to hand-derive it per dataset.
    """
    def __init__(self, in_channels=1, num_classes=10, input_size=28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            dummy = F.max_pool2d(self.conv1(dummy), 2)
            dummy = F.max_pool2d(self.conv2(dummy), 2)
            flat_dim = dummy.view(1, -1).size(1)

        self.fc1 = nn.Linear(flat_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def extract_features(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        return self.fc2(self.extract_features(x))


def make_model():
    """
    Builds a CNN matching the currently loaded dataset (set as globals
    by DataCreator) and moves it to `device`. Use this everywhere in
    place of the old bare `CNN().to(device)` so every algorithm picks
    up the right architecture automatically.
    """
    return CNN(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        input_size=IMG_SIZE
    ).to(device)

# ============================================================
# 3. Dataset
# ============================================================
def DataCreator(val_size=5000, whichdata="MNIST", batch_size=512):
    """
    whichdata: "MNIST" or "CIFAR10"

    Loads the requested dataset and sets a handful of globals that the
    rest of the pipeline (model construction, partitioning, training)
    relies on:
        - trainset, testset, valset, val_loader, test_loader  (as before)
        - IN_CHANNELS, IMG_SIZE, NUM_CLASSES  (used by make_model())
    """

    global full_trainset, num_train, perm, val_indices, train_indices
    global valset, val_loader, trainset, testset, test_loader
    global IN_CHANNELS, IMG_SIZE, NUM_CLASSES

    whichdata = whichdata.upper()

    if whichdata == "MNIST":
        IN_CHANNELS, IMG_SIZE, NUM_CLASSES = 1, 28, 10
        transform = transforms.ToTensor()

        full_trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=False, transform=transform
        )

    elif whichdata == "CIFAR10":
        IN_CHANNELS, IMG_SIZE, NUM_CLASSES = 3, 32, 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )
        ])

        full_trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=False, transform=transform
        )

    else:
        raise ValueError(
            f"Unknown whichdata='{whichdata}'. Use 'MNIST' or 'CIFAR10'."
        )

    # torchvision stores .targets as a tensor for MNIST but as a plain
    # Python list for CIFAR10. Standardize to a tensor for both so that
    # downstream code (dirichlet_partition, fancy indexing, etc.) works
    # identically regardless of which dataset was loaded.
    full_trainset.targets = torch.as_tensor(full_trainset.targets)
    trainset.targets = torch.as_tensor(trainset.targets)
    testset.targets = torch.as_tensor(testset.targets)

    # Testset data loader
    test_loader = DataLoader(testset, batch_size, shuffle=False)

    # Partition train set in validation set and training sett
    num_train = len(full_trainset)
    perm = np.random.permutation(num_train)
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    valset = torch.utils.data.Subset(full_trainset, val_indices)
    val_loader = DataLoader(
        valset,
        batch_size=256,
        shuffle=False
    )

    trainset.data = trainset.data[train_indices]
    trainset.targets = trainset.targets[train_indices]

def iid_partition(dataset, n_clients):
    """
    True IID partition: shuffle all indices and split into n_clients
    equal-ish chunks, ignoring labels entirely. Every client ends up
    with roughly the same class distribution as the full dataset.
    """
    n = len(dataset)
    perm_idx = np.random.permutation(n)
    splits = np.array_split(perm_idx, n_clients)
    return [split.tolist() for split in splits]

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
# 4. Utilities
# ============================================================

def copy_model(model):
    new_model = make_model()
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

def local_train(global_model, client):
    """
    Returns (model, grad_steps).
    grad_steps = number of optimizer.step() calls performed, i.e. the
    number of times parameters were updated / a gradient was computed
    for this client in this call. This is the base unit used to
    compute each algorithm's total computation cost.
    """
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

    grad_steps = 0
    for _ in range(client["local_epochs"]):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            grad_steps += 1

    return model, grad_steps

def sample_arrivals(clients, t):
    selected = []
    for k, c in enumerate(clients):
        # fast clients arrive frequently
        # if c["compute_time"] <= 10:
        #     prob = 0.2#0.2
        # else:
        #     prob = 0.05#0.01   # slow clients are RARE
        if c["availble"][t]==1:
            selected.append(k)
    return selected

def estimate_client_quality(model):
    #Estimate quality using validation loss. Larger loss -> higher quality
    # loader = DataLoader(Subset(valset,client["indices_val"]),batch_size=256, shuffle = False)
    model.eval()
    total_loss = 0
    total = 0
    with torch.no_grad():
        for x,y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y , reduction = "sum")
            total_loss += loss.item()
            total += y.size(0)
    return total_loss / total

def estimate_covariance(model, client):

    loader = DataLoader(
        Subset(trainset, client["indices"]),
        batch_size=min(128,len(client["indices"])),
        shuffle=True
    )

    x,_ = next(iter(loader))

    x = x.to(device)

    with torch.no_grad():

        Fmat = model.extract_features(x)

    Fmat = Fmat - Fmat.mean(0,keepdim=True)

    cov = (Fmat.T @ Fmat)/(Fmat.shape[0]-1)

    return cov

def Bp(p,Btilde,K):

    p=np.asarray(p)

    m=len(p)

    ones=np.ones(m)

    term1=p @ Btilde @ ones / m

    term2=p @ Btilde @ p / K

    return 2*(term1+term2)

def expected_delay(p,tau,K):

    order=np.argsort(tau)

    tau=np.array(tau)[order]

    p=np.array(p)[order]

    c=np.cumsum(p)

    delay=0.

    prev=0.

    for i in range(len(tau)):

        delay+=(c[i]**K-prev**K)*tau[i]

        prev=c[i]

    return delay

def update_eta(eta=1,curr_time=1):
    # if curr_time == 25:
    #     return eta/2
    
    return eta
# ============================================================
# 5. Algorithms
# ============================================================



###############################################################
# DelayHetSampling (Algorithm 3)
###############################################################

def delayhetsampling(
        clients,
        MAX_TIME,
        LOG_INTERVAL,
        eta=0.05,
        lam=0.05,
        K=5,
        OPT_INTERVAL=50,
        DELAY_ALPHA=0.1):

    W = make_model()
    W_cov = copy_model(W)
    W_cov.eval()
    arrivals = defaultdict(list)

    wall = []
    test_log = []
    train_log = []
    comp_cost_log = []
    comm_cost_log = []
    comp_cost = 0
    # compute_pstar() below queries every client's local covariance
    # ONCE, up front, before training starts -- that's one round of
    # every client sending information to the server, so we seed the
    # communication counter with it rather than silently dropping it.
    comm_cost = 0  # incremented to len(clients) right after compute_pstar() runs

    current_time = 0

    m = len(clients)

    def estimate_covariance(model, client):

        loader = DataLoader(
            Subset(trainset, client["indices"]),
            batch_size=min(128, len(client["indices"])),
            shuffle=True
        )

        x, _ = next(iter(loader))
        x = x.to(device)

        model.eval()

        with torch.no_grad():
            F = model.extract_features(x)

        F = F - F.mean(0, keepdim=True)

        cov = (F.T @ F) / max(F.shape[0]-1, 1)

        return cov.cpu()

    ############################################################
    # Build optimal sampling distribution
    ############################################################

    def compute_pstar():

        A_list = [
            estimate_covariance(W_cov, c)
            for c in clients
        ]

        A = torch.stack(A_list).mean(0)
        Ainv = torch.linalg.pinv(A)

        B = np.zeros((m, m))

        for i in range(m):
            for j in range(i, m):

                M = (A_list[i]-A_list[j]) @ Ainv

                val = torch.linalg.matrix_norm(
                    M,
                    ord=2
                ).item()

                B[i,j] = val
                B[j,i] = val

        Btilde = B**2

        tau = np.array([
            c["compute_time"]
            for c in clients
        ])

        ########################################################

        def Bp(p):

            p = np.asarray(p)

            ones = np.ones(len(p))

            term1 = p @ Btilde @ ones / len(p)

            term2 = p @ Btilde @ p / K

            return 2*(term1+term2)

        ########################################################

        def expected_delay(p):

            order = np.argsort(tau)

            tau_sorted = tau[order]

            p_sorted = np.asarray(p)[order]

            cdf = np.cumsum(p_sorted)

            delay = 0.

            prev = 0.

            for i in range(len(tau_sorted)):

                delay += (
                    cdf[i]**K - prev**K
                ) * tau_sorted[i]

                prev = cdf[i]

            return delay

        ########################################################

        def objective(p):

            b = Bp(p)

            if b >= 0.999:
                return 1e10

            return expected_delay(p)/(1-b)

        ########################################################

        p0 = np.ones(m)/m

        cons = [{
            "type":"eq",
            "fun":lambda p: np.sum(p)-1
        }]

        bounds = [(0,1)]*m

        res = minimize(
            objective,
            p0,
            method="trust-constr",
            bounds=bounds,
            constraints=cons
        )

        if res.success:
            p = np.maximum(res.x,0)
            p /= p.sum()
            return p

        return p0

    ############################################################

    pstar = compute_pstar()
    comm_cost += len(clients)  # every client reported local covariance stats once

    ############################################################
    # Main loop
    ############################################################
    print(datetime.now().strftime("%H:%M:%S"))
    while current_time < MAX_TIME:

        ########################################################
        # Re-optimize only periodically
        ########################################################

        # if current_time % OPT_INTERVAL == 0 and current_time > 0:
        #     pstar = compute_pstar()

        ########################################################
        # Available clients
        ########################################################

        available = sample_arrivals(clients,current_time)

        if len(available) > 0:

            probs = pstar[available]
            # probs = probs / probs.sum()
            if probs.sum() <= 1e-12:
                probs = np.ones(len(available))
                probs /= probs.sum()
            else:
                probs /= probs.sum()

            ####################################################
            # Algorithm 3 samples WITH replacement
            ####################################################

            chosen = np.random.choice(
                available,
                size=K,#min(K, len(available)),
                replace=True,
                p=probs
            )
            # chosen = np.unique(chosen)

            ####################################################

            for k in chosen:

                local, steps = local_train(W, clients[k])
                comp_cost += steps
                comm_cost += 1  # each draw is its own client upload, duplicates included

                delta = model_diff(local, W)

                finish = current_time + clients[k]["compute_time"]

                arrivals[finish].append(
                    (k, delta, current_time)
                )

                del local

        ########################################################
        # Aggregate
        ########################################################

        if current_time in arrivals:

            updates = arrivals.pop(current_time)

            weights = []

            # for k, delta, t0 in updates:

            #     observed = current_time - t0

            #     ################################################
            #     # Update expected delay estimate
            #     ################################################

            #     clients[k]["tau_est"] = (
            #         (1-DELAY_ALPHA)*clients[k]["tau_est"]
            #         + DELAY_ALPHA*observed
            #     )

            #     weights.append(
            #         np.exp(-lam*observed)
            #     )

            # weights = np.asarray(weights)

            # if weights.sum() == 0:
            #     weights = np.ones_like(weights)
            weights = np.ones(len(updates))

            weights /= weights.sum()

            alpha = 1.0 / len(updates)
            for (_, delta, _) in updates:

                apply_update(
                    W,
                    delta,
                    eta*alpha
                )

        ########################################################

        wall.append(current_time)

        test_log.append(
            evaluate(W, test_loader)
        )

        train_log.append(
            evaluate_train(W, clients)
        )

        comp_cost_log.append(comp_cost)
        comm_cost_log.append(comm_cost)

        current_time += 1

    return wall, test_log, train_log, comp_cost_log, comm_cost_log

def synchronous_fedavg(clients, MAX_TIME, LOG_INTERVAL, eta=1.0):
    W = make_model()
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
            local, _ = local_train(W, c)
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
    W = make_model()
    wall, test_log, train_log = [], [], []
    comp_cost_log, comm_cost_log = [], []
    comp_cost, comm_cost = 0, 0

    current_time = 0
    last_log = 0
    next_log = LOG_INTERVAL
    # for r in range(R):
    print(datetime.now().strftime("%H:%M:%S"))
    while current_time < MAX_TIME:
        if current_time%100 == 0 or current_time%15 == 0:
            print(f"\n Current time in PoC: {current_time}")
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
        # every probed candidate reports its loss back to the server --
        # a communication event, but not a gradient/parameter update.
        comm_cost += m_eff

        losses = np.array(losses)
        order = np.argsort(-losses, kind="stable")
        topk_indices = order[:topK]#order[:k]

        for i in topk_indices:
            local, steps = local_train(W, candidates[i])
            comp_cost += steps
            comm_cost += 1  # the trained delta is a second, separate upload
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
            comp_cost_log.append(comp_cost)
            comm_cost_log.append(comm_cost)
            current_time += 1

        torch.cuda.empty_cache()

    return wall, test_log, train_log, comp_cost_log, comm_cost_log

def generalized_fedavg(
        clients,
        MAX_TIME,
        LOG_INTERVAL,
        gamma=0.01,          # local learning rate
        eta=2,             # amplification factor
        P=5,                 # amplification interval
        participation_prob=1.0, 
        topK=2
    ):
    wall, test_log, train_log = [], [], []
    comp_cost_log, comm_cost_log = [], []
   
    return wall, test_log, train_log, comp_cost_log, comm_cost_log

def async_fl_noexp(clients, MAX_TIME, LOG_INTERVAL, eta=0.1, lam=0.01, topK=2):
    wall, test_log, train_log = [], [], []
    comp_cost_log, comm_cost_log = [], []
    return wall, test_log, train_log, comp_cost_log, comm_cost_log
    # current_time = 0
    # last_log = 0
    # next_log = LOG_INTERVAL
    # num_clients = len(clients)

    # # delta = [] * num_clients
    # # for t in range(T):
    # print(datetime.now().strftime("%H:%M:%S"))
    # while current_time < MAX_TIME:
    #     # same arrival process as FLANP
    #     # selected = sample_arrivals(clients, prob=0.05)
    #     if current_time%20 == 0:
    #         print(f"\n Current time in QUAD: {current_time}")

    #     available_clients = sample_arrivals(clients,current_time)
    #     if len(available_clients) >0:

    #         qualities = {}
    #         #for k in available_clients:
                
    #         weights, deltas = [], []

    #         for k in available_clients:
    #             # if k in selected:
    #                 c = clients[k]
    #                 local = local_train(W, c)
    #                 qualities[k] = 1.0 / (1.0 + estimate_client_quality(local))
    #                 # qualities[k] = (clients[k]["est_quality"]*clients[k]["num_selected"] +qualities[k] )/(clients[k]["num_selected"]+1)
    #                 # clients[k]["num_selected"] +=1
    #                 # clients[k]["est_quality"] = qualities[k]
    #                 delta = model_diff(local, W)
    #                 # del local
    #                 w = qualities[k]#* math.exp(-lam * clients[k]["compute_time"])
    #                 weights.append(w)
    #                 deltas.append(delta)
    #                 del local

    #         selected = sorted(
    #             available_clients,
    #             key=lambda k: qualities[k],
    #             reverse=True
    #         )
    #         selected =  selected[:topK]#[0:min(len(selected),topK)]
    #         weights = [w if i in selected else 0.0 for i, w in enumerate(weights)]
    #         weights = torch.tensor(weights, dtype=torch.float32)
    #         if weights.sum() > 0:
    #             alphas = weights / weights.sum()
    #         else:
    #             alphas = torch.ones_like(weights) / len(weights)

    #         eta =  update_eta(eta, current_time)
    #         for delta, alpha in zip(deltas, alphas):
    #             apply_update(W, delta, eta * alpha.item())

        
    #     wall.append(current_time)
    #     test_log.append(evaluate(W, test_loader))
    #     train_log.append(evaluate_train(W, clients))
    #     current_time += 1

    #     torch.cuda.empty_cache()

    # return wall, test_log, train_log

def async_fl(clients, MAX_TIME, LOG_INTERVAL, eta=0.1, lam=0.01, topK=2):
    W = make_model()
    arrivals = defaultdict(list)
    wall, test_log, train_log = [], [], []
    comp_cost_log, comm_cost_log = [], []
    comp_cost, comm_cost = 0, 0
    current_time = 0
    last_log = 0
    next_log = LOG_INTERVAL
    num_clients = len(clients)

    # delta = [] * num_clients
    # for t in range(T):
    print(datetime.now().strftime("%H:%M:%S"))
    while current_time < MAX_TIME:
        # same arrival process as FLANP
        # selected = sample_arrivals(clients, prob=0.05)
        if current_time%20 == 0:
            print(f"\n Current time in QUAD: {current_time}")

        available_clients = sample_arrivals(clients,current_time)
        if len(available_clients) >0:

            qualities = {}
            #for k in available_clients:
                
            weights, deltas = [], []

            for k in available_clients:
                # if k in selected:
                    c = clients[k]
                    local, steps = local_train(W, c)
                    # every available client trains locally and uploads
                    # its result to the server, whether or not it ends
                    # up in the topK selected for aggregation.
                    comp_cost += steps
                    comm_cost += 1
                    qualities[k] = 1.0 / (1.0 + estimate_client_quality(local))
                    qualities[k] = (clients[k]["est_quality"]*clients[k]["num_selected"] +qualities[k] )/(clients[k]["num_selected"]+1)
                    clients[k]["num_selected"] +=1
                    clients[k]["est_quality"] = qualities[k]
                    delta = model_diff(local, W)
                    # del local
                    w = qualities[k]* math.exp(-lam * clients[k]["compute_time"])
                    weights.append(w)
                    deltas.append(delta)
                    del local

            selected = sorted(
                available_clients,
                key=lambda k: qualities[k],
                reverse=True
            )
            selected =  selected[:topK]#[0:min(len(selected),topK)]
            weights = [w if i in selected else 0.0 for i, w in enumerate(weights)]
            weights = torch.tensor(weights, dtype=torch.float32)
            if weights.sum() > 0:
                alphas = weights / weights.sum()
            else:
                alphas = torch.ones_like(weights) / len(weights)

            eta =  update_eta(eta, current_time)
            for delta, alpha in zip(deltas, alphas):
                apply_update(W, delta, eta * alpha.item())

        wall.append(current_time)
        test_log.append(evaluate(W, test_loader))
        train_log.append(evaluate_train(W, clients))
        comp_cost_log.append(comp_cost)
        comm_cost_log.append(comm_cost)
        current_time += 1

        torch.cuda.empty_cache()

    return wall, test_log, train_log, comp_cost_log, comm_cost_log

# Reorganizing the code to fix the issue of unequal logging which creates an issue when computing mean at the end.
def flanp(clients, MAX_TIME, LOG_INTERVAL,
          eta=0.05, lam=0.01,
          mu=0.1, init_m=2, max_m=20):

    W = make_model()
    arrivals = defaultdict(list)

    wall, test_log, train_log = [], [], []
    comp_cost_log, comm_cost_log = [], []
    comp_cost, comm_cost = 0, 0

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

    print(datetime.now().strftime("%H:%M:%S"))
    # =====================================================
    # Main loop
    # =====================================================
    while current_time < MAX_TIME:
        if current_time%100 == 0:
            print(f"\n Current time in FLANP: {current_time}")
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
        available = sample_arrivals(clients,current_time)

        # =================================================
        # 3. No clients available
        # =================================================
        if len(available) == 0:

            wall.append(current_time)
            test_log.append(evaluate(W, test_loader))
            train_log.append(evaluate_train(W, clients))
            comp_cost_log.append(comp_cost)
            comm_cost_log.append(comm_cost)

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

            # this backward pass computes a gradient (used only to
            # estimate grad_norm_sq for FLANP's sampling decision) but
            # never calls optimizer.step() -- still counts as "compute
            # the gradient" under our cost definition.
            comp_cost += 1

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
            local, steps = local_train(W, c)
            comp_cost += steps
            # the gradient-probe result and the trained delta both come
            # from this same client in this same round -- treated as
            # one bundled upload rather than two separate messages.
            comm_cost += 1

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

        comp_cost_log.append(comp_cost)
        comm_cost_log.append(comm_cost)

        current_time += 1

    return wall, test_log, train_log, comp_cost_log, comm_cost_log

def unified(clients,
            MAX_TIME,
            LOG_INTERVAL,
            eta=0.05,
            lam=0.05,
            m=5):
    wall = []
    test_log = []
    train_log = []

    return wall, test_log, train_log

# ============================================================
# 6. Multi-Seed Experiment
# ============================================================

def run_all_algos(  NUM_RUNS        = 4,
                    NUM_CLIENTS     = 30,
                    MAX_TIME        = 30 , 
                    topK_factor     = 0.1,
                    partition       = "nonIID",
                    myData          = "MNIST",
                    qualityRatio    = 4.0,
                    dirichlet_alpha = 0.05):
    """
    partition : "IID" or "nonIID"
        "IID"    -> data split uniformly at random across clients
                    (iid_partition)
        "nonIID" -> data split via a Dirichlet distribution over labels
                    (dirichlet_partition), producing label-skewed clients
    myData : "MNIST" or "CIFAR10"
    """
    partition = partition.upper()
    if partition not in ("IID", "NONIID"):
        raise ValueError(f"Unknown partition='{partition}'. Use 'IID' or 'nonIID'.")

    
    # storage of stats
    async_runs_test             = []
    sync_runs_test              = []
    async_runs_train            = []
    sync_runs_train             = []
    poc_runs_test               = []
    flanp_runs_test             = []
    poc_runs_train              = []
    flanp_runs_train            = []
    unified_runs_test           = []
    unified_runs_train          = []
    delayhetsampling_runs_test  = []
    delayhetsampling_runs_train = []
    generalized_runs_test       = []
    generalized_runs_train      = []

    # per-seed computation/communication cost curves (cumulative over
    # wall-clock time, one list of values per seed -- same shape as
    # the accuracy logs above)
    async_runs_comp             = []
    async_runs_comm             = []
    poc_runs_comp                = []
    poc_runs_comm                = []
    flanp_runs_comp               = []
    flanp_runs_comm               = []
    unified_runs_comp            = []
    unified_runs_comm            = []
    delayhetsampling_runs_comp  = []
    delayhetsampling_runs_comm  = []
    generalized_runs_comp       = []
    generalized_runs_comm       = []

    # Create data sets
    DataCreator(val_size=5000, whichdata=myData, batch_size=512)

    # Parameter setting
    LOG_INTERVAL                = 1
    lam                         = 0.05
    eta_quaad                   = 2
    eta_flanp                   = 0.3
    topK                        = int(topK_factor*NUM_CLIENTS)
    num_classes                 = NUM_CLASSES  # set by DataCreator for the loaded dataset
    labels                      = np.array(trainset.targets)

    for seed in range(NUM_RUNS):
        print(f"\n==== Seed {seed} ====")
        set_seed(seed)
        print(datetime.now().strftime("%H:%M:%S"))

        if partition == "IID":
            client_indices = iid_partition(trainset, NUM_CLIENTS)
        else:  # "NONIID"
            client_indices = dirichlet_partition(trainset, NUM_CLIENTS, dirichlet_alpha, min_size=10)

        clients         = []      

        for k in range(NUM_CLIENTS):
            isGood = np.random.rand()
            # Strong compute heterogeneity
            if isGood < 0.8:#20 // 2:
                compute_time = np.random.randint(1, 3)#np.random.randint(2)      # fast clients
                quality = 0.99+0.01*np.random.rand()
                clients.append({
                "indices": client_indices[k],
                "compute_time": compute_time,
                # "quality": 0.3,       # noisy
                "lr": 0.01, # 0.02,           # small LR
                "local_epochs": 1,    # VERY IMPORTANT
                "quality": quality, # 0.2        # remove bias advantage
                "availble": np.zeros(MAX_TIME),
                "est_quality": 0,
                "num_selected": 0
                })
                # introducing errors based on the quality
                idx = client_indices[k]
                for i in idx:
                    z = np.random.rand()
                    if z > quality:
                        labels[i] = np.random.randint(0,num_classes-1)
                    else:
                        labels[i] = trainset.targets[i]
                # setting arrival time intervals
                for t in range(compute_time,MAX_TIME,compute_time):
                        clients[k]["availble"][t] = 1

            else:
                compute_time = int(2*qualityRatio)+np.random.randint(6)    # slow clients
                quality = 1#0.9+0.1*np.random.rand()
                clients.append({
                "indices": client_indices[k],
                "compute_time": compute_time,
                "lr": 0.01,             # small LR
                "local_epochs": int(qualityRatio),      # VERY IMPORTANT
                "quality": quality,      # 4.0        # remove bias advantage
                "availble": np.zeros(MAX_TIME),
                "est_quality": 0,
                "num_selected": 0    
                })

                idx = client_indices[k]
   
                # setting arrival time intervals
                for t in range(compute_time,MAX_TIME,compute_time):
                        clients[k]["availble"][t] = 1

        # setting the targets based on the quality
        trainset.targets = torch.tensor(labels)


        wall_generalized, test_generalized, train_generalized, comp_generalized, comm_generalized = generalized_fedavg(clients, MAX_TIME, LOG_INTERVAL, gamma=0.01, eta=2.0, P=15, participation_prob=1.0,topK=topK)
        wall_async, test_async, train_async, comp_async, comm_async                               = async_fl(clients, MAX_TIME, LOG_INTERVAL, eta=eta_quaad, lam=lam, topK=topK)
        wall_poc, test_poc, train_poc, comp_poc, comm_poc                                          = power_of_choice(clients, MAX_TIME, LOG_INTERVAL)
        wall_flanp, test_flanp, train_flanp, comp_flanp, comm_flanp                                = flanp(clients, MAX_TIME, LOG_INTERVAL, eta=eta_flanp, lam=lam, mu=0.1, init_m=2, max_m=20)
        wall_unified, test_unified, train_unified, comp_unified, comm_unified                      = async_fl_noexp(clients, MAX_TIME, LOG_INTERVAL, eta=eta_quaad, lam=lam, topK=topK)
        wall_delayhetsampling, test_delayhetsampling, train_delayhetsampling, comp_delayhetsampling, comm_delayhetsampling = delayhetsampling(clients, MAX_TIME, LOG_INTERVAL, eta=0.05, lam=0.05, K=topK)
    
        async_runs_test.append(test_async)
        async_runs_train.append(train_async)
        async_runs_comp.append(comp_async)
        async_runs_comm.append(comm_async)
        poc_runs_test.append(test_poc)
        flanp_runs_test.append(test_flanp)
        poc_runs_train.append(train_poc)
        flanp_runs_train.append(train_flanp)
        poc_runs_comp.append(comp_poc)
        poc_runs_comm.append(comm_poc)
        flanp_runs_comp.append(comp_flanp)
        flanp_runs_comm.append(comm_flanp)
        unified_runs_test.append(test_unified)
        unified_runs_train.append(train_unified)
        unified_runs_comp.append(comp_unified)
        unified_runs_comm.append(comm_unified)
        delayhetsampling_runs_test.append(test_delayhetsampling)
        delayhetsampling_runs_train.append(train_delayhetsampling)
        delayhetsampling_runs_comp.append(comp_delayhetsampling)
        delayhetsampling_runs_comm.append(comm_delayhetsampling)
        generalized_runs_test.append(test_generalized)
        generalized_runs_train.append(train_generalized)
        generalized_runs_comp.append(comp_generalized)
        generalized_runs_comm.append(comm_generalized)

        dir_name = myData+f"_n{NUM_CLIENTS}_q{int(qualityRatio)}"
        dir_name = dir_name + ("_IID" if partition == "IID" else "_NonIID")

        print(dir_name)
        os.makedirs(dir_name, exist_ok=True)

        plt.figure()
        plt.plot(wall_async, test_async, label="QUAAD")
        plt.plot(wall_poc, test_poc, label="Power-of-Choice")
        plt.plot(wall_flanp, test_flanp, label="FLANP")
        plt.plot(wall_unified, test_unified, label="Unified")
        plt.plot(wall_delayhetsampling, test_delayhetsampling, label="DelayHetSampling")
        plt.plot(wall_generalized, test_generalized, label="Generalized FedAvg")
        plt.xlabel("Wall Clock")
        plt.ylabel("Test Accuracy")
        plt.legend()
        plt.grid()
        plt.savefig(f"{dir_name}/test_accuracy_{seed}.png", dpi=300)
        plt.show()
        plt.close()

        plt.figure()
        plt.plot(wall_async, train_async, label="QUAAD")
        plt.plot(wall_poc, train_poc, label="Power-of-Choice")
        plt.plot(wall_flanp, train_flanp, label="FLANP")
        plt.plot(wall_unified, train_unified, label="Unified")
        plt.plot(wall_delayhetsampling, train_delayhetsampling, label="DelayHetSampling")
        plt.plot(wall_generalized, train_generalized, label="Generalized FedAvg")
        plt.xlabel("Wall Clock")
        plt.ylabel("Train Accuracy")
        plt.legend()
        plt.grid()
        plt.savefig(f"{dir_name}/train_accuracy_{seed}.png", dpi=300)
        plt.show()
        plt.close()

        np.savez(f"{dir_name}/results_fl_gpu_{seed}.npz",
        wall_async=wall_async,
        async_test=test_async,
        async_train=train_async,
        async_comp=comp_async,
        async_comm=comm_async,
        wall_poc=wall_poc,
        wall_flanp=wall_flanp,
        poc_test=test_poc,
        flanp_test=test_flanp,
        poc_train=train_poc,
        flanp_train=train_flanp,
        poc_comp=comp_poc,
        poc_comm=comm_poc,
        flanp_comp=comp_flanp,
        flanp_comm=comm_flanp,
        wall_unified=wall_unified,
        unified_test=test_unified,
        unified_train=train_unified,
        unified_comp=comp_unified,
        unified_comm=comm_unified,
        delayhetsampling_test=test_delayhetsampling,
        delayhetsampling_train=train_delayhetsampling,
        delayhetsampling_comp=comp_delayhetsampling,
        delayhetsampling_comm=comm_delayhetsampling,
        generalized_test=test_generalized,
        generalized_train=train_generalized,
        generalized_comp=comp_generalized,
        generalized_comm=comm_generalized)

        

    # Convert to numpy
    async_mean_test = np.mean(async_runs_test, axis=0)
    async_mean_train = np.mean(async_runs_train, axis=0)
    poc_mean_test = np.mean(poc_runs_test, axis=0)
    flanp_mean_test = np.mean(flanp_runs_test, axis=0)
    poc_mean_train = np.mean(poc_runs_train, axis=0)
    flanp_mean_train = np.mean(flanp_runs_train, axis=0)
    unified_mean_test = np.mean(unified_runs_test, axis=0)
    unified_mean_train = np.mean(unified_runs_train, axis=0)
    delayhetsampling_mean_test = np.mean(delayhetsampling_runs_test, axis=0)
    delayhetsampling_mean_train = np.mean(delayhetsampling_runs_train, axis=0)
    generalized_mean_test = np.mean(generalized_runs_test, axis=0)
    generalized_mean_train = np.mean(generalized_runs_train, axis=0)

    # Mean cumulative computation / communication cost curves (same
    # length/cadence as the accuracy curves above, so they can be
    # plotted against wall-clock time or used as an x-axis for accuracy)
    async_mean_comp = np.mean(async_runs_comp, axis=0)
    async_mean_comm = np.mean(async_runs_comm, axis=0)
    poc_mean_comp = np.mean(poc_runs_comp, axis=0)
    poc_mean_comm = np.mean(poc_runs_comm, axis=0)
    flanp_mean_comp = np.mean(flanp_runs_comp, axis=0)
    flanp_mean_comm = np.mean(flanp_runs_comm, axis=0)
    unified_mean_comp = np.mean(unified_runs_comp, axis=0)
    unified_mean_comm = np.mean(unified_runs_comm, axis=0)
    delayhetsampling_mean_comp = np.mean(delayhetsampling_runs_comp, axis=0)
    delayhetsampling_mean_comm = np.mean(delayhetsampling_runs_comm, axis=0)
    generalized_mean_comp = np.mean(generalized_runs_comp, axis=0)
    generalized_mean_comm = np.mean(generalized_runs_comm, axis=0)


    # ============================================================
    # 9. Plot + Save
    # ============================================================

    os.makedirs(dir_name, exist_ok=True)

    plt.figure()
    plt.plot(wall_async, async_mean_test, label="QUAAD")
    plt.plot(wall_poc, poc_mean_test, label="Power-of-Choice")
    plt.plot(wall_flanp, flanp_mean_test, label="FLANP")
    plt.plot(wall_unified, unified_mean_test, label="Unified")
    plt.plot(wall_delayhetsampling, delayhetsampling_mean_test, label="DelayHetSampling")
    plt.plot(wall_generalized, generalized_mean_test, label="Generalized FedAvg")
    plt.xlabel("Wall Clock")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f"{dir_name}/test_accuracy.png", dpi=300)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(wall_async, async_mean_train, label="QUAAD")
    plt.plot(wall_poc, poc_mean_train, label="Power-of-Choice")
    plt.plot(wall_flanp, flanp_mean_train, label="FLANP")
    plt.plot(wall_unified, unified_mean_train, label="Unified")
    plt.plot(wall_delayhetsampling, delayhetsampling_mean_train, label="DelayHetSampling")
    plt.plot(wall_generalized, generalized_mean_train, label="Generalized FedAvg")
    plt.xlabel("Wall Clock")
    plt.ylabel("Train Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f"{dir_name}/train_accuracy.png", dpi=300)
    plt.show()
    plt.close()

    # ---- Computation cost vs wall clock ----
    plt.figure()
    plt.plot(wall_async, async_mean_comp, label="QUAAD")
    plt.plot(wall_poc, poc_mean_comp, label="Power-of-Choice")
    plt.plot(wall_flanp, flanp_mean_comp, label="FLANP")
    plt.plot(wall_unified, unified_mean_comp, label="Unified")
    plt.plot(wall_delayhetsampling, delayhetsampling_mean_comp, label="DelayHetSampling")
    plt.plot(wall_generalized, generalized_mean_comp, label="Generalized FedAvg")
    plt.xlabel("Wall Clock")
    plt.ylabel("Cumulative Computation Cost\n(# gradient / parameter updates)")
    plt.legend()
    plt.grid()
    plt.savefig(f"{dir_name}/computation_cost.png", dpi=300)
    plt.show()
    plt.close()

    # ---- Communication cost vs wall clock ----
    plt.figure()
    plt.plot(wall_async, async_mean_comm, label="QUAAD")
    plt.plot(wall_poc, poc_mean_comm, label="Power-of-Choice")
    plt.plot(wall_flanp, flanp_mean_comm, label="FLANP")
    plt.plot(wall_unified, unified_mean_comm, label="Unified")
    plt.plot(wall_delayhetsampling, delayhetsampling_mean_comm, label="DelayHetSampling")
    plt.plot(wall_generalized, generalized_mean_comm, label="Generalized FedAvg")
    plt.xlabel("Wall Clock")
    plt.ylabel("Cumulative Communication Cost\n(# client\u2192server uploads)")
    plt.legend()
    plt.grid()
    plt.savefig(f"{dir_name}/communication_cost.png", dpi=300)
    plt.show()
    plt.close()

    # ---- Test accuracy vs computation cost (efficiency view) ----
    plt.figure()
    plt.plot(async_mean_comp, async_mean_test, label="QUAAD")
    plt.plot(poc_mean_comp, poc_mean_test, label="Power-of-Choice")
    plt.plot(flanp_mean_comp, flanp_mean_test, label="FLANP")
    plt.plot(unified_mean_comp, unified_mean_test, label="Unified")
    plt.plot(delayhetsampling_mean_comp, delayhetsampling_mean_test, label="DelayHetSampling")
    plt.plot(generalized_mean_comp, generalized_mean_test, label="Generalized FedAvg")
    plt.xlabel("Cumulative Computation Cost")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f"{dir_name}/test_accuracy_vs_computation_cost.png", dpi=300)
    plt.show()
    plt.close()

    # ---- Test accuracy vs communication cost (efficiency view) ----
    plt.figure()
    plt.plot(async_mean_comm, async_mean_test, label="QUAAD")
    plt.plot(poc_mean_comm, poc_mean_test, label="Power-of-Choice")
    plt.plot(flanp_mean_comm, flanp_mean_test, label="FLANP")
    plt.plot(unified_mean_comm, unified_mean_test, label="Unified")
    plt.plot(delayhetsampling_mean_comm, delayhetsampling_mean_test, label="DelayHetSampling")
    plt.plot(generalized_mean_comm, generalized_mean_test, label="Generalized FedAvg")
    plt.xlabel("Cumulative Communication Cost")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f"{dir_name}/test_accuracy_vs_communication_cost.png", dpi=300)
    plt.show()
    plt.close()

    np.savez(
        f"{dir_name}/results_fl_gpu.npz",
        wall_async              =wall_async,
        async_test              =async_mean_test,
        async_train             =async_mean_train,
        async_comp              =async_mean_comp,
        async_comm              =async_mean_comm,
        wall_poc                =wall_poc,
        wall_flanp              =wall_flanp,
        poc_test                =poc_mean_test,
        flanp_test              =flanp_mean_test,
        poc_train               =poc_mean_train,
        flanp_train             =flanp_mean_train,
        poc_comp                =poc_mean_comp,
        poc_comm                =poc_mean_comm,
        flanp_comp              =flanp_mean_comp,
        flanp_comm              =flanp_mean_comm,
        wall_unified            =wall_unified,
        unified_test            =unified_mean_test,
        unified_train           =unified_mean_train,
        unified_comp            =unified_mean_comp,
        unified_comm            =unified_mean_comm,
        delayhetsampling_test   =delayhetsampling_mean_test,
        delayhetsampling_train  =delayhetsampling_mean_train,
        delayhetsampling_comp   =delayhetsampling_mean_comp,
        delayhetsampling_comm   =delayhetsampling_mean_comm,
        generalized_test   =generalized_mean_test,
        generalized_train  =generalized_mean_train,
        generalized_comp   =generalized_mean_comp,
        generalized_comm   =generalized_mean_comm
    )

    print("\n✅ Experiment Complete – GPU Safe – Results Saved")

    # ------------------------------------------------------------
    # Final cost summary table (total cost incurred by MAX_TIME,
    # mean ± std across seeds) -- a quick readable comparison
    # alongside the saved plots.
    # ------------------------------------------------------------
    def _final_cost_stats(runs):
        finals = [r[-1] for r in runs if len(r) > 0]
        if len(finals) == 0:
            return float("nan"), float("nan")
        return float(np.mean(finals)), float(np.std(finals))

    cost_table = {
        "QUAAD":              (async_runs_comp, async_runs_comm),
        "Power-of-Choice":    (poc_runs_comp, poc_runs_comm),
        "FLANP":              (flanp_runs_comp, flanp_runs_comm),
        "Unified":            (unified_runs_comp, unified_runs_comm),
        "DelayHetSampling":   (delayhetsampling_runs_comp, delayhetsampling_runs_comm),
        "Generalized FedAvg": (generalized_runs_comp, generalized_runs_comm),
    }

    print(f"\nFinal cost @ MAX_TIME={MAX_TIME} (mean ± std over {NUM_RUNS} seeds)")
    print(f"{'Algorithm':<20}{'Computation Cost':<26}{'Communication Cost':<26}")
    for name, (comp_runs, comm_runs) in cost_table.items():
        comp_mean, comp_std = _final_cost_stats(comp_runs)
        comm_mean, comm_std = _final_cost_stats(comm_runs)
        print(f"{name:<20}{comp_mean:>10.1f} ± {comp_std:<10.1f}{comm_mean:>10.1f} ± {comm_std:<10.1f}")
    # Package everything up and return it so callers (e.g. a sweep
    # over NUM_CLIENTS) can pull out summary numbers without having
    # to re-parse the saved .npz files.
    #   test_mean / train_mean : accuracy averaged over NUM_RUNS seeds
    #   test_runs / train_runs : the raw per-seed logs (one list per
    #                            seed) so std / error bars can be
    #                            computed later
    # Note: "Unified" and "Generalized FedAvg" are currently stubbed
    # out in this codebase (their bodies just return empty lists), so
    # their entries here will be empty until those functions are
    # implemented.
    # ------------------------------------------------------------
    results = {
        "QUAAD": {
            "wall": wall_async, "test_mean": async_mean_test, "train_mean": async_mean_train,
            "test_runs": async_runs_test, "train_runs": async_runs_train,
            "comp_cost_mean": async_mean_comp, "comm_cost_mean": async_mean_comm,
            "comp_cost_runs": async_runs_comp, "comm_cost_runs": async_runs_comm,
        },
        "Power-of-Choice": {
            "wall": wall_poc, "test_mean": poc_mean_test, "train_mean": poc_mean_train,
            "test_runs": poc_runs_test, "train_runs": poc_runs_train,
            "comp_cost_mean": poc_mean_comp, "comm_cost_mean": poc_mean_comm,
            "comp_cost_runs": poc_runs_comp, "comm_cost_runs": poc_runs_comm,
        },
        "FLANP": {
            "wall": wall_flanp, "test_mean": flanp_mean_test, "train_mean": flanp_mean_train,
            "test_runs": flanp_runs_test, "train_runs": flanp_runs_train,
            "comp_cost_mean": flanp_mean_comp, "comm_cost_mean": flanp_mean_comm,
            "comp_cost_runs": flanp_runs_comp, "comm_cost_runs": flanp_runs_comm,
        },
        "Unified": {
            "wall": wall_unified, "test_mean": unified_mean_test, "train_mean": unified_mean_train,
            "test_runs": unified_runs_test, "train_runs": unified_runs_train,
            "comp_cost_mean": unified_mean_comp, "comm_cost_mean": unified_mean_comm,
            "comp_cost_runs": unified_runs_comp, "comm_cost_runs": unified_runs_comm,
        },
        "DelayHetSampling": {
            "wall": wall_delayhetsampling, "test_mean": delayhetsampling_mean_test, "train_mean": delayhetsampling_mean_train,
            "test_runs": delayhetsampling_runs_test, "train_runs": delayhetsampling_runs_train,
            "comp_cost_mean": delayhetsampling_mean_comp, "comm_cost_mean": delayhetsampling_mean_comm,
            "comp_cost_runs": delayhetsampling_runs_comp, "comm_cost_runs": delayhetsampling_runs_comm,
        },
        "Generalized FedAvg": {
            "wall": wall_generalized, "test_mean": generalized_mean_test, "train_mean": generalized_mean_train,
            "test_runs": generalized_runs_test, "train_runs": generalized_runs_train,
            "comp_cost_mean": generalized_mean_comp, "comm_cost_mean": generalized_mean_comm,
            "comp_cost_runs": generalized_runs_comp, "comm_cost_runs": generalized_runs_comm,
        },
    }
    return results


# ============================================================
# 9. Sweep over NUM_CLIENTS at a fixed MAX_TIME
# ============================================================

def plot_vs_num_clients(
        client_list,
        MAX_TIME,
        myData          = "MNIST",
        partition       = "nonIID",
        NUM_RUNS        = 4,
        topK_factor     = 0.1,
        qualityRatio    = 4.0,
        dirichlet_alpha = 0.05,
        metric          = "final",
        save_dir        = None
    ):
    """
    Runs the full algorithm suite once per NUM_CLIENTS value in
    `client_list`, holding MAX_TIME (and every other setting) fixed,
    and plots each algorithm's test accuracy against NUM_CLIENTS.

    This calls run_all_algos(...) once per entry in client_list, so it
    reuses all of its existing per-run plots/.npz saves -- this
    function just additionally collects the "how good was each
    algorithm by the time budget MAX_TIME ran out" number from each
    call and plots those against NUM_CLIENTS.

    Parameters
    ----------
    client_list : list[int]
        NUM_CLIENTS values to sweep over, e.g. [10, 20, 50, 100].
    MAX_TIME : int
        Fixed wall-clock time budget used for every point in the sweep.
    metric : "final" or "best"
        "final" -> test accuracy at the last logged timestep (i.e.
                   what each algorithm actually achieved by MAX_TIME).
        "best"  -> highest test accuracy reached at any point within
                   MAX_TIME (less sensitive to noise right at the end).
    save_dir : str or None
        Where to save the summary plot/.npz. Defaults to
        "<myData>_vs_clients_<partition>".

    Returns
    -------
    client_list, summary_mean, summary_std : the same lists you'd need
    to remake the plot yourself (e.g. in a notebook) without rerunning
    the experiments.
    """
    if metric not in ("final", "best"):
        raise ValueError("metric must be 'final' or 'best'")

    algo_names = ["QUAAD", "Power-of-Choice", "FLANP", "Unified",
                  "DelayHetSampling", "Generalized FedAvg"]
    summary_mean = {name: [] for name in algo_names}
    summary_std  = {name: [] for name in algo_names}

    for n_clients in client_list:
        print(f"\n########## NUM_CLIENTS = {n_clients} (MAX_TIME={MAX_TIME}) ##########")
        results = run_all_algos(
            NUM_RUNS        = NUM_RUNS,
            NUM_CLIENTS     = n_clients,
            MAX_TIME        = MAX_TIME,
            topK_factor     = topK_factor,
            partition       = partition,
            myData          = myData,
            qualityRatio    = qualityRatio,
            dirichlet_alpha = dirichlet_alpha
        )

        for name in algo_names:
            runs = results[name]["test_runs"]  # one accuracy curve per seed
            if metric == "final":
                vals = [r[-1] for r in runs if len(r) > 0]
            else:
                vals = [max(r) for r in runs if len(r) > 0]

            if len(vals) == 0:
                # Happens for algorithms that are currently stubbed
                # out (e.g. Unified / Generalized FedAvg) and return
                # no logged points at all.
                summary_mean[name].append(np.nan)
                summary_std[name].append(np.nan)
            else:
                summary_mean[name].append(float(np.mean(vals)))
                summary_std[name].append(float(np.std(vals)))

    # ---- Plot accuracy vs NUM_CLIENTS ----
    plt.figure(figsize=(8, 6))
    for name in algo_names:
        means = np.array(summary_mean[name])
        stds  = np.array(summary_std[name])
        if np.all(np.isnan(means)):
            print(f"Skipping '{name}' in the sweep plot — it returned no data "
                  f"(this algorithm is currently stubbed out).")
            continue
        plt.errorbar(client_list, means, yerr=stds, marker="o", capsize=3, label=name)

    plt.xlabel("Number of Clients")
    plt.ylabel("Test Accuracy @ MAX_TIME" if metric == "final" else "Best Test Accuracy (within MAX_TIME)")
    plt.title(f"{myData} ({partition}) — Test Accuracy vs. Number of Clients  [MAX_TIME={MAX_TIME}]")
    plt.legend()
    plt.grid()

    if save_dir is None:
        save_dir = f"{myData}_vs_clients_{partition}"
    os.makedirs(save_dir, exist_ok=True)
    fig_path = f"{save_dir}/accuracy_vs_clients_{metric}.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()
    plt.close()

    npz_kwargs = {"client_list": np.array(client_list)}
    for name in algo_names:
        key = name.replace(" ", "_").replace("-", "_")
        npz_kwargs[f"{key}_mean"] = np.array(summary_mean[name])
        npz_kwargs[f"{key}_std"]  = np.array(summary_std[name])
    npz_path = f"{save_dir}/accuracy_vs_clients_{metric}.npz"
    np.savez(npz_path, **npz_kwargs)

    print(f"\n✅ Sweep complete — plot saved to {fig_path}, data saved to {npz_path}")
    return client_list, summary_mean, summary_std


# ============================================================
# 7. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ============================================================
# 8. Run
# ============================================================
# Every combination below uses the exact same codebase — only the
# `myData` ("MNIST" / "CIFAR10") and `partition` ("IID" / "nonIID")
# flags change. Comment/uncomment whichever runs you want.

# --- Single run: MNIST, non-IID ---
run_all_algos(NUM_RUNS=8, NUM_CLIENTS=50, MAX_TIME=1200, topK_factor=0.3,
              partition="nonIID", myData="MNIST", qualityRatio=4.0)

# --- MNIST, IID ---
# run_all_algos(NUM_RUNS=8, NUM_CLIENTS=50, MAX_TIME=1200, topK_factor=0.3,
#               partition="IID", myData="MNIST", qualityRatio=4.0)

# --- CIFAR10, non-IID ---
# run_all_algos(NUM_RUNS=8, NUM_CLIENTS=50, MAX_TIME=1200, topK_factor=0.3,
#               partition="nonIID", myData="CIFAR10", qualityRatio=4.0)

# --- CIFAR10, IID ---
# run_all_algos(NUM_RUNS=8, NUM_CLIENTS=50, MAX_TIME=1200, topK_factor=0.3,
#               partition="IID", myData="CIFAR10", qualityRatio=4.0)

# --- Sweep: performance vs. NUM_CLIENTS at a fixed MAX_TIME ---
# Runs the whole suite once per client count in the list below, then
# plots each algorithm's test accuracy against NUM_CLIENTS.
# NOTE: this multiplies your total runtime by len(client_list), so
# NUM_RUNS is lowered here relative to the single-run example above.
# plot_vs_num_clients(
#     client_list = [10, 20, 50, 100, 200],
#     MAX_TIME    = 1200,
#     myData      = "MNIST",
#     partition   = "nonIID",
#     NUM_RUNS    = 3,
#     topK_factor = 0.3,
#     qualityRatio= 4.0,
#     metric      = "final"   # or "best"
# )