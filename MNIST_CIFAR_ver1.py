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

class CNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def extract_features(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        return self.fc2(self.extract_features(x))
        
class CNN_CIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def extract_features(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        return self.fc2(self.extract_features(x))
# ============================================================
# 3. Dataset
# ============================================================
def DataCreator(val_size=5000, whichdata = "MNIST", batch_size=512):

    global full_trainset, num_train, perm, val_indices,train_indices,valset, val_loader,trainset,testset,test_loader

    transform = transforms.ToTensor()

    # Load MNIST Dataset
    if whichdata == "MNIST":
        full_trainset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )        
        trainset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=False,
            transform=transform
        )    

    # Load CIFAR Dataset
    if whichdata == "CIFAR":
        full_trainset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )        
        trainset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=False,
            transform=transform
        )  

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
    new_model = model
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

    W_cov = copy_model(W)
    W_cov.eval()
    arrivals = defaultdict(list)

    wall = []
    test_log = []
    train_log = []

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

                local = local_train(W, clients[k])

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

        current_time += 1

    return wall, test_log, train_log

def synchronous_fedavg(clients, MAX_TIME, LOG_INTERVAL, eta=1.0):
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

    wall, test_log, train_log = [], [], []

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
   
    return wall, test_log, train_log

def async_fl_noexp(clients, MAX_TIME, LOG_INTERVAL, eta=0.1, lam=0.01, topK=2):
    wall, test_log, train_log = [], [], []
    return wall, test_log, train_log
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

    arrivals = defaultdict(list)
    wall, test_log, train_log = [], [], []
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
                    local = local_train(W, c)
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
        current_time += 1

        torch.cuda.empty_cache()

    return wall, test_log, train_log

# Reorganizing the code to fix the issue of unequal logging which creates an issue when computing mean at the end.
def flanp(clients, MAX_TIME, LOG_INTERVAL,
          eta=0.05, lam=0.01,
          mu=0.1, init_m=2, max_m=20):

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
                    partition       = 0,
                    myData          = "MNIST",
                    qualityRatio    = 4.0):
    global W
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

    # Create data sets
    DataCreator(val_size=5000, whichdata=myData, batch_size=512)
    if myData == "MNIST":
        W = CNN_MNIST().to(device)
    if myData ==  "CIFAR":
        W = CNN_CIFAR().to(device)        

    # Parameter setting
    LOG_INTERVAL                = 1
    lam                         = 0.05
    eta_quaad                   = 2
    eta_flanp                   = 0.3
    topK                        = int(topK_factor*NUM_CLIENTS)
    num_classes                 = len(set(label for _, label in trainset))
    labels                      = np.array(trainset.targets)

    for seed in range(NUM_RUNS):
        print(f"\n==== Seed {seed} ====")
        set_seed(seed)
        print(datetime.now().strftime("%H:%M:%S"))

        # clients = create_non_iid_clients()
        if partition == 0:
            client_indices = dirichlet_partition(trainset, NUM_CLIENTS, 0.05, min_size=10) # dirichlet_partition(trainset, 20, 0.3)  #Try dirichlet_partition(trainset, 10, 0.5) for faster run and reduce R=50 and T=800
        if partition == 1:
            client_indices = dirichlet_partition(trainset, NUM_CLIENTS, 0.05, min_size=10) # IID Partition

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


        wall_generalized, test_generalized, train_generalized = generalized_fedavg(clients, MAX_TIME, LOG_INTERVAL, gamma=0.01, eta=2.0, P=15, participation_prob=1.0,topK=topK)
        wall_async, test_async, train_async       = async_fl(clients, MAX_TIME, LOG_INTERVAL, eta=eta_quaad, lam=lam, topK=topK)
        wall_poc, test_poc, train_poc             = power_of_choice(clients, MAX_TIME, LOG_INTERVAL)
        wall_flanp, test_flanp, train_flanp       = flanp(clients, MAX_TIME, LOG_INTERVAL, eta=eta_flanp, lam=lam, mu=0.1, init_m=2, max_m=20)
        wall_unified, test_unified, train_unified = async_fl_noexp(clients, MAX_TIME, LOG_INTERVAL, eta=eta_quaad, lam=lam, topK=topK)
        wall_delayhetsampling, test_delayhetsampling, train_delayhetsampling = delayhetsampling(clients, MAX_TIME, LOG_INTERVAL, eta=0.05, lam=0.05, K=topK)
    
        async_runs_test.append(test_async)
        async_runs_train.append(train_async)
        poc_runs_test.append(test_poc)
        flanp_runs_test.append(test_flanp)
        poc_runs_train.append(train_poc)
        flanp_runs_train.append(train_flanp)
        unified_runs_test.append(test_unified)
        unified_runs_train.append(train_unified)
        delayhetsampling_runs_test.append(test_delayhetsampling)
        delayhetsampling_runs_train.append(train_delayhetsampling)
        generalized_runs_test.append(test_generalized)
        generalized_runs_train.append(train_generalized)

        dir_name = myData+f"_n{NUM_CLIENTS}_q{int(qualityRatio)}"
        if partition == 0:
            dir_name = dir_name+f"_NonIID"
        if partition == 1:
            dir_name = dir_name+f"_IID"          

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
    
        async_runs_test.append(test_async)
        async_runs_train.append(train_async)
        poc_runs_test.append(test_poc)
        flanp_runs_test.append(test_flanp)
        poc_runs_train.append(train_poc)
        flanp_runs_train.append(train_flanp)
        unified_runs_test.append(test_unified)
        unified_runs_train.append(train_unified)
        delayhetsampling_runs_test.append(test_delayhetsampling)
        delayhetsampling_runs_train.append(train_delayhetsampling)
        generalized_runs_test.append(test_generalized)
        generalized_runs_train.append(train_generalized)
        
        np.savez(f"{dir_name}/results_fl_gpu_{seed}.npz",
        wall_async=wall_async,
        async_test=test_async,
        async_train=train_async,
        wall_poc=wall_poc,
        wall_flanp=wall_flanp,
        poc_test=test_poc,
        flanp_test=test_flanp,
        poc_train=train_poc,
        flanp_train=train_flanp,
        wall_unified=wall_unified,
        unified_test=test_unified,
        unified_train=train_unified,
        delayhetsampling_test=test_delayhetsampling,
        delayhetsampling_train=train_delayhetsampling,
        generalized_test=test_generalized,
        generalized_train=train_generalized)

        

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

    np.savez(
        f"{dir_name}/results_fl_gpu.npz",
        wall_async              =wall_async,
        async_test              =async_mean_test,
        async_train             =async_mean_train,
        wall_poc                =wall_poc,
        wall_flanp              =wall_flanp,
        poc_test                =poc_mean_test,
        flanp_test              =flanp_mean_test,
        poc_train               =poc_mean_train,
        flanp_train             =flanp_mean_train,
        wall_unified            =wall_unified,
        unified_test            =unified_mean_test,
        unified_train           =unified_mean_train,
        delayhetsampling_test   =delayhetsampling_mean_test,
        delayhetsampling_train  =delayhetsampling_mean_train,
        generalized_test   =generalized_mean_test,
        generalized_train  =generalized_mean_train
    )

    print("\n✅ Experiment Complete – GPU Safe – Results Saved")


# ============================================================
# 7. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
run_all_algos(NUM_RUNS=2,NUM_CLIENTS=10,MAX_TIME=30,topK_factor=0.3,partition=0,myData="MNIST", qualityRatio=4.0)
#run_all_algos(NUM_RUNS=2,NUM_CLIENTS=10,MAX_TIME=30,topK_factor=0.3,partition=0,myData="MNIST", qualityRatio=2.0)
