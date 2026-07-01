# =====================================================
# MULTI-CLASS CIFAR10: ASYNC vs SYNC FEDERATED LEARNING
# =====================================================

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import math

# -----------------------------
# Reproducibility
# -----------------------------
# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_global_training_set(clients):
    X_all = np.vstack([c["X"] for c in clients])
    y_all = np.hstack([c["y"] for c in clients])
    return X_all, y_all

# =====================================================
# 1. LOAD CIFAR10 (10-CLASS)
# =====================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    ),
    transforms.Lambda(lambda x: x.view(-1))  # flatten
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

def load_dataset(dataset):
    X, y = [], []
    for x, label in dataset:
        X.append(x.numpy())
        y.append(label)
    return np.array(X), np.array(y)

X_train, y_train = load_dataset(trainset)
X_test, y_test = load_dataset(testset)

d = X_train.shape[1]
C = 10  # number of classes

# =====================================================
# 2. SOFTMAX REGRESSION MODEL
# =====================================================

def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def one_hot(y, C):
    Y = np.zeros((len(y), C))
    Y[np.arange(len(y)), y] = 1
    return Y

def softmax_grad(W, X, y):
    Y = one_hot(y, C)
    P = softmax(X @ W)
    return X.T @ (P - Y) / X.shape[0]

def accuracy(W, X, y):
    preds = np.argmax(X @ W, axis=1)
    return np.mean(preds == y)

# =====================================================
# 3. CLIENT CREATION (IID)
# =====================================================

def create_clients(K=20, samples_per_client=3000, good_fraction=0.3):
    idx = np.random.permutation(len(X_train))
    splits = np.array_split(idx, K)

    clients = []
    num_good = int(K * good_fraction)

    for k in range(K):
        X = X_train[splits[k]]
        y = y_train[splits[k]]

        if k < num_good:
            # GOOD clients: slow, accurate
            clients.append({
                "X": X,
                "y": y,
                "compute_time": 200, #40,
                "epochs": 5,
                "lr": 0.01, #0.05,
                "sigma": 0.01,
                "quality": 2.0 #2.0
            })
        else:
            # BAD clients: fast, noisy
            clients.append({
                "X": X,
                "y": y,
                "compute_time": 10,
                "epochs": 1,
                "lr": 0.08, #0.3,
                "sigma": 0.5, #0.3,
                "quality": 0.3 #0.5
            })

    return clients

def lambda_schedule(t, lam_max=0.05, tau=300):
    """
    Smooth warm-up for staleness penalty.
    - lam_max: final lambda value
    - tau: time constant controlling how fast lambda grows
    """
    return lam_max * (1.0 - np.exp(-t / tau))

# =====================================================
# 3. NON-IID CLIENT CREATION (LABEL SKEW)
# =====================================================

def create_non_iid_clients(
    K=20,
    samples_per_client=3000,
    classes_per_client=2,
    good_fraction=0.3
):
    clients = []
    num_good = int(K * good_fraction)

    label_indices = {
        i: np.where(y_train == i)[0] for i in range(C)
    }

    for k in range(K):
        chosen_labels = np.random.choice(
            C, classes_per_client, replace=False
        )

        X_list, y_list = [], []
        per_label = samples_per_client // classes_per_client

        for lbl in chosen_labels:
            idx = np.random.choice(
                label_indices[lbl],
                per_label,
                replace=True   # critical for stability
            )
            X_list.append(X_train[idx])
            y_list.append(y_train[idx])

        X = np.vstack(X_list)
        y = np.hstack(y_list)

        if k < num_good:
            clients.append({
                "X": X,
                "y": y,
                "compute_time": 200,
                "epochs": 5,
                "lr": 0.01,
                "sigma": 0.01,
                "quality": 2.0
            })
        else:
            clients.append({
                "X": X,
                "y": y,
                "compute_time": 10,
                "epochs": 1,
                "lr": 0.08,
                "sigma": 0.5,
                "quality": 0.3
            })

    return clients

# =====================================================
# 4. LOCAL TRAINING
# =====================================================

def local_train(W, client):
    W_local = W.copy()
    for _ in range(client["epochs"]):
        grad = softmax_grad(W_local, client["X"], client["y"])
        grad += client["sigma"] * np.random.randn(*grad.shape)
        W_local -= client["lr"] * grad
    return W_local

# =====================================================
# 5. ASYNCHRONOUS FL (OUR SCHEME)
# =====================================================

def async_fl(
    clients,
    T=1500,
    eta=1.0,
    lam_max=0.05,
    tau=300
):
    W = np.zeros((d, C))
    arrivals = defaultdict(list)

    acc_test_log, acc_train_log, wall_log = [], [], []

    X_train_all, y_train_all = build_global_training_set(clients)

    for t in range(T):

        # compute current lambda
        lam_t = lambda_schedule(t, lam_max=lam_max, tau=tau)

        # clients start training
        for k, c in enumerate(clients):
            if t % c["compute_time"] == 0:
                W_snap = W.copy()
                W_local = local_train(W_snap, c)
                delta = W_local - W_snap
                arrivals[t + c["compute_time"]].append((k, delta, t))

        # aggregate arrivals
        if t in arrivals:
            updates = arrivals[t]
            weights, deltas = [], []

            for k, delta, t0 in updates:
                staleness = t - t0
                weight = clients[k]["quality"] * np.exp(-lam_t * staleness)
                weights.append(weight)
                deltas.append(delta)

            weights = np.array(weights)
            alphas = weights / np.sum(weights)
            W += eta * np.sum(alphas[:, None, None] * np.array(deltas), axis=0)

        acc_test_log.append(accuracy(W, X_test, y_test))
        acc_train_log.append(accuracy(W, X_train_all, y_train_all))
        wall_log.append(t)

    return (
        np.array(wall_log),
        np.array(acc_test_log),
        np.array(acc_train_log),
    )

# =====================================================
# 6. SYNCHRONOUS FL (THEIR SCHEME)
# =====================================================

def sync_fl(clients, R=120, M=5, eta=1.0):
    W = np.zeros((d, C))
    wall = 0

    acc_test_log, acc_train_log, wall_log = [], [], []

    X_train_all, y_train_all = build_global_training_set(clients)

    for _ in range(R):
        selected = random.sample(range(len(clients)), M)
        deltas, times = [], []

        for k in selected:
            c = clients[k]
            W_local = local_train(W, c)
            deltas.append(W_local - W)
            times.append(c["compute_time"])

        wall += max(times)
        W += eta * np.mean(deltas, axis=0)

        wall_log.append(wall)
        acc_test_log.append(accuracy(W, X_test, y_test))
        acc_train_log.append(accuracy(W, X_train_all, y_train_all))

    return np.array(wall_log), np.array(acc_test_log), np.array(acc_train_log)

# =====================================================
# FLANP (STAGE-WISE SYNCHRONOUS PREFIX SELECTION)
# =====================================================

def flanp_fl(
    clients,
    stages=[5, 10, 20],     # prefix sizes
    rounds_per_stage= 10, #20,
    eta=1.0
):
    """
    FLANP-style training:
    - Clients sorted by speed
    - Stage-wise inclusion of clients
    - Fully synchronous within each stage
    """
    # sort clients by compute_time (fastest first)
    clients_sorted = sorted(clients, key=lambda c: c["compute_time"])

    W = np.zeros((d, C))
    wall = 0

    acc_test_log, acc_train_log, wall_log = [], [], []

    X_train_all, y_train_all = build_global_training_set(clients)

    for stage_size in stages:
        active_clients = clients_sorted[:stage_size]

        for _ in range(rounds_per_stage):
            deltas, times = [], []

            for c in active_clients:
                W_local = local_train(W, c)
                deltas.append(W_local - W)
                times.append(c["compute_time"])

            # synchronous barrier
            wall += max(times)
            W += eta * np.mean(deltas, axis=0)

            wall_log.append(wall)
            acc_test_log.append(accuracy(W, X_test, y_test))
            acc_train_log.append(accuracy(W, X_train_all, y_train_all))

    return (
        np.array(wall_log),
        np.array(acc_test_log),
        np.array(acc_train_log),
    )

# =====================================================
# POWER-OF-CHOICE (Cho et al.)
# =====================================================

def power_of_choice_fl(
    clients,
    R=120,              # communication rounds
    m=5,                # number of selected clients
    d_pool=10,          # candidate pool size (>= m)
    eta=1.0
):
    """
    Power-of-Choice FL:
    1. Sample d_pool clients uniformly
    2. Compute their local losses at current global model
    3. Select top-m highest loss clients
    4. Aggregate synchronously
    """

    W = np.zeros((d, C))
    wall = 0

    acc_test_log, acc_train_log, wall_log = [], [], []

    X_train_all, y_train_all = build_global_training_set(clients)

    for _ in range(R):

        # -----------------------------
        # Step 1: Random candidate pool
        # -----------------------------
        candidate_ids = random.sample(range(len(clients)), d_pool)

        # -----------------------------
        # Step 2: Evaluate local losses
        # -----------------------------
        losses = []
        for k in candidate_ids:
            c = clients[k]
            Y = one_hot(c["y"], C)
            P = softmax(c["X"] @ W)
            loss = -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))
            losses.append(loss)

        # -----------------------------
        # Step 3: Select top-m highest loss
        # -----------------------------
        sorted_idx = np.argsort(losses)[::-1]   # descending
        selected = [candidate_ids[i] for i in sorted_idx[:m]]

        # -----------------------------
        # Step 4: Local training
        # -----------------------------
        deltas, times = [], []

        for k in selected:
            c = clients[k]
            W_local = local_train(W, c)
            deltas.append(W_local - W)
            times.append(c["compute_time"])

        # -----------------------------
        # Step 5: Synchronous aggregation
        # -----------------------------
        wall += max(times)
        W += eta * np.mean(deltas, axis=0)

        wall_log.append(wall)
        acc_test_log.append(accuracy(W, X_test, y_test))
        acc_train_log.append(accuracy(W, X_train_all, y_train_all))

    return (
        np.array(wall_log),
        np.array(acc_test_log),
        np.array(acc_train_log),
    )

# =====================================================
# UNIFIED APPROACH (Generalized FedAvg, Arbitrary Participation)
# =====================================================

def unified_fl(
    clients,
    R=120,              # communication rounds
    M=5,                # number of participating clients per round
    eta=1.0,
    biased_sampling=False
):
    """
    Unified FL (Arbitrary Participation)

    Implements:
        W_{t+1} = W_t + eta * sum_k q_t^k Δ_t^k

    where q_t^k reflects arbitrary participation.
    If biased_sampling=True, clients are sampled
    with probability proportional to their quality
    (to simulate arbitrary participation bias).
    """

    W = np.zeros((d, C))
    wall = 0

    acc_test_log, acc_train_log, wall_log, W_log = [], [], [], []

    X_train_all, y_train_all = build_global_training_set(clients)

    K = len(clients)

    # sampling probabilities (can be arbitrary)
    if biased_sampling:
        qualities = np.array([c["quality"] for c in clients])
        probs = qualities / np.sum(qualities)
    else:
        probs = None  # uniform

    for _ in range(R):

        # -----------------------------
        # 1. Arbitrary client selection
        # -----------------------------
        if biased_sampling:
            selected = np.random.choice(
                K, size=M, replace=False, p=probs
            )
        else:
            selected = random.sample(range(K), M)

        deltas, times = [], []

        # -----------------------------
        # 2. Local training
        # -----------------------------
        for k in selected:
            c = clients[k]
            W_local = local_train(W, c)
            deltas.append(W_local - W)
            times.append(c["compute_time"])

        # -----------------------------
        # 3. Aggregation
        # q_t^k = 1/M for selected clients
        # -----------------------------
        deltas = np.array(deltas)
        q = 1.0 / M

        W += eta * q * np.sum(deltas, axis=0)

        # synchronous wall-clock
        wall += max(times)

        wall_log.append(wall)
        acc_test_log.append(accuracy(W, X_test, y_test))
        acc_train_log.append(accuracy(W, X_train_all, y_train_all))
        W_log.append(W.copy())

    return (
        np.array(wall_log),
        np.array(acc_test_log),
        np.array(acc_train_log),
        np.array(W_log)
    )

# =====================================================
# 7. RUN MULTI-SEED EXPERIMENT
# =====================================================

NUM_RUNS = 5 #10

# storage
async_runs_test = []
sync_runs_test = []
flanp_runs_test = []
poc_runs_test = []
unified_runs_test = []

async_runs_train = []
sync_runs_train = []
flanp_runs_train = []
poc_runs_train = []
unified_runs_train = []

for seed in range(NUM_RUNS):

    print(f"Running seed {seed}")
    set_seed(seed)

    # regenerate clients each run (important!)
    clients = create_non_iid_clients()

    # run algorithms
    wall_async, acc_async_test, acc_async_train = async_fl(
        clients, T=24000, lam_max=0.05, tau=300
    )

    wall_sync, acc_sync_test, acc_sync_train = sync_fl(clients)

    wall_flanp, acc_flanp_test, acc_flanp_train = flanp_fl(
        clients, stages=[5, 10, 20, 40], rounds_per_stage=20
    )

    wall_poc, acc_poc_test, acc_poc_train = power_of_choice_fl(
        clients, R=120, m=5, d_pool=10
    )

    wall_unified, acc_unified_test, acc_unified_train, W_unified = unified_fl(
        clients, R=120, M=5, eta=1.0, biased_sampling=True
    )

    # store results
    async_runs_test.append(acc_async_test)
    sync_runs_test.append(acc_sync_test)
    flanp_runs_test.append(acc_flanp_test)
    poc_runs_test.append(acc_poc_test)
    unified_runs_test.append(acc_unified_test)

    async_runs_train.append(acc_async_train)
    sync_runs_train.append(acc_sync_train)
    flanp_runs_train.append(acc_flanp_train)
    poc_runs_train.append(acc_poc_train)
    unified_runs_train.append(acc_unified_train)

# Convert to numpy arrays
async_runs_test = np.array(async_runs_test)
sync_runs_test = np.array(sync_runs_test)
flanp_runs_test = np.array(flanp_runs_test)
poc_runs_test = np.array(poc_runs_test)
unified_runs_test = np.array(unified_runs_test)

async_runs_train = np.array(async_runs_train)
sync_runs_train = np.array(sync_runs_train)
flanp_runs_train = np.array(flanp_runs_train)
poc_runs_train = np.array(poc_runs_train)
unified_runs_train = np.array(unified_runs_train)

# Means
async_mean_test = async_runs_test.mean(axis=0)
sync_mean_test = sync_runs_test.mean(axis=0)
flanp_mean_test = flanp_runs_test.mean(axis=0)
poc_mean_test = poc_runs_test.mean(axis=0)
unified_mean_test = unified_runs_test.mean(axis=0)

async_mean_train = async_runs_train.mean(axis=0)
sync_mean_train = sync_runs_train.mean(axis=0)
flanp_mean_train = flanp_runs_train.mean(axis=0)
poc_mean_train = poc_runs_train.mean(axis=0)
unified_mean_train = unified_runs_train.mean(axis=0)

# =====================================================
# 8. PLOTS
# =====================================================
window = 5 #10

def smooth(y, window=10):
    if window <= 1:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')

# Test accuracy
plt.figure(figsize=(7,4))
# plt.plot(wall_async_s, acc_async_test, label="QUAAD (ours)")
# plt.plot(wall_sync, acc_sync_test, label="Synchronous", linewidth=2)
# plt.plot(wall_flanp, acc_flanp_test, label="FLANP", linestyle="--", linewidth=2)
# plt.plot(wall_poc, acc_poc_test, label="Power-of-Choice", linestyle=":", linewidth=2)
# plt.plot(wall_unified, acc_unified_test, label="Unified (Arbitrary Participation)", linestyle="-.", linewidth=2)

plt.plot(wall_async[window-1:], smooth(async_mean_test, window), label="QUAAD (ours)")
plt.plot(wall_sync[window-1:], smooth(sync_mean_test, window), label="Synchronous", linewidth=2)
plt.plot(wall_flanp[window-1:], smooth(flanp_mean_test, window), label="FLANP", linestyle="--", linewidth=2)
plt.plot(wall_poc[window-1:], smooth(poc_mean_test, window), label="Power-of-Choice", linestyle=":", linewidth=2)
plt.plot(wall_unified[window-1:], smooth(unified_mean_test, window), label="Unified", linestyle="-.", linewidth=2)

plt.xlabel("Wall-clock time")
plt.ylabel("Test accuracy")
plt.title("Non IID CIFAR10 (10-class): Test Accuracy vs Wall-clock Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots_cifar_noniid_log_regression_cpu/Test_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()

# Training accuracy
plt.figure(figsize=(7,4))
# plt.plot(wall_async[window-1:], acc_async_train, label="QUAAD – Train")
# plt.plot(wall_sync[window-1:], acc_sync_train, label="Sync – Train", linewidth=2)
# plt.plot(wall_flanp[window-1:], acc_flanp_train, label="FLANP – Train", linestyle="--", linewidth=2)
# plt.plot(wall_poc[window-1:], acc_poc_train, label="PoC – Train", linestyle=":", linewidth=2)
# plt.plot(wall_unified[window-1:], acc_unified_train, label="Unified – Train", linestyle="-.", linewidth=2)

plt.plot(wall_async[window-1:], smooth(async_mean_train, window), label="QUAAD (ours)")
plt.plot(wall_sync[window-1:], smooth(sync_mean_train, window), label="Synchronous", linewidth=2)
plt.plot(wall_flanp[window-1:], smooth(flanp_mean_train, window), label="FLANP", linestyle="--", linewidth=2)
plt.plot(wall_poc[window-1:], smooth(poc_mean_train, window), label="Power-of-Choice", linestyle=":", linewidth=2)
plt.plot(wall_unified[window-1:], smooth(unified_mean_train, window), label="Unified", linestyle="-.", linewidth=2)

plt.xlabel("Wall-clock time")
plt.ylabel("Training accuracy")
plt.title("Non IID CIFAR10 (10-class): Training Accuracy vs Wall-clock Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots_cifar_noniid_log_regression_cpu/Training_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()

np.savez(
    f"results_seed_noniid_cifar_log_regression_cpu_{seed}.npz",
    wall_async=wall_async[window-1:],
    acc_async_test=smooth(async_mean_test, window),
    acc_async_train=smooth(async_mean_train, window),
    wall_sync=wall_sync[window-1:],
    acc_sync_test=smooth(sync_mean_test, window),
    acc_sync_train=smooth(sync_mean_train, window),
    wall_flanp=wall_flanp[window-1:],
    acc_flanp_test=smooth(flanp_mean_test, window),
    acc_flanp_train=smooth(flanp_mean_train, window),
    wall_poc=wall_poc[window-1:],
    acc_poc_test=smooth(poc_mean_test, window),
    acc_poc_train=smooth(poc_mean_train, window),
    wall_unified=wall_unified[window-1:],
    acc_unified_test=smooth(unified_mean_test, window),
    acc_unified_train=smooth(unified_mean_train, window),
    W_unified=W_unified
)

# torch.save(model.state_dict(),f"model_async_seed_{seed}.pt")