"""
Data partitioning matching SplitOMC reference (DataManager.py).

Two functions:
- make_es_topology: sequential overlap topology (matches create_clusters_seq)
- nd1_partition: each ES = 40-70% classes; each client = 20% of total = 2 for CIFAR-10
- build_per_client_test_set: matches Evaluator.get_ratio_based_data
"""
import random
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


def make_es_topology(num_clients, num_edge_servers, overlap_percentage, seed=0):
    """
    Sequential overlap topology.

    Clients are sorted into cells; the last `overlap_count` clients of each cell
    are also associated with the next cell.

    Returns:
        client_to_es: dict {cid -> [primary_es, secondary_es?]}
        es_to_clients: dict {es_id -> [cid, ...]}
    """
    rng = random.Random(seed)
    client_to_es = defaultdict(list)
    min_clients_per_server = num_clients // num_edge_servers
    overlap_count = int(min_clients_per_server * (overlap_percentage / 100))

    for server_id in range(num_edge_servers):
        start = server_id * min_clients_per_server
        end = num_clients if server_id == num_edge_servers - 1 else (server_id + 1) * min_clients_per_server
        for cid in range(start, end):
            client_to_es[cid].append(server_id)
        if server_id < num_edge_servers - 1:
            for cid in range(end - overlap_count, end):
                client_to_es[cid].append(server_id + 1)

    es_to_clients = defaultdict(list)
    for cid, es_list in client_to_es.items():
        for es_id in es_list:
            es_to_clients[es_id].append(cid)

    return dict(client_to_es), dict(es_to_clients)


def nd1_partition(train_labels, num_classes, num_clients, client_to_es, es_to_clients,
                  classes_per_es_frac=(0.4, 0.7), classes_per_client_frac=0.2, seed=0):
    """
    ND1 partition.

    Step 1: Each ES picks a random 40-70% of all classes as its scope.
    Step 2: Each client picks 20% of total classes from its PRIMARY ES's scope.
    Step 3: For each class c, find all clients having c as main; partition the
            train samples of c uniformly among them.

    Returns:
        client_indices: {cid -> [train_idx, ...]}
        client_main_classes: {cid -> set of class IDs}
        es_scope: {es_id -> set of class IDs}
    """
    rng = random.Random(seed)
    all_classes = list(range(num_classes))

    es_scope = {}
    for es_id in es_to_clients.keys():
        n_classes = rng.randint(
            max(1, int(num_classes * classes_per_es_frac[0])),
            max(1, int(num_classes * classes_per_es_frac[1]))
        )
        es_scope[es_id] = set(rng.sample(all_classes, n_classes))

    n_main = max(1, int(num_classes * classes_per_client_frac))
    client_main_classes = {}
    for cid in range(num_clients):
        primary_es = client_to_es[cid][0]
        scope = list(es_scope[primary_es])
        if len(scope) >= n_main:
            client_main_classes[cid] = set(rng.sample(scope, n_main))
        else:
            client_main_classes[cid] = set(scope)

    class_to_clients = defaultdict(list)
    for cid, classes in client_main_classes.items():
        for c in classes:
            class_to_clients[c].append(cid)

    train_labels = np.asarray(train_labels)
    client_indices = {cid: [] for cid in range(num_clients)}
    for c in range(num_classes):
        clients_with_c = class_to_clients.get(c, [])
        if not clients_with_c:
            continue
        idxs = np.where(train_labels == c)[0].tolist()
        rng.shuffle(idxs)
        n_each = len(idxs) // len(clients_with_c)
        for i, cid in enumerate(clients_with_c):
            start, end = i * n_each, (i + 1) * n_each
            client_indices[cid].extend(idxs[start:end])

    return client_indices, client_main_classes, es_scope


def build_per_client_test_set(test_labels, client_main_classes, es_scope,
                              client_primary_es, all_used_classes,
                              oop_ratio, oor_ratio_factor=0.3):
    """
    Build per-client test set matching Evaluator.get_ratio_based_data.

    main: all global test samples of client's main classes
    OOP: classes in client's primary ES scope but not in main, sampled per ratio
    OOR: classes not in any client's main and not in primary ES scope
    """
    test_labels = np.asarray(test_labels)
    main_classes = set(client_main_classes)
    remaining = set(all_used_classes) - main_classes
    primary_scope = set(es_scope[client_primary_es])
    oop_classes = primary_scope & remaining
    oor_classes = remaining - primary_scope

    test_idxs = []
    for c in main_classes:
        idxs = np.where(test_labels == c)[0].tolist()
        test_idxs.extend(idxs)
    main_total = len(test_idxs)

    if len(oop_classes) > 0 and oop_ratio > 0:
        per_oop = max(1, int(main_total * oop_ratio / len(oop_classes)))
        for c in oop_classes:
            idxs = np.where(test_labels == c)[0].tolist()[:per_oop]
            test_idxs.extend(idxs)

    oor_ratio = oor_ratio_factor * oop_ratio
    if len(oor_classes) > 0 and oor_ratio > 0:
        per_oor = max(1, int(main_total * oor_ratio / len(oor_classes)))
        for c in oor_classes:
            idxs = np.where(test_labels == c)[0].tolist()[:per_oor]
            test_idxs.extend(idxs)

    return test_idxs, oop_classes, oor_classes


# ---- Dataset helpers ----

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10(data_root='./data_cache', augment=False):
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train = datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)
    return train, test


def make_client_dataloader(train_dataset, indices, batch_size=32, num_workers=0, shuffle=True):
    """Construct DataLoader for a single client's local data."""
    sub = Subset(train_dataset, indices)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle,
                     num_workers=num_workers, drop_last=False, pin_memory=False)


def compute_all_used_classes(client_main_classes):
    """Union of main classes across all clients."""
    all_used = set()
    for c, classes in client_main_classes.items():
        all_used.update(classes)
    return all_used


def es_neighbors_sequential(num_edge_servers):
    """Sequential ES topology neighbors (ring/line)."""
    nbrs = {}
    for i in range(num_edge_servers):
        n = []
        if i > 0:
            n.append(i - 1)
        if i < num_edge_servers - 1:
            n.append(i + 1)
        nbrs[i] = n
    return nbrs
