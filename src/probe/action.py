import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
import math
import tqdm
from typing import Optional, List
import wandb

from .probe_utils import ProbeDataset, stratified_split

def train_action_probe(cfg, dataset_list, outer_step, device="cuda", action_meanings: Optional[List[str]] = None):
    dataset = ProbeDataset(dataset_list)

    action_counts = defaultdict(int)
    for _, action, _ in dataset:
        action_counts[int(action)] += 1
    
    # --- METRIC 1: Action Entropy ---
    total_actions = sum(action_counts.values())
    action_entropy = -sum((c / total_actions) * math.log2(c / total_actions) for c in action_counts.values() if c > 0)
    
    if action_meanings is not None:
        action_counts = {action_meanings[int(k)]: v for k, v in action_counts.items()}
    print(f"Action distribution in dataset: {action_counts}")

    # --- METRIC 2: Representation Effective Rank ---
    # Stack all representations to compute covariance (Fast because rep_dim is usually small)
    states_tensor = torch.stack([x[0] for x in dataset_list]).float()
    state_var = torch.var(states_tensor, dim=0).mean().item()
    
    # Calculate eigenvalues of the covariance matrix
    C = torch.cov(states_tensor.T)
    eigenvalues = torch.linalg.eigvalsh(C)
    eigenvalues = torch.relu(eigenvalues) # Guard against tiny numerical negatives
    
    # Calculate Shannon entropy of the normalized eigenvalues (Effective Dimensionality)
    p = eigenvalues / (eigenvalues.sum() + 1e-8)
    p = p[p > 1e-7] # Filter out zeros for log
    effective_rank = torch.exp(-torch.sum(p * torch.log(p))).item()

    # Split and Train as usual
    train_ds, test_ds = stratified_split(dataset, test_frac=cfg.probe.test_frac, seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.probe.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.probe.batch_size)

    rep_dim = dataset[0][0].shape[-1]
    probe = nn.Linear(rep_dim, cfg.action_size).to(device)

    optimizer = optim.Adam(probe.parameters(), lr=cfg.probe.lr)
    criterion = nn.CrossEntropyLoss()

    patience = getattr(cfg.probe, "patience", 5)
    min_delta = getattr(cfg.probe, "min_delta", 1e-4)
    best_loss = float('inf')
    epochs_no_improve = 0

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    pbar = tqdm.tqdm(range(cfg.probe.epochs), desc="Training action probe")

    for epoch in pbar:
        # ----- Train -----
        probe.train()
        train_loss_sum = 0.0
        correct, total = 0, 0
        for states, actions, _ in train_loader:
            states = states.to(device)
            actions = actions.to(device)

            preds = probe(states)
            loss = criterion(preds, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * states.size(0)
            correct += (preds.argmax(1) == actions).sum().item()
            total += actions.size(0)
        
        avg_train_loss = train_loss_sum / total
        train_acc = correct / total

        # ----- Evaluate -----
        probe.eval()
        test_loss_sum = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for states, actions, _ in test_loader:
                states = states.to(device)
                actions = actions.to(device)
                
                preds = probe(states)
                loss = criterion(preds, actions)
                
                test_loss_sum += loss.item() * states.size(0)
                correct += (preds.argmax(1) == actions).sum().item()
                total += actions.size(0)

        avg_test_loss = test_loss_sum / total
        test_acc = correct / total

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        pbar.set_postfix({
            "test_acc": f"{test_acc:.3f}",
            "test_loss": f"{avg_test_loss:.3f}",
        })

        if avg_test_loss < best_loss - min_delta:
            best_loss = avg_test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    # Log metrics and FULL INTERACTIVE CHARTS tied to the outer RL step
    epochs_list = list(range(len(train_losses)))
    wandb.log({
        "outer_step": outer_step,
        "probe_action/final_train_accuracy": train_acc,
        "probe_action/final_test_accuracy": test_acc,
        "probe_action/final_test_loss": avg_test_loss,
        # Dataset Complexity Metrics
        "probe_action/dataset_action_entropy": action_entropy,
        "probe_action/dataset_rep_mean_variance": state_var,
        "probe_action/dataset_rep_effective_rank": effective_rank,
        "probe_action/dataset_size": len(dataset),
        # Curves
        "probe_action/loss_curve": wandb.plot.line_series(
            xs=epochs_list, ys=[train_losses, test_losses], keys=["Train", "Test"], title="Action Probe Loss", xname="Epoch"
        ),
        "probe_action/accuracy_curve": wandb.plot.line_series(
            xs=epochs_list, ys=[train_accs, test_accs], keys=["Train", "Test"], title="Action Probe Accuracy", xname="Epoch"
        )
    })