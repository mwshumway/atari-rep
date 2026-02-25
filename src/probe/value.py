import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import wandb

from .probe_utils import ProbeDataset, stratified_split

def compute_metrics(preds, targets):
    mse = nn.functional.mse_loss(preds, targets).item()
    target_mean = torch.mean(targets)
    ss_tot = torch.sum((targets - target_mean) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    return mse, r2.item()

def train_value_probe(cfg, dataset_list, outer_step, device="cuda"):
    
    # --- METRIC 3: Return Variance ---
    returns_tensor = torch.tensor([x[2] for x in dataset_list], dtype=torch.float32)
    return_var = torch.var(returns_tensor).item()
    return_mean = torch.mean(returns_tensor).item()
    
    dataset = ProbeDataset(dataset_list)
    train_ds, test_ds = stratified_split(dataset, test_frac=cfg.probe.test_frac, seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.probe.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.probe.batch_size)

    rep_dim = dataset[0][0].shape[-1]
    probe = nn.Linear(rep_dim, 1).to(device)

    optimizer = optim.Adam(probe.parameters(), lr=cfg.probe.lr)
    criterion = nn.MSELoss()

    patience = getattr(cfg.probe, "patience", 5)
    min_delta = getattr(cfg.probe, "min_delta", 1e-4)
    best_loss = float('inf')
    epochs_no_improve = 0

    train_mses, test_mses = [], []
    train_r2s, test_r2s = [], []

    pbar = tqdm.tqdm(range(cfg.probe.epochs), desc="Training value probe")

    for epoch in pbar:
        # ----- Train -----
        probe.train()
        train_preds, train_targets = [], []
        
        for states, _, returns in train_loader:
            states = states.to(device)
            returns = returns.to(device).float().unsqueeze(1)

            preds = probe(states)
            loss = criterion(preds, returns)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_preds.append(preds.detach())
            train_targets.append(returns.detach())
        
        train_preds = torch.cat(train_preds, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        train_mse, train_r2 = compute_metrics(train_preds, train_targets)

        # ----- Evaluate -----
        probe.eval()
        test_preds, test_targets = [], []

        with torch.no_grad():
            for states, _, returns in test_loader:
                states = states.to(device)
                returns = returns.to(device).float().unsqueeze(1)
                
                preds = probe(states)
                
                test_preds.append(preds)
                test_targets.append(returns)

        test_preds = torch.cat(test_preds, dim=0)
        test_targets = torch.cat(test_targets, dim=0)
        test_mse, test_r2 = compute_metrics(test_preds, test_targets)

        train_mses.append(train_mse)
        test_mses.append(test_mse)
        train_r2s.append(train_r2)
        test_r2s.append(test_r2)

        pbar.set_postfix({
            "test_r2": f"{test_r2:.3f}",
        })

        if test_mse < best_loss - min_delta:
            best_loss = test_mse
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    # Log metrics and FULL INTERACTIVE CHARTS tied to the outer RL step
    epochs_list = list(range(len(train_mses)))
    wandb.log({
        "outer_step": outer_step,
        "probe_value/final_train_mse": train_mse,
        "probe_value/final_test_mse": test_mse,
        "probe_value/final_train_r2": train_r2,
        "probe_value/final_test_r2": test_r2,
        # Dataset Complexity Metrics
        "probe_value/dataset_return_variance": return_var,
        "probe_value/dataset_return_mean": return_mean,
        "probe_value/dataset_size": len(dataset),
        # Curves
        "probe_value/mse_curve": wandb.plot.line_series(
            xs=epochs_list, ys=[train_mses, test_mses], keys=["Train", "Test"], title="Value Probe MSE", xname="Epoch"
        ),
        "probe_value/r2_curve": wandb.plot.line_series(
            xs=epochs_list, ys=[train_r2s, test_r2s], keys=["Train", "Test"], title="Value Probe R2", xname="Epoch"
        )
    })