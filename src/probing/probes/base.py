"""
src/probing/probes/base.py
==========================

Base class for probes.
"""

from torch import nn, optim
from einops import rearrange
from sklearn.metrics import f1_score, mean_squared_error, r2_score
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt

from src.probing.datasets.base import BaseProbeDataset


class BaseProbe(ABC):
    """
    Base class for linear probing experiments.
    
    :param cfg: configuration object with necessary parameters
    :param device: device to run the probe on
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        # will be set during train()
        self.model = None
        self.optimizer = None
        self.criterion = None

        self.plot = self.cfg.probing.plot
        self.zero_shot = self.cfg.probing.zero_shot
        self.epochs = self.cfg.probing.epochs
        self.action_epochs = self.cfg.probing.action_epochs
    
    @abstractmethod
    def _build_model(self, input_dim: int) -> nn.Module:
        """Build the probe model."""
        pass

    @abstractmethod
    def _build_criterion(self) -> nn.Module:
        """Build the loss function."""
        pass

    @abstractmethod
    def _compute_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray,
        game_id=None
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        pass

    @abstractmethod
    def _process_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Process model output to get predictions (e.g., argmax for classification)."""
        pass

    def _prepare_target(self, target):
        """Prepare the target for loss and metric computation."""
        return target

    def train(
        self,
        train_dataset: BaseProbeDataset,
        batch_size: int = 512,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ) -> Tuple[Dict[str, float], list, list]:
        """
        Train the probe and return metrics. Evaluation is done on the same 
        dataset as we train on. This is done to measure the level of which 
        the representation linearly (or simply) encodes the information.
        We are testing if the representation is expressive enough to encode
        the information, not if the probe can generalize well.
        
        Args:
            train_dataset: Training probe dataset
            batch_size: Batch size for training
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            
        Returns:
            Dictionary of evaluation metrics
        """    
        ptype = train_dataset.probe_type
        if ptype == 'action':
            num_epochs = self.action_epochs # actions may need more training
        else:
            num_epochs = self.epochs


        if ptype == 'value': # then normalize targets and save mean and std
            target_mean = train_dataset.targets.mean(dim=0, keepdim=True)
            target_std = train_dataset.targets.std(dim=0, keepdim=True) + 1e-8
            train_dataset.targets = (train_dataset.targets - target_mean) / target_std
            train_dataset.target_mean = target_mean # type: ignore
            train_dataset.target_std = target_std # type: ignore

        # Create dataloaders
        dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size,
                                  shuffle=True)
        
        # Build model, criterion, optimizer
        input_dim = train_dataset.representations[0].shape[-1] # feature dim
        self.model = self._build_model(input_dim).to(self.device)
        self.criterion = self._build_criterion().to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        print(f"Training {self.__class__.__name__}...")
        print(f"Train size: {len(train_dataset)}")

        loss_history, metrics_history = [], []
        # Training loop
        self.model.train()
        pbar = tqdm.tqdm(range(num_epochs), desc=f"Training {ptype.capitalize()} Probe")
        for epoch in pbar:

            epoch_loss = 0.0
            epoch_metrics = []
            for batch in dataloader:
                # Move to device
                feat = batch['representation'].to(self.device)
                target = batch['target'].to(self.device)
                target = self._prepare_target(target) # necessary for reward probe to shift (-1, 0, 1) -> (0, 1, 2)
                game_id = batch['game_id'].to(self.device)

                # Forward pass
                if not self.zero_shot:
                    output = self._forward(feat, game_id)
                else:
                    output = self._forward(feat, None)
                
                # Flatten for loss computation
                output = rearrange(output, 'n t d -> (n t) d')
                target = rearrange(target, 'n t -> (n t)')

                # if value, we need to flatten the output from ((nt), 1) to (nt)
                if ptype == 'value':
                    output = output.squeeze(1)

                # Compute loss
                loss = self.criterion(output, target)
                epoch_loss += loss.item()

                # Compute metrics on batch
                predictions = self._process_predictions(output)

                batch_metrics = self._compute_metrics(
                    predictions.detach().cpu().numpy(), 
                    target.cpu().numpy()
                )
                epoch_metrics.append(batch_metrics)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            avg_epoch_loss = epoch_loss / len(dataloader) # divide by number of batches
            loss_history.append(avg_epoch_loss)

            # Average metrics over epoch
            avg_epoch_metrics = {}
            for key in epoch_metrics[0].keys():
                avg_epoch_metrics[key] = np.mean([m[key] for m in epoch_metrics])
            metrics_history.append(avg_epoch_metrics)

            _display = {'loss': f'{avg_epoch_loss:.4f}'}
            if ptype in ['reward', 'action']:
                _display['acc'] = f'{avg_epoch_metrics["accuracy"]:.3f}'
                _display['f1_macro'] = f'{avg_epoch_metrics["f1_macro"]:.3f}'
            else:
                _display['mse'] = f'{avg_epoch_metrics["mse"]:.4f}'
                _display['r2'] = f'{avg_epoch_metrics["r2"]:.3f}'

            pbar.set_postfix(_display)

        if self.plot:
            self._plot_loss_curve(loss_history)
            self._plot_metrics_curve(metrics_history)
        
        # Evaluation
        eval_metrics = self._evaluate(dataloader)

        return eval_metrics, loss_history, metrics_history


    def _forward(self, x: torch.Tensor, game_id) -> torch.Tensor:
        """Forward pass through the probe model."""
        if hasattr(self.model, 'forward'):
            output, _ = self.model(x, game_id) # type: ignore
        else:
            output = self.model(x) # type: ignore
        return output
        
    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the probe on the data."""
        self.model.eval() # type: ignore
        
        all_predictions = []
        all_targets = []
        all_game_ids = []
        
        with torch.no_grad():
            for batch in dataloader:
                feat = batch['representation'].to(self.device)
                target = batch['target'].to(self.device)
                target = self._prepare_target(target) # necessary for reward probe to shift (-1, 0, 1) -> (0, 1, 2)
                game_id = batch['game_id'].to(self.device)
                
                # Forward pass
                if not self.zero_shot:
                    output = self._forward(feat, game_id)
                else:
                    output = self._forward(feat, None)
                
                # Flatten
                output = rearrange(output, 'n t d -> (n t) d')
                target = rearrange(target, 'n t ... -> (n t) ...')
                game_id = rearrange(game_id, 'n t -> (n t)')
                
                # Get predictions
                predictions = self._process_predictions(output)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                all_game_ids.append(game_id.cpu().numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        game_ids = np.concatenate(all_game_ids)
        
        # unnormalize if value probe
        if dataloader.dataset.probe_type == 'value': # type: ignore
            target_mean = dataloader.dataset.target_mean.item() # type: ignore
            target_std = dataloader.dataset.target_std.item() # type: ignore
            predictions = predictions * target_std + target_mean
            targets = targets * target_std + target_mean

        # Compute metrics
        metrics = self._compute_metrics(predictions, targets, game_id=game_ids)
        
        return metrics

    def _plot_loss_curve(self, loss_history):
        """Plot the training loss curve."""
        plt.figure(figsize=(8, 5))
        plt.plot(loss_history, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Probe Training Loss Curve')
        plt.legend()
        plt.grid()
        fig_path = f'/projectnb/ds598xz/students/mshumway/atari-rep-bench/img/probing/{self.cfg.probing.img_tag}_loss_curve.png'
        from pathlib import Path
        Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'/projectnb/ds598xz/students/mshumway/atari-rep-bench/img/probing/{self.cfg.probing.img_tag}_loss_curve.png')
        plt.close()
        print(f"Saved loss curve to /projectnb/ds598xz/students/mshumway/atari-rep-bench/img/probing/{self.cfg.probing.img_tag}_loss_curve.png")

    @abstractmethod
    def _plot_metrics_curve(self, metrics_history):
        """Plot the training metrics curve."""
        pass

   