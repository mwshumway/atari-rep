from torch.utils.data import Dataset, Subset
from torch import nn
import random
from collections import defaultdict
from einops import rearrange
import torch

# ---------------- Dataset ----------------
class ProbeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, reward = self.data[idx]
        return state.float(), action, reward
    
def create_probe_dataset(env_trajectories, cfg):
    dataset = []
    n_step = cfg.probe.n_step
    gamma = cfg.gamma

    def _is_transition(x):
        return isinstance(x, (list, tuple)) and len(x) == 4

    def _signed_reward(r):
        r = float(r)
        if r > 0:
            return 1.0
        if r < 0:
            return -1.0
        return 0.0

    # Support both:
    # 1) on-policy: list of trajectories, each trajectory is a list of (state, action, reward, done)
    # 2) off-policy: a single mixed list of (state, action, reward, done)
    #    (possibly wrapped as [mixed_list])
    if len(env_trajectories) == 0:
        raise ValueError("No trajectories provided to create_probe_dataset.")

    first = env_trajectories[0]
    if _is_transition(first):
        # Flat transition list -> split into pseudo-trajectories at done=True boundaries
        trajectories = []
        current_traj = []
        for transition in env_trajectories:
            if not _is_transition(transition):
                raise ValueError("Malformed transition encountered in flat probe dataset.")
            current_traj.append(transition)
            if bool(transition[3]):
                trajectories.append(current_traj)
                current_traj = []
        if len(current_traj) > 0:
            trajectories.append(current_traj)
    else:
        # List of trajectories
        trajectories = env_trajectories

    for traj in trajectories:
        T = len(traj)
        if T == 0:
            continue

        # Efficiently compute n-step returns
        for t in range(T):
            ret = 0.0
            discount = 1.0
            for k in range(n_step):
                if t + k >= T:
                    break
                _, _, reward_k, done_k = traj[t + k]
                ret += discount * _signed_reward(reward_k)
                if done_k:  # Episode ended, stop bootstrapping
                    break
                discount *= gamma

            # Append state, action, and the computed return-to-go
            state_t, action_t, _, _ = traj[t]
            dataset.append((state_t, action_t, ret))

    if len(dataset) >= cfg.probe.dataset_size:
        dataset = dataset[:cfg.probe.dataset_size]
    else:
        raise ValueError(f"Not enough data to create probe dataset. Required: {cfg.probe.dataset_size}, Available: {len(dataset)}")

    return dataset

def build_probe(rep_dim, action_size, hidden_sizes, device):
    layers = []

    if len(hidden_sizes) == 0:
        layers.append(nn.Linear(rep_dim, action_size))
    else:
        # input layer
        layers.append(nn.Linear(rep_dim, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())

        # output layer
        layers.append(nn.Linear(hidden_sizes[-1], action_size))

    probe = nn.Sequential(*layers).to(device)
    return probe

def _collate(batch, f=4):
    """
    [params] 
        observation: (n, t+f-1, c, h, w) 
        next_observation: (n, t+f-1, c, h, w)
        action:   (n, t+f-1)
        reward:   (n, t+f-1)
        terminal: (n, n+t+f-1) * different n's (batch vs n_step)
        rtg:      (n, t+f-1)
        game_id:  (n, t+f-1)            
    [returns] 
        (c = 1 in ATARI)
        obs:      (n, t, f, c, h, w) 
        next_obs: (n, t, f, c, h, w)
        action:   (n, t)
        reward:   (n, t)
        done:     (n, n+t)
        rtg:      (n, t)
        game_id:  (n, t)    
    """
    obs = batch['observation']
    action = batch['action']
    reward = batch['reward']
    done = batch['terminal']
    rtg = batch['rtg']
    game_id = batch['game_id']
    next_obs = batch['next_observation']

    # process data-format
    obs = rearrange(obs, 'n tf c h w -> n tf 1 c h w')
    obs = obs.repeat(1, 1, f, 1, 1, 1)
    next_obs = rearrange(next_obs, 'n tf c h w -> n tf 1 c h w')
    next_obs = next_obs.repeat(1, 1, f, 1, 1, 1)
    action = action.long()
    reward = torch.nan_to_num(reward).sign()
    done = done.bool()
    rtg = rtg.float()
    game_id = game_id.long()

    # frame-stack
    if f != 1:
        for i in range(1, f):
            obs[:, :, i] = obs[:, :, i].roll(-i, 1)
            next_obs[:, :, i] = next_obs[:, :, i].roll(-i, 1)
        obs = obs[:, :-(f-1)]
        next_obs = next_obs[:, :-(f-1)]
        action = action[:, f-1:]
        reward = reward[:, f-1:]
        done = done[:, f-1:]
        rtg = rtg[:, f-1:]
        game_id = game_id[:, f-1:]
        
    # lazy frame to float
    obs = obs / 255.0
    next_obs = next_obs / 255.0
        
    batch = {
        'obs': obs,
        'next_obs': next_obs,
        'act': action,
        'rew': reward,
        'done': done,
        'rtg': rtg,
        'game_id': game_id,                            
    }            
        
    return batch