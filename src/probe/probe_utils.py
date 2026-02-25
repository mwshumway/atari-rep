from torch.utils.data import Dataset, Subset
from torch import nn
import random
from collections import defaultdict

# ---------------- Dataset ----------------
class ProbeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, reward = self.data[idx]
        return state.float(), action, reward
    
# ---------------- Stratified Split ----------------
def stratified_split(dataset, test_frac=0.2, seed=0):
    rng = random.Random(seed)

    # group indices by action
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, action, _ = dataset[idx]
        class_indices[int(action)].append(idx)

    train_indices = []
    test_indices = []

    for cls, indices in class_indices.items():
        rng.shuffle(indices)
        split = int(len(indices) * (1 - test_frac))
        train_indices.extend(indices[:split])
        test_indices.extend(indices[split:])

    return Subset(dataset, train_indices), Subset(dataset, test_indices)

def create_probe_dataset(env_trajectories, cfg):
    dataset = []
    n_step = cfg.probe.n_step
    gamma = cfg.gamma
    
    for i in range(cfg.num_eval_envs):
        traj = env_trajectories[i]
        T = len(traj)
        # Efficiently compute n-step returns
        for t in range(T):
            ret = 0.0
            discount = 1.0
            for k in range(n_step):
                if t + k >= T:
                    break
                state_k, action_k, reward_k, done_k = traj[t + k]
                ret += discount * reward_k
                if done_k:  # Episode ended, stop bootstrapping
                    break
                discount *= gamma
            
            # Append state, action, and the computed return-to-go
            state_t, action_t, _, _ = traj[t]
            dataset.append((state_t, action_t, ret))
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