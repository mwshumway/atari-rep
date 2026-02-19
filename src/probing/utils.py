"""
Probing utilities.
"""

import torch
import tqdm
from einops import rearrange
import matplotlib.pyplot as plt


def extract_representations(model, dataloader, device, collate_fn):
    """
    Extract representations from the model for all data in the dataloader.
    
    :param model: the model to extract representations from (has backbone, neck and head)
    :param dataloader: the data loader providing input data
    :param device: the device to run the model on
    :param collate_fn: the function to collate batches  

    :return: a dictionary containing stacked representations and associated data (e.g obs_neck, action, reward, done, game_id, etc.)
    """
    model.eval()
    _stack_of_batches = {}
    for batch in tqdm.tqdm(dataloader, desc="Representation Extraction"):
        for k, v in batch.items():
            batch[k] = v.to(device)
        batch = collate_fn(batch)
        obs, game_id = batch["obs"], batch["game_id"] # (obs: (n, t, f, c, h, w), game_id: (n, t))
        probe_obs = obs[:, :1] # only use the first frame for probing (n, 1, f, c, h, w)
        probe_game_id = game_id[:, :1] # (n, 1)

        with torch.no_grad():
            backbone_feat, _ = model.backbone(probe_obs)
            neck_feat, _ = model.neck(backbone_feat, probe_game_id)
            neck_feat = neck_feat[:, 0, :] # take representation corresponding to the first frame
            neck_feat = neck_feat.unsqueeze(1) # (n, 1, d)
                
        batch['obs_neck'] = neck_feat
        del batch['obs']  # remove obs to save memory
        del batch['next_obs']  # remove next_obs to save memory

        for k, v in batch.items():
            if k not in _stack_of_batches:
                _stack_of_batches[k] = []
            _stack_of_batches[k].append(v)
    
    stack_of_batches = {}
    for k, v in _stack_of_batches.items():
        _v = torch.cat(v)
        _v = _v.flatten(start_dim=0, end_dim=1).unsqueeze(1)
        stack_of_batches[k] = _v
    
    return stack_of_batches


def extract_layer_representations(model, dataloader, device, collate_fn, zero_shot=True):
    """
    Extract representations from the model for all data in the dataloader.

    This function does so for each layer in the model's neck (currently a total of 4 layers)
    
    :param model: the model to extract representations from (has backbone, neck and head)
    :param dataloader: the data loader providing input data
    :param device: the device to run the model on
    :param collate_fn: the function to collate batches  

    :return: a dictionary containing stacked representations and associated data (e.g obs_neck, action, reward, done, game_id, etc.)
    """
    model.eval()
    _stack_of_batches = {}
    for batch in tqdm.tqdm(dataloader, desc="Representation Extraction"):
        for k, v in batch.items():
            batch[k] = v.to(device)
        batch = collate_fn(batch)
        obs, game_id = batch["obs"], batch["game_id"]
        probe_obs = obs[:, :1] # only use the first frame for probing (n, 1, f, c, h, w)
        probe_game_id = game_id[:, :1] # (n, 1)

        # sample_obs = probe_obs[:1]
        # from matplotlib import pyplot as plt
        # plt.imshow(sample_obs[0,0,-1].cpu().squeeze(), cmap='gray')
        # plt.title('Sample Observation Frame for Probing')
        # plt.axis('off')
        # plt.savefig('sample_observation_frame2.png')
        # plt.close()
        # return

        with torch.no_grad():
            backbone_feat, _ = model.backbone(probe_obs)
            if zero_shot:
                _, neck_info = model.neck(backbone_feat, None) # zero-shot setting, not finetuned on specific game. Will use just the base spatial embedding.
            else:
                _, neck_info = model.neck(backbone_feat, probe_game_id)
        
        n_reps = len(neck_info)
        for i in range(1, n_reps + 1):
            batch[f"obs_rep{i}"] = neck_info[f'rep_candidate_{i}']
        
        del batch['obs']  # remove obs to save memory
        del batch['next_obs']  # remove next_obs to save memory

        for k, v in batch.items():
            if k not in _stack_of_batches:
                _stack_of_batches[k] = []
            _stack_of_batches[k].append(v)
    
    stack_of_batches = {}
    for k, v in _stack_of_batches.items():
        _v = torch.cat(v)
        _v = _v.flatten(start_dim=0, end_dim=1).unsqueeze(1)
        stack_of_batches[k] = _v
    
    return stack_of_batches, n_reps
    

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


def plot_layer_probing_results(probing_metrics, save_path):
    """
    Plot the probing results for each layer and save the figures.

    Given value, reward, and action probing results for each layer.

    Make 9 subplots: 3 columns (value, action, reward), 3 rows (training losses, training metrics, eval metrics)

    Args:
        probing_metrics (dict): {layer_name: {probe_type: [eval_metrics, loss_hist, metrics_hist]}} 
        save_path: (str): path to save the figure
    Returns:
        None
    """
    plt.figure(figsize=(18, 12))
    probe_types = ["value", "action", "reward"]
    row_titles = ["Evaluation Metrics", "Training Loss", "Training Metrics"]

    for col, probe_type in enumerate(probe_types):
        for row in range(3):
            plt.subplot(3, 3, row * 3 + col + 1)
            
            # For eval metrics, plot bar chart of final eval metric for each layer
            if row == 0:
                final_metrics = []
                for layer_name in probing_metrics.keys():
                    eval_metrics = probing_metrics[layer_name][probe_type][0]
                    if probe_type == "action":
                        final_metrics.append(eval_metrics['accuracy'])
                    elif probe_type == "reward":
                        final_metrics.append(eval_metrics['accuracy'])
                    elif probe_type == "value":
                        final_metrics.append(eval_metrics['r2'])
                plt.bar(probing_metrics.keys(), final_metrics)
                plt.ylabel("Final Eval Metric")
                plt.ylim(0, 1)
            else:
                for layer_name in probing_metrics.keys():
                    if row == 1:
                        loss_hist = probing_metrics[layer_name][probe_type][1]
                        plt.plot(loss_hist, label=layer_name)
                        plt.ylabel("Training Loss")
                    elif row == 2:
                        metrics_hist = probing_metrics[layer_name][probe_type][2]
                        if probe_type == "action" or probe_type == "reward":
                            plt.plot([m['accuracy'] for m in metrics_hist], label=layer_name)
                            plt.ylabel("Training Accuracy")
                        elif probe_type == "value":
                            plt.plot([m['r2'] for m in metrics_hist], label=layer_name)
                            plt.ylabel("Training R2")
                plt.xlabel("Training Epochs")
                plt.legend()
            plt.title(f"{probe_type.capitalize()} Probing - {row_titles[row]}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Probing results figure saved to {save_path}")

