import torch
import torch.nn as nn

def init_normalization(channels, norm_type="bn", one_d=False, num_groups=4):
    assert norm_type in ["bn", "bn_nt", "ln", "ln_nt", "gn", None]
    if norm_type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=True, momentum=0.01)
        else:
            return nn.BatchNorm2d(channels, affine=True, momentum=0.01)
        
    elif norm_type == "bn_nt":
        if one_d:
            return nn.BatchNorm1d(channels, affine=False, momentum=0.01)
        else:
            return nn.BatchNorm2d(channels, affine=False, momentum=0.01)
        
    elif norm_type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=True)
        else:
            return nn.GroupNorm(1, channels, affine=True)
    
    elif norm_type == "ln_nt":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=False)
        else:
            return nn.GroupNorm(1, channels, affine=False)
        
    elif norm_type == 'gn':
        return nn.GroupNorm(num_groups, channels, affine=False)
    
    elif norm_type is None:
        return nn.Identity()
    
    else:
        raise ValueError(f"Invalid normalization type: {norm_type}")
    

def load_model_weights(model, checkpoint_path, device, load_layers):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        ex_key = list(checkpoint["backbone"].keys())[0]
        if "_orig_mod.module" in ex_key or "_orig_mod." in ex_key:
            def fix_state_dict(state_dict):
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k.replace("_orig_mod.module.", "").replace("_orig_mod.", "")
                    new_state_dict[name] = v
                return new_state_dict
            checkpoint["backbone"] = fix_state_dict(checkpoint["backbone"])
            checkpoint["neck"] = fix_state_dict(checkpoint["neck"])
            checkpoint["head"] = fix_state_dict(checkpoint["head"])
        
        for ln in load_layers:
            module = getattr(model, ln)
            module.load_state_dict(checkpoint[ln])
        
        print(f"Successfully loaded weights from {checkpoint_path} for layers: {load_layers}")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise KeyError(f"Failed to load checkpoint from {checkpoint_path}. Please check the file and try again.")