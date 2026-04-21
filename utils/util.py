import numpy as np
import math
import torch
import yaml
import os
from easydict import EasyDict as edict


def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_algorithm_profile(merged):
    """
    功能:
        根据exp.algorithm_name将defaults中的算法参数组覆盖到通用参数区。
    输入:
        merged (dict): defaults与用户yaml已深度合并后的配置。
    输出:
        dict: 叠加算法档案后的配置。
    """
    exp_cfg = dict(merged.get("exp", {}))
    if "algorithm_name" not in exp_cfg:
        raise KeyError("Missing exp.algorithm_name in merged config.")
    algo_name = str(exp_cfg["algorithm_name"])

    profiles = dict(merged.get("algorithm_profiles", {}))
    if algo_name not in profiles:
        raise KeyError(
            "Unsupported exp.algorithm_name: {}. Available profiles: {}".format(
                str(algo_name),
                list(profiles.keys()),
            )
        )

    profile = dict(profiles[algo_name])
    merged_out = dict(merged)
    if "overrides" in profile:
        merged_out = _deep_merge(merged_out, dict(profile["overrides"]))

    exp_out = dict(merged_out.get("exp", {}))
    if "backend" in profile:
        exp_out["algorithm_backend"] = str(profile["backend"])
    exp_out["algorithm_profile"] = str(algo_name)
    merged_out["exp"] = exp_out
    return merged_out


def load_config(path):
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "defaults.yaml",
    )
    with open(default_path, "r", encoding="utf-8") as f:
        defaults = yaml.safe_load(f) or {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    merged = _deep_merge(defaults, data)
    merged = _apply_algorithm_profile(merged)
    return edict(merged)

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
        
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c
