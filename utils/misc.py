import torch


def load_ckpt(ckpt_file, map_location=None):
    ckpt = torch.load(ckpt_file, map_location=map_location)
    module_state_dict = dict()
    for k, m in ckpt.items():
        module_state_dict[k] = m

    return module_state_dict
