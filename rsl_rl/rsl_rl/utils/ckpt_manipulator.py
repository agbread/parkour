"""
# A python module that manipulates torch checkpoint file in a hacky way.
Each function should be used with caution and should be used only when thoughtfully considered.
---
Args:
    source_state_dict: the state_dict loaded using torch.load
    algo_state_dict: the algorithm state_dict summarized from algorithm as an example
---
Returns:
    new_state_dict: the state_dict that has been manipulated or directly saved as a checkpoint file.
"""
import torch
from collections import OrderedDict

def replace_encoder0_and_critic(source_state_dict, algo_state_dict):
    """ Warm-start a distill student from a specialist whose critic input layout differs
    (e.g. privileged obs gained engaging_block): keep the specialist's actor side
    (actor/memory_a/std/estimator), but take the first actor encoder (new modality,
    e.g. forward_depth) and the whole critic side (memory_c/critic/critic_encoders)
    from the untrained student.
    """
    print("\033[1;36m Keeping source actor weights; encoders.0 and critic side use untrained weights \033[0m")
    new_model_state_dict = OrderedDict()
    for key, algo_param in algo_state_dict["model_state_dict"].items():
        is_critic_side = key.startswith("memory_c") or key.startswith("critic")
        is_actor_encoder0 = ("encoders.0" in key) and not key.startswith("critic_encoders")
        if is_actor_encoder0 or is_critic_side:
            print("key:", key, "shape:", tuple(algo_param.shape), "using untrained module weights.")
            new_model_state_dict[key] = algo_param
        else:
            source_param = source_state_dict["model_state_dict"][key]
            assert source_param.shape == algo_param.shape, \
                "Actor-side shape mismatch at {}: source {} vs student {}".format(
                    key, tuple(source_param.shape), tuple(algo_param.shape))
            new_model_state_dict[key] = source_param
    new_state_dict = dict(
        model_state_dict= new_model_state_dict,
        # No optimizer_state_dict
        iter= source_state_dict["iter"],
        infos= source_state_dict["infos"],
    )
    return new_state_dict

def replace_encoder0(source_state_dict, algo_state_dict):
    print("\033[1;36m Replacing encoder.0 weights with untrained weights and avoid critic_encoder.0 \033[0m")
    new_model_state_dict = OrderedDict()
    for key in algo_state_dict["model_state_dict"].keys():
        if "critic_encoders.0" in key:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
        elif "encoders.0" in key:
            print(
                "key:", key,
                "shape:", algo_state_dict["model_state_dict"][key].shape,
                "using untrained module weights.")
            new_model_state_dict[key] = algo_state_dict["model_state_dict"][key]
        else:
            new_model_state_dict[key] = source_state_dict["model_state_dict"][key]
    new_state_dict = dict(
        model_state_dict= new_model_state_dict,
        # No optimizer_state_dict
        iter= source_state_dict["iter"],
        infos= source_state_dict["infos"],
    )
    return new_state_dict
