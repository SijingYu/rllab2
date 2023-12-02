import numpy as np
from rllab.misc import tensor_utils
import time

import torchvision

import matplotlib.pyplot as plt
# %matplotlib inline

import clip as clip
import torch
from PIL import Image


def freeze(layer):
    
    # Looping through each child layer of the input layer
    for child in layer.children():
        
        # Looping through each parameter of each child layer
        for param in child.parameters():
            # Setting the parameter's 'requires_grad' attribute to False
            # to freeze it's learning during training
            param.requires_grad = False


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    render = lambda : plt.imshow(env.render(mode='rgb_array'))
    o = env.reset()
    agent.reset()
    path_length = 0

    active_model, preprocess = clip.load('ViT-B/16')

    freeze(active_model)

    if animated:
        # env.render()
        render()
    while path_length < max_path_length:

        with torch.no_grad():

            reward_info = "The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points."

            image_input = preprocess(Image.fromarray(o)).unsqueeze(0).cuda()
            text_inputs = torch.cat([clip.tokenize(reward_info)]).cuda()

            image_features = active_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_features = active_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # print('image_features.shape:', image_features.shape)
            # print('text_features.shape:', text_features.shape)
            # print('torch.cat([image_features, text_features], 0)', torch.cat([image_features, text_features], 1).shape)
            # o = (image_features + text_features).detach().cpu()
            o = torch.cat([image_features, text_features], 1).detach().cpu()

        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )
