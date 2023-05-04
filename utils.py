
import torch
import gym
from procgen import ProcgenEnv
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from agents.impala import ImpalaAgent
from agents.mixer import MixerAgent
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def create_env(num_envs, env_name, num_levels, start_level=0, distribution_mode="easy", use_sequential_levels=False):
    envs = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode, use_sequential_levels=use_sequential_levels)
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    return envs

def compute_saliency_maps(agent, envs, obs):
    # agent.network.eval()
    obs.requires_grad_(True)

    act, lp_act, ent, value, logits = agent.get_action_and_value(obs)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Policy saliency map
    saliency_maps_policy = []

    for i in range(envs.single_action_space.n):
        log_prob = log_probs[:, i]
        grad_log_prob = torch.autograd.grad(log_prob.sum(), obs, create_graph=True)[0]
        saliency_maps_policy.append(grad_log_prob)

    saliency_map_policy = torch.stack(saliency_maps_policy).sum(0).abs()

    # Value saliency map
    grad_value = torch.autograd.grad(value.sum(), obs, create_graph=True)[0]
    saliency_map_value = grad_value.abs()

    # Calculate entropy of policy and value saliency maps flattened
    saliency_map_policy_softmax = torch.nn.functional.softmax(saliency_map_policy.flatten(), dim=-1)
    saliency_map_value_softmax = torch.nn.functional.softmax(saliency_map_value.flatten(), dim=-1)

    # sum entropy
    entropy_policy = -(saliency_map_policy_softmax * torch.log(saliency_map_policy_softmax)).sum()
    entropy_value = -(saliency_map_value_softmax * torch.log(saliency_map_value_softmax)).sum()

    # scale maps to [0, 1]
    saliency_map_policy = saliency_map_policy / saliency_map_policy.max()
    saliency_map_value = saliency_map_value / saliency_map_value.max()

    return saliency_map_policy.squeeze().detach().cpu().numpy(), saliency_map_value.squeeze().detach().cpu().numpy(), entropy_policy, entropy_value


import cv2
import numpy as np

def overlay_heatmap_on_observation(observation, heatmap):
    # Increase size of observation with no interpolation or smoothing
    observation = cv2.resize(observation, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_NEAREST) # 64x64x3 -> 512x512x3

    # make obs grey
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = cv2.cvtColor(observation, cv2.COLOR_GRAY2RGB)

    # Mean across the color channels
    heatmap_normalized = heatmap.mean(axis=-1)
    heatmap_normalized = (heatmap_normalized - heatmap_normalized.min()) / (heatmap_normalized.max() - heatmap_normalized.min())

    # make a new observation image where the heatmap is the alpha channel
    heatmap_normalized = np.stack([heatmap_normalized, heatmap_normalized, heatmap_normalized], axis=-1)
    heatmap_normalized = (heatmap_normalized * 255).astype(np.uint8)
    heatmap_normalized = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_BONE)
    heatmap_normalized = cv2.cvtColor(heatmap_normalized, cv2.COLOR_BGR2RGB)

    # Increase size of heatmap with no interpolation or smoothing
    heatmap_normalized = cv2.resize(heatmap_normalized, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_NEAREST) # 64x64x3 -> 512x512x3

    # smooth heatmap for 8 pixels
    heatmap_normalized = cv2.blur(heatmap_normalized, (12,12))

    blended_image = (observation * 0.5 + heatmap_normalized * 0.5).astype(np.uint8)

    return blended_image


def run_saliency_steps(agent, envs, num_steps, writer:SummaryWriter, global_step, device):
    print
    obs = envs.reset()
    step_count = 0

    video_frames = {name: [] for name in ['Observation', 'Policy', 'Policy_combined', 'Value', 'Value_combined']}
    entropies_policy = []
    entropies_value = []

    while step_count < num_steps:
        # print(f'Step {step_count} of {num_steps}')
        action, _, _, _, _ = agent.get_action_and_value(torch.from_numpy(obs).float().to(device))
        next_obs, _, done, _ = envs.step(action.cpu().numpy())

        sm_policy, sm_value, ent_policy, ent_value = compute_saliency_maps(agent, envs, torch.tensor(obs).float().to(device))
        entropies_policy.append(ent_policy.item())
        entropies_value.append(ent_value.item())

        sm_policy_combined = overlay_heatmap_on_observation(obs[0], sm_policy)
        sm_value_combined = overlay_heatmap_on_observation(obs[0], sm_value)

        video_frames['Observation'].append(obs[0])
        video_frames['Policy'].append(sm_policy * 255)
        video_frames['Policy_combined'].append(sm_policy_combined)
        video_frames['Value'].append(sm_value * 255)
        video_frames['Value_combined'].append(sm_value_combined)

        obs = next_obs
        step_count += 1

    avg_entropy_policy = np.mean(entropies_policy)
    avg_entropy_value = np.mean(entropies_value)
    writer.add_scalar('entropy_policy', avg_entropy_policy, global_step)
    writer.add_scalar('entropy_value', avg_entropy_value, global_step)

    print(f'avg_entropy_policy: {avg_entropy_policy}, avg_entropy_value: {avg_entropy_value}')

    for name, frames in video_frames.items():
        video_array = np.stack(frames, axis=0).astype(np.uint8)
        path = writer.file_writer.get_logdir()

        # Save video to file
        with imageio.get_writer(f'{path}/{name}_video_{global_step}.mp4', fps=15) as file_writer:
            for frame in video_array:
                file_writer.append_data(frame)

        # Add video to TensorBoard
        video_array_tb = torch.tensor(video_array.clip(0,255),dtype=torch.uint8).permute(0, 3, 1, 2).unsqueeze(0).numpy()
        writer.add_video(f'{name}/video', video_array_tb, global_step=global_step, fps=15)



if __name__ == '__main__':

    envs = create_env(num_envs=1, env_name='bigfish', num_levels=1, use_sequential_levels=True, start_level=42, distribution_mode="easy")
    agent = MixerAgent(envs)

    writer = SummaryWriter(log_dir='saliency_map_tests/a')

    run_saliency_steps(agent, envs, num_steps=40, writer=writer, global_step=202)

