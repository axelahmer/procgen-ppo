# modified code from cleanrl v1.0.0 to support additional evaluation metrics, also fixed some bugs.

import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from procgen import ProcgenEnv
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from agents.impala import ImpalaAgent
from agents.mixer import MixerAgent
from agents.mixer_flat_val import MixerAgentFlatVal
from agents.mixer_sigmoid import MixerAgentSigmoid
from agents.mixer_sigmoid_individual_entropy import MixerAgentSigmoidIndividualEntropy

AGENTS = {
    "impala": ImpalaAgent,
    "mixer": MixerAgent,
    "mixer-flat-val": MixerAgentFlatVal,
    "mixer-sigmoid": MixerAgentSigmoid,
    "mixer-sigmoid-individual-entropy": MixerAgentSigmoidIndividualEntropy
}


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--gpu-id", type=int, default=0,
        help="the id of the GPU to use")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="starpilot",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(25e6),
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=64,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=256,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.999,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=3,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--eval-freq", type=int, default=16,
        help="the frequency of evaluation")
    parser.add_argument("--num-eval-eps", type=int, default=256,
        help="the number of episodes for evaluation")
    parser.add_argument("--agent", type=str, default="impala",
        help="the agent to use")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

#TODO add agent choice, device choice. include in run_name.
#TODO look at rendering to video


if __name__ == "__main__":
    args = parse_args()

    run_name = f"{args.env_id}_{args.exp_name}_agent={args.agent}_seed={args.seed}_time={int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # env setup
    num_levels_train = 200 if args.eval_freq > 0 else 0
    envs = ProcgenEnv(num_envs=args.num_envs, env_name=args.env_id, num_levels=num_levels_train, start_level=0, distribution_mode="easy")
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
    
    norm_gamma = 0.99 # self.argsgamma
    envs = gym.wrappers.NormalizeReward(envs, gamma=norm_gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # eval env fn : evaluate on full distribution of levels
    def build_eval_envs():
        envs_eval = ProcgenEnv(num_envs=args.num_envs, env_name=args.env_id, num_levels=0, start_level=0, distribution_mode="easy")
        envs_eval = gym.wrappers.TransformObservation(envs_eval, lambda obs: obs["rgb"])
        envs_eval.single_action_space = envs_eval.action_space
        envs_eval.single_observation_space = envs_eval.observation_space["rgb"]
        envs_eval.is_vector_env = True
        envs_eval = gym.wrappers.RecordEpisodeStatistics(envs_eval)
        assert isinstance(envs_eval.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        return envs_eval
    

    agent = AGENTS[args.agent](envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        ep_returns_train = []
        ep_lengths_train = []
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    ep_returns_train.append(item["episode"]["r"])
                    ep_lengths_train.append(item["episode"]["l"])

        mean_return, std_return = np.mean(ep_returns_train), np.std(ep_returns_train)
        mean_length, std_length = np.mean(ep_lengths_train), np.std(ep_lengths_train)

        writer.add_scalar("charts/episodic_return_mean", mean_return, global_step)
        writer.add_scalar("charts/episodic_length_mean", mean_length, global_step)
        writer.add_scalar("charts/episodic_return_std", std_return, global_step)
        writer.add_scalar("charts/episodic_length_std", std_length, global_step)
        writer.add_scalar("charts/episodes_per_update", len(ep_returns_train), global_step)

        # pretty print step - mean episodic return - mean episodic length
        print(f"{args.env_id:<10} (seed={args.seed:^1} gpu={args.gpu_id:^1}) | step: {global_step:^10,d} | episodic return: {mean_return:6.2f} +/- {std_return:<6.2f} | episodic length: {mean_length:6.2f} +/- {std_length:<6.2f}")

        # run num_eval_episodes evaluation episodes if needed, using envs_eval
        if args.eval_freq > 0 and update % args.eval_freq == 0:

            ep_returns_eval = []
            ep_lengths_eval = []
            envs_eval = build_eval_envs() # discard the old envs_eval and build new ones
            next_obs_eval = torch.Tensor(envs_eval.reset()).to(device)

            while len(ep_returns_eval) < args.num_eval_eps:
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(next_obs_eval)
                next_obs_eval, reward, done, info = envs_eval.step(action.cpu().numpy())
                next_obs_eval = torch.Tensor(next_obs_eval).to(device)
                for item in info:
                    if "episode" in item.keys():
                        ep_returns_eval.append(item["episode"]["r"])
                        ep_lengths_eval.append(item["episode"]["l"])
            
            mean_return_eval, std_return_eval = np.mean(ep_returns_eval), np.std(ep_returns_eval)
            mean_length_eval, std_length_eval = np.mean(ep_lengths_eval), np.std(ep_lengths_eval)
            writer.add_scalar("charts/episodic_return_mean_eval", mean_return_eval, global_step)
            writer.add_scalar("charts/episodic_length_mean_eval", mean_length_eval, global_step)
            writer.add_scalar("charts/episodic_return_std_eval", std_return_eval, global_step)
            writer.add_scalar("charts/episodic_length_std_eval", std_length_eval, global_step)

            # print eval stats in blue
            eval_str = f"{args.env_id:<10} (seed={args.seed:^1} gpu={args.gpu_id:^1}) | step: {global_step:^10,d} | episodic return: {mean_return_eval:6.2f} +/- {std_return_eval:<6.2f} | episodic length: {mean_length_eval:6.2f} +/- {std_length_eval:<6.2f}"
            print("\033[94m" + eval_str + "\033[0m")


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
