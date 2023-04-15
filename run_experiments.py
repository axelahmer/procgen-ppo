import argparse
import multiprocessing
import subprocess
import procgen
from itertools import product

# Define the default environment IDs
default_env_ids = procgen.env.ENV_NAMES

# Define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0],
                    help='List of GPU IDs to use')
parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                    help='List of seeds to use')
parser.add_argument('--env-ids', type=str, nargs='+', default=default_env_ids,
                    help='List of environment IDs to use')
parser.add_argument('--agents', type=str, nargs='+', default=['impala'],
                    help='List of agents to use')
parser.add_argument("--ent-coef", type=float, default=0.01,
                    help="coefficient of the entropy")
parser.add_argument('--exp-name', type=str, required=True,
                    help='Name of the experiment')

# Parse the arguments
args = parser.parse_args()

gpu_ids = args.gpu_ids
seeds = args.seeds
env_ids = args.env_ids
agents = args.agents
exp_name = args.exp_name


# Function to run ppo_procgen.py with the supplied flags and specified GPU
def run_ppo_procgen(args):
    seed, env_id, agent, name, gpu_id = args
    cmd = f"python ppo_procgen.py --seed {seed} --env-id {env_id} --agent {agent} --exp-name {name} --gpu-id {gpu_id}"
    subprocess.run(cmd, shell=True)

# Create combinations of seeds and environment IDs
combinations = list(product(seeds, env_ids, agents, [exp_name]))

# divide the combinations into queues, one for each GPU, and add the GPU ID to each combination
queues = [list() for _ in gpu_ids]
for i, combination in enumerate(combinations):
    queues[i % len(gpu_ids)].append(combination + (gpu_ids[i % len(gpu_ids)],))

def run_gpu_queue(queue):
    for args in queue:
        run_ppo_procgen(args)

# Create a process for each GPU queue
processes = []
for queue in queues:
    process = multiprocessing.Process(target=run_gpu_queue, args=(queue,))
    process.start()
    processes.append(process)

# Wait for all processes to finish
for process in processes:
    process.join()
