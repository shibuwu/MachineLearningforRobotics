import os
import sys

# Step 1: Tell Python to look two levels up to find the 'helpers' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.multiprocessing as mp

from helpers.config import load_config
from helpers.logger import A3CLogger
from helpers.utils import get_kuka_action_dim
from lib.a3c.agent import worker_process
from lib.a3c.model import ActorCritic
from lib.a3c.shared_optim import SharedAdam


def build_global_model(config, device):
    action_dim = get_kuka_action_dim(config)
    model = ActorCritic(
        state_size=config["network"]["state_size"],
        action_size=action_dim,
        shared_layers=config["network"]["shared_layers"],
        critic_hidden_layers=config["network"]["critic_hidden_layers"],
        actor_hidden_layers=config["network"]["actor_hidden_layers"],
        init_type=config["network"]["init_type"],
        seed=0,
    ).to(device)
    model.share_memory()

    return model


def save_final_checkpoint(global_net, optimizer, config):
    model_path = os.path.join(
        config["logging"]["model_dir"], "a3c_kuka_model_final.pth"
    )
    torch.save(
        {
            "model_state_dict": global_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": config["hyperparameters"]["max_episodes"],
        },
        model_path,
    )
    return model_path


def train_a3c():
    config = load_config()
    device = torch.device(config["device"])
    logger = A3CLogger(config)

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # already set

    global_net = build_global_model(config, device)
    optimizer = SharedAdam(global_net.parameters(), lr=config["hyperparameters"]["lr"])
    optimizer.share_memory()
    global_ep = mp.Value("i", 0)
    lock = mp.Lock()
    manager = mp.Manager()
    shared_stats = manager.dict({
        "rewards": manager.list(),
        "lengths": manager.list(),
        "losses": manager.list(),
    })

    os.makedirs(config["logging"]["model_dir"], exist_ok=True)

    logger.info("Starting A3C training for Kuka pick and place task...")
    logger.info(
        f"Using {config['hyperparameters']['num_workers']} workers on {device}"
    )

    processes = []
    for worker_id in range(config["hyperparameters"]["num_workers"]):
        p = mp.Process(
            target=worker_process,
            args=(
                worker_id, global_net, optimizer, global_ep,
                config["hyperparameters"]["max_episodes"],
                lock, config, device, shared_stats, logger.log_path,
            ),
        )

        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    model_path = save_final_checkpoint(global_net, optimizer, config)
    manager.shutdown()
    logger.info(f"Final model saved to {model_path}. Training complete!")
    logger.close()
