import argparse
import os
import numpy as np
import torch
import gymnasium as gym

from actor import Actor
# Import the specific run function we fixed earlier
from train import run_lunar_lander
from utils import ObservationNormalizer, load_config, set_random_seed

def load_actor_from_checkpoint(config, checkpoint_path):
    """Loads the actor and its corresponding normalizer from the saved .pt file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    # Load with weights_only=False to allow loading the custom normalizer and config dict
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dim  = 8
    action_dim = 4
    hidden_dim = 512 
    
    print(f"Loading model: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")
    # Initialize Actor with the CORRECT hidden_dim
    actor = Actor(state_dim, action_dim, hidden_dim=hidden_dim)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    
    # Load the exact Normalizer state (mean/std) used during the best training phase
    obs_normalizer = ObservationNormalizer(state_dim)
    if "obs_normalizer_state" in checkpoint:
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
        print("Successfully loaded ObservationNormalizer stats.")
    else:
        print("Warning: No normalizer found in checkpoint. Eval results will likely be poor.")
    
    # Attach normalizer to actor so get_action/eval functions can use it
    actor.obs_normalizer = obs_normalizer
    
    return actor

def evaluate_actor(actor, config, num_episodes):
    """Performs deterministic evaluation."""
    env = gym.make(config["env_id"]) 
    rewards = []

    print(f"Starting evaluation over {num_episodes} episodes...")

    try:
        for ep in range(num_episodes):
            # Using gymnasium style reset
            state, _ = env.reset(seed=config["random_seed"] + 1000 + ep)
            total_reward = 0.0
            done = False

            while not done:
                # 1. Normalize using the SAVED stats from training
                obs_norm = actor.obs_normalizer.normalize(state) if hasattr(actor, 'obs_normalizer') else state
                state_tensor = torch.tensor(obs_norm, dtype=torch.float32)
                
                # 2. Predict best action (deterministic=True)
                with torch.no_grad():
                    action = actor.get_action(state_tensor, deterministic=True)
                
                # 3. Step environment
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

            rewards.append(total_reward)
            if (ep + 1) % 10 == 0 or num_episodes <= 20:
                print(f"  Episode {ep+1}: {total_reward:.2f}")
                
    finally:
        env.close()

    return float(np.mean(rewards)), float(np.std(rewards))

def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved LunarLander policy.")
    parser.add_argument("--config", default="config.json", help="Path to config.")
    parser.add_argument("--checkpoint", default="./checkpoints/best_actor_critic.pt", help="Path to .pt file.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes.")
    parser.add_argument("--video", default="eval_landing.mp4", help="Video filename.")
    
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = args.checkpoint
    eval_episodes = args.episodes

    try:
        # Load the Best Actor
        actor = load_actor_from_checkpoint(config, checkpoint_path)
        
        # 1. Run the 500-episode Evaluation for the Deliverable
        mean_reward, std_reward = evaluate_actor(actor, config, eval_episodes)
        print("-" * 30)
        print(f"FINAL EVALUATION RESULTS (n={eval_episodes})")
        print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print("-" * 30)

        # 2. Video Generation (deterministic)
        if args.video:
            print(f"Generating best-policy video: {args.video}...")
            run_lunar_lander(actor, args.video, config=config)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()