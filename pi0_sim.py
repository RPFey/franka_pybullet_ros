import os
import os.path as osp
import json
import argparse
import time
import datetime
import tqdm
import numpy as np
import cv2
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import logging
import matplotlib.pyplot as plt

from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

from franka_env.mp_wrapper import FrankaClutter

logger = logging.getLogger("curobo")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(osp.join('./', "curobo.log"))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

logger_fn = logger.info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep_file", type=str, default=None)
    parser.add_argument("--ep_root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viz", action="store_true", default=False)
    opt = parser.parse_args()
    logger_fn("Downloading checkpoint ...")

    config = config.get_config("pi0_fast_droid")
    checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_droid")
    logger_fn("Checkpoint downloaded")

    # Create a trained policy.
    logger_fn("Creating policy ...")
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    logger_fn("Policy created")
    
    with open(opt.ep_file, "r") as f:
        episode = json.load(f)
    
    seed = episode["seed"]
    sdf_file = os.path.join(opt.ep_root, episode["env"])
    env = FrankaClutter(sdf_file, False, seed=episode["seed"], gui=opt.viz, logdir='./')
    logger_fn("Environment created")

    rgb, rgb_left, depth = env.get_image()
    while rgb.max() == 0:
        time.sleep(1)
        logger_fn("Waiting for the camera to start")
        rgb, rgb_left, depth = env.get_image()
    
    plt.figure()
    plt.imshow(rgb)
    plt.savefig("rgb.png")
    plt.close()
    
    
    while True:
        instruction = input("Enter instruction: ")

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        bar = tqdm.tqdm(range(2000))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:
                
                rgb, rgb_left, depth = env.get_image()
                joint_state = env.get_joint_data()
                video.append(rgb_left)

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= 8:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    example = {
                        "observation/wrist_image_left": rgb.astype(np.float32) / 255.,
                        "observation/exterior_image_1_left": rgb_left.astype(np.float32) / 255.,
                        "observation/joint_position": joint_state[0, :7],
                        "observation/gripper_position": joint_state[0, [7]],
                        "prompt": "pick up the yellow banana"
                    }
                    
                    pred_action_chunk = policy.infer(example)["actions"]

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                    env.open_gripper()
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])
                    env.close_gripper()

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)
                env.set_joint_input(action[:7])

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
            
            
            except KeyboardInterrupt:
                break

        video = np.stack(video)
        video = video[:, :, :, ::-1]
        save_filename = "video.mp4"
        
        # write video to disk
        logger_fn(f"Saving video to {save_filename}")
        video_writer = cv2.VideoWriter(save_filename, cv2.VideoWriter_fourcc(*"mp4v"), 30, (video.shape[2], video.shape[1]))
        for i in range(len(video)):
            video_writer.write(video[i])

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
    
    env.end()
