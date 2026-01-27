#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from typing import Optional
import logging
import sys
import os
sys.path.append("/media/kemove/data/av_nav/network/audionet")
sys.path.append("/media/kemove/data/av_nav/utlis")
from prob_update_doa import StreamingSourceMapFusion, align_for_occ
import numpy as np
import habitat
import torch
from habitat import Config, Dataset
# from habitat.utils.visualizations.utils import observations_to_image
from ss_baselines.common.utils import observations_to_image


from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.av_wan.models.planner import Planner
try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
except Exception:
    _gaussian_filter = None

def gaussian_smooth(P, sigma):
    if sigma is None or sigma <= 0:
        return P
    if _gaussian_filter is not None:
        return _gaussian_filter(P, sigma=sigma, mode="nearest")

    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x**2) / (2 * (sigma**2) + 1e-12))
    k = k / (k.sum() + 1e-12)

    def conv1d_axis(A, axis):
        pad = [(0, 0)] * A.ndim
        pad[axis] = (radius, radius)
        Ap = np.pad(A, pad, mode="edge")
        out = np.zeros_like(A, dtype=np.float32)

        if axis == 1:  # x
            for i in range(A.shape[1]):
                out[:, i] = (Ap[:, i:i + 2 * radius + 1] * k[None, :]).sum(axis=1)
        else:  # y
            for i in range(A.shape[0]):
                out[i, :] = (Ap[i:i + 2 * radius + 1, :] * k[:, None]).sum(axis=0)
        return out

    A = A.astype(np.float32)
    A = conv1d_axis(A, axis=1)
    A = conv1d_axis(A, axis=0)
    return A


@baseline_registry.register_env(name="MapNavEnv")
class MapNavEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._config = config
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG

        self._previous_target_distance = None
        self._previous_action = None
        self._previous_observation = None
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS.SUCCESS_DISTANCE
        super().__init__(self._core_env_config, dataset)

        self.planner = Planner(model_dir=self._config.MODEL_DIR,
                               use_acoustic_map='ACOUSTIC_MAP' in config.TASK_CONFIG.TASK.SENSORS,
                               masking=self._config.MASKING,
                               task_config=config.TASK_CONFIG
                               )
        torch.set_num_threads(1)

    def reset(self):
        self._previous_action = None

        observations = super().reset()
        self.planner.update_map_and_graph(observations)
        self.planner.add_maps_to_observation(observations)
        self._previous_observation = observations
        logging.debug(super().current_episode)

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        return observations

    def step(self, *args, **kwargs):
        # world_goal = kwargs["action"]
        target_goal = kwargs["target_goal"]
        geometric_map, _,_,_,_ = self.planner.mapper.get_maps_and_agent_pose()
        obs = geometric_map[:, :, 0] > 0.5   # obstacle
        refiner     = kwargs["refiner"]
        vis_fuser   = kwargs["vis_fuser"]
        agent_pose  = kwargs["agent_pos"]
        use_visual  = kwargs["use_visual"]
        audio_intensity = kwargs["audio_intensity"]
        id_name         = kwargs["id_name"]
        save_vis       = kwargs["save_vis"]
        step           = kwargs["step"]


        sound_map          = refiner.P
        sound_map_rotate   = align_for_occ(sound_map.T)
        sound_map_refine   = (1-obs) * sound_map_rotate
        sound_map          = align_for_occ(sound_map_refine).T
        sound_map          = sound_map
        sound_bounds = (
                    (refiner.x_min - 0.5*refiner.res, 0.0, refiner.z_min - 0.5*refiner.res),
                    (refiner.x_max + 0.5*refiner.res, 0.0, refiner.z_max + 0.5*refiner.res),
                )
        if use_visual:
            sound_map_gaussian =sound_map
            # exp = geometric_map[:, :, 1] > 0.5  
            vis_map            = vis_fuser.P
            if vis_map is None or np.max(vis_map) <= 0:
                fused_sound_map    = sound_map
                refiner.P          = fused_sound_map
            else:
                fused_sound_map = ((vis_map + 1e-6)*0.2 + 0.8*(sound_map_gaussian + 1e-6))
                refiner.P          = fused_sound_map
        else:
            fused_sound_map = sound_map
            refiner.P          = fused_sound_map

        world_goal,_       = refiner._readout(agent_pose[0],agent_pose[1])
        refiner.P          = sound_map

        goal = self.planner.mapper.world_to_map(world_goal[0],world_goal[1])

        stop = np.linalg.norm(world_goal - agent_pose) <=0.5
        observation = self._previous_observation
        cumulative_reward = 0
        done = False
        reaching_waypoint = False
        cant_reach_waypoint = False
        if len(self._config.VIDEO_OPTION) > 0:
            rgb_frames = list()
            audios = list()

        for step_count in range(self._config.PREDICTION_INTERVAL):
            if step_count != 0 and not self.planner.check_navigability(goal):
                cant_reach_waypoint = True
                break

            max_turns = 4
            n_turns = 0
            if self._env.sim._episode_step_count==499:
                action = HabitatSimActions.STOP
            else:
                action = self.planner.plan_world(observation, goal_world=world_goal, stop=stop,id_name=id_name,save_vis=save_vis,source=target_goal)
            # print(self._env.sim._episode_step_count,self._env.task.is_stop_called,action,self._episode_success(),self._env.sim.reaching_goal)
            while action in (HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT):
                observation, reward, done, info = super().step({"action": action})
                if 'intermediate' in observation:
                    for observation_rgb in observation['intermediate']:
                        frame = observations_to_image(observation_rgb, info,fused_sound_map,sim=self._env.sim,sound_bounds=sound_bounds,goal_position=target_goal,pred_position=world_goal)
                        rgb_frames.append(frame)
                        audios.append(observation['audiogoal'])
                    del observation['intermediate']
                    
                
                # save_dir = "/home/Disk/yyz/sound-spaces/vis/debug_npz_ral"
                # save_dir = os.path.join(save_dir,id_name)
                # os.makedirs(save_dir, exist_ok=True)
                # np.savez_compressed(
                #     os.path.join(save_dir, f"{self._env.sim._episode_step_count}.npz"),
                #     agent_pos=np.array([agent_pose[0],agent_pose[-1]])
                # )
                
                cumulative_reward += reward

                if done:
                    self.planner.reset()
                    observation = self.reset()
                    break 
                else:
                    self.planner.update_map_and_graph(observation)
                    x, y = self.planner.mapper.get_maps_and_agent_pose()[2:4]
                    if (x - goal[0]) == (y - goal[1]) == 0:
                        reaching_waypoint = True
                        break

                n_turns += 1
                if n_turns >= max_turns:
                    break

                if self._env.sim._episode_step_count==499:
                    action = HabitatSimActions.STOP
                else:
                    action = self.planner.plan_world(observation, goal_world=world_goal, stop=stop,id_name=id_name,save_vis=save_vis,source=target_goal)
                    # action = HabitatSimActions.MOVE_FORWARD

                # print(self._env.sim._episode_step_count,self._env.task.is_stop_called,action,self._episode_success(),self._env.sim.reaching_goal)

            if done or reaching_waypoint:
                break

            observation, reward, done, info = super().step({"action": action})
            if 'intermediate' in observation:
                for observation_rgb in observation['intermediate']:
                    frame = observations_to_image(observation_rgb, info,fused_sound_map,sim=self._env.sim,sound_bounds=sound_bounds,goal_position=target_goal,pred_position=world_goal)
                    rgb_frames.append(frame)
                    # print(len(rgb_frames))
                    audios.append(observation['audiogoal'])
                del observation['intermediate']


            cumulative_reward += reward

            if done:
                self.planner.reset()
                observation = self.reset()
                break
            else:
                self.planner.update_map_and_graph(observation)

                x, y = self.planner.mapper.get_maps_and_agent_pose()[2:4]
                if (x - goal[0]) == (y - goal[1]) == 0:
                    reaching_waypoint = True
                    break

        
        if not done:
            self.planner.add_maps_to_observation(observation)
        self._previous_observation = observation
        info['reaching_waypoint'] = done or reaching_waypoint
        info['cant_reach_waypoint'] = cant_reach_waypoint
        if len(self._config.VIDEO_OPTION) > 0:
            # assert len(rgb_frames) != 0
            if len(rgb_frames) != 0:
                info['rgb_frames'] = rgb_frames
                info['audios'] = audios

        return observation, cumulative_reward, done, info

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = 0

        if self._rl_config.WITH_TIME_PENALTY:
            reward += self._rl_config.SLACK_REWARD

        if self._rl_config.WITH_DISTANCE_REWARD:
            current_target_distance = self._distance_target()
            # if current_target_distance < self._previous_target_distance:
            reward += (self._previous_target_distance - current_target_distance) * self._rl_config.DISTANCE_REWARD_SCALE
            self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
            logging.debug('Reaching goal!')
        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = [goal.position for goal in self._env.current_episode.goals]
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance

    def _episode_success(self):
        if (
                self._env.task.is_stop_called
                # and self._distance_target() < self._success_distance
                and self._env.sim.reaching_goal
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    # for data collection
    def get_current_episode_id(self):
        return self.habitat_env.current_episode.episode_id

    def global_to_egocentric(self, pg):
        return self.planner.mapper.global_to_egocentric(*pg)

    def egocentric_to_global(self, pg):
        return self.planner.mapper.egocentric_to_global(*pg)