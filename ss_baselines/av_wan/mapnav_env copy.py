#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from typing import Optional
import logging
import sys
sys.path.append("/media/kemove/data/av_nav/network/audionet")
sys.path.append("/media/kemove/data/av_nav/utlis")
from prob_update_doa import StreamingSourceMapFusion, align_for_occ
import numpy as np
import habitat
import torch
from habitat import Config, Dataset
from habitat.utils.visualizations.utils import observations_to_image
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.av_wan.models.planner import Planner
try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
except Exception:
    _gaussian_filter = None
try:
    from scipy.ndimage import binary_dilation as _binary_dilation
except Exception:
    _binary_dilation = None

def expand_to_ones(P, r=3, thr=0.0, connectivity=2):
    """
    把 P 的高值区域向外扩张 r 个像素，扩张后的区域全为 1，其余为 0。
    - r: 扩张半径（像素）
    - thr: 阈值，P > thr 作为种子
    - connectivity: 1=十字邻域(4-neigh), 2=全邻域(8-neigh)
    """
    seed = (P > thr)

    if r is None or r <= 0:
        return seed.astype(np.float32)

    if _binary_dilation is None:
        # 没有 scipy 时走纯 numpy 版本（见下面版本B）
        return expand_to_ones_numpy(P, r=r, thr=thr, connectivity=connectivity)

    # 构造结构元素：圆盘 / 正方形都可以。这里用圆盘更符合“半径”
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    if connectivity == 1:
        # 曼哈顿距离（菱形）
        footprint = (np.abs(xx) + np.abs(yy) <= r)
    else:
        # 欧氏距离（圆盘）
        footprint = (xx*xx + yy*yy <= r*r)

    out = _binary_dilation(seed, structure=footprint)
    return out.astype(np.float32)

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

def any_one_in_window(binary_map, cy, cx, r=3, require_all_one=False):
    """检查 (cy,cx) 周围 r 像素窗口内是否有 1。require_all_one=True 则要求窗口全为1"""
    H, W = binary_map.shape
    y0 = max(0, cy - r); y1 = min(H, cy + r + 1)
    x0 = max(0, cx - r); x1 = min(W, cx + r + 1)
    win = binary_map[y0:y1, x0:x1]
    if win.size == 0:
        return False
    if require_all_one:
        return np.all(win == 1)
    else:
        return np.any(win == 1)

def pick_frontier_target(planner,unexplored_mask,agent_pose):
    """
    最简单：在未探索区域里取一个“离当前位置较近”或“离当前位置较远”的点都行。
    这里用 argmax(unexplored) == 任意一点（全1），会偏向左上，不太好，
    更稳的是取“离当前位置最近的未探索点”。
    """
    ys, xs = np.where(unexplored_mask)
    if ys.size == 0:
        return None

    # 当前位置 map 坐标
    ax, ay = planner.mapper.world_to_map(agent_pose[0], agent_pose[1])
    ax = int(ax); ay = int(ay)

    d2 = (xs - ax) ** 2 + (ys - ay) ** 2
    k = int(np.argmin(d2))  # 最近未探索点
    return np.array([xs[k], ys[k]], dtype=np.int32)  # (x, y)

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
        self._visual_only_mode = False

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
        obs_bin = geometric_map[:, :, 0] > 0.5
        exp_bin = geometric_map[:, :, 1] > 0.5
        free_or_unknown = ~obs_bin
        unexplored = (~exp_bin) & free_or_unknown

        refiner     = kwargs["refiner"]
        vis_fuser   = kwargs["vis_fuser"]
        agent_pose  = kwargs["agent_pos"]
        use_visual  = kwargs["use_visual"]
        audio_intensity = kwargs["audio_intensity"]
        id_name         = kwargs["id_name"]
        save_vis       = kwargs["save_vis"]
        goal           = None
        goal_is_set    = False
        sound_map          = refiner.P
        sound_map_rotate   = align_for_occ(sound_map.T)
        sound_map_refine   = (1-obs) * sound_map_rotate
        sound_map          = align_for_occ(sound_map_refine).T
        exp = geometric_map[:, :, 1] > 0.5  

        if use_visual:
            vis_map            =  expand_to_ones(vis_fuser.P)
            if audio_intensity==0:
                if not self._visual_only_mode:
                    fused = ((vis_map + 1e-6)*0.4 + 0.6*(sound_map + 1e-6))
                    refiner.P          = fused
                # else:
                #     # if vis_map.sum() == 0:
                #     #     goal = pick_frontier_target(self.planner,unexplored,agent_pose)
                #     #     goal_is_set = True
                #     # else:
                #     refiner.P          = vis_map
                #     goal_is_set    = False

            else:
                refiner.P          = sound_map

        world_goal,map_goal      = refiner._readout(agent_pose[0],agent_pose[1])
        refiner.P                = sound_map

        if not goal_is_set:
            goal = self.planner.mapper.world_to_map(world_goal[0],world_goal[1])

        if use_visual:
            near_vis_ok = any_one_in_window(vis_map, cy=map_goal[0], cx=map_goal[1], r=5, require_all_one=False)
            distance_ok = np.linalg.norm(world_goal - agent_pose) <=0.5
            stop = bool(distance_ok and near_vis_ok)

            if (not stop) and distance_ok:
                self._visual_only_mode = True
        else:
            stop = np.linalg.norm(world_goal - agent_pose) <= 0.5
            near_vis_ok = True

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

            action = self.planner.plan_world(observation, goal_world=world_goal, stop=stop,id_name=id_name,save_vis=save_vis,source=target_goal,near_vis_ok=near_vis_ok)

            while action in (HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT):
                observation, reward, done, info = super().step({"action": action})

                if len(self._config.VIDEO_OPTION) > 0:
                    if "rgb" not in observation:
                        observation["rgb"] = np.zeros((self.config.DISPLAY_RESOLUTION,
                                                    self.config.DISPLAY_RESOLUTION, 3))
                    frame = observations_to_image(observation, info)
                    rgb_frames.append(frame)
                    audios.append(observation['audiogoal'])

                cumulative_reward += reward

                if done:
                    self.planner.reset()
                    observation = self.reset()
                    break  # 跳出 while(turn)；下面也会 break 外层 for
                else:
                    self.planner.update_map_and_graph(observation)

                    # reaching intermediate goal（如果你希望 turn 也能触发到达判断）
                    x, y = self.planner.mapper.get_maps_and_agent_pose()[2:4]
                    if (x - goal[0]) == (y - goal[1]) == 0:
                        reaching_waypoint = True
                        break

                n_turns += 1
                if n_turns >= max_turns:
                    break

                action = self.planner.plan_world(observation, goal_world=world_goal, stop=stop,id_name=id_name,save_vis=save_vis,source=target_goal,near_vis_ok=near_vis_ok)
            # 如果 turn-loop 里已经 done 或 reaching_waypoint，就结束本次 interval
            if done or reaching_waypoint:
                break

            # --------- 执行最终非 TURN 的动作（期望是 MOVE_FORWARD） ---------
            observation, reward, done, info = super().step({"action": action})

            if len(self._config.VIDEO_OPTION) > 0:
                if "rgb" not in observation:
                    observation["rgb"] = np.zeros((self.config.DISPLAY_RESOLUTION,
                                                self.config.DISPLAY_RESOLUTION, 3))
                frame = observations_to_image(observation, info)
                rgb_frames.append(frame)
                audios.append(observation['audiogoal'])

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

        
        # for step_count in range(self._config.PREDICTION_INTERVAL):
        #     if step_count != 0 and not self.planner.check_navigability(goal):
        #         cant_reach_waypoint = True
        #         break
        #     # action = self.planner.plan(observation, waypoint_map, stop=stop)
        #     action = self.planner.plan_world(observation, goal_world=world_goal, stop=stop,id_name=id_name,save_vis=save_vis,source=target_goal)
        #     observation, reward, done, info = super().step({"action": action})
        #     if len(self._config.VIDEO_OPTION) > 0:
        #         if "rgb" not in observation:
        #             observation["rgb"] = np.zeros((self.config.DISPLAY_RESOLUTION,
        #                                            self.config.DISPLAY_RESOLUTION, 3))
        #         frame = observations_to_image(observation, info)
        #         rgb_frames.append(frame)
        #         audios.append(observation['audiogoal'])
        #     cumulative_reward += reward
        #     if done:
        #         self.planner.reset()
        #         observation = self.reset()
        #         break
        #     else:
        #         self.planner.update_map_and_graph(observation)
        #         # reaching intermediate goal
        #         x, y = self.planner.mapper.get_maps_and_agent_pose()[2:4]
        #         if (x - goal[0]) == (y - goal[1]) == 0:
        #             reaching_waypoint = True
        #             break

        if not done:
            self.planner.add_maps_to_observation(observation)
        self._previous_observation = observation
        info['reaching_waypoint'] = done or reaching_waypoint
        info['cant_reach_waypoint'] = cant_reach_waypoint
        if len(self._config.VIDEO_OPTION) > 0:
            assert len(rgb_frames) != 0
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


