#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from typing import Optional
import logging
import sys
import cv2
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
    from scipy.ndimage import binary_dilation,convolve
except Exception:
    _binary_dilation = None

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
    k = int(np.argmax(d2))  # 最近未探索点
    return np.array([xs[k], ys[k]], dtype=np.int32)  # (x, y)

class FrontierBlacklist:
    def __init__(self):
        self.ttl = {}  # (x,y) -> ttl

    def tick(self):
        if not self.ttl:
            return
        for k in list(self.ttl.keys()):
            self.ttl[k] -= 1
            if self.ttl[k] <= 0:
                del self.ttl[k]

    def add(self, xy, ttl=80):
        if xy is None:
            return
        k = (int(xy[0]), int(xy[1]))
        self.ttl[k] = ttl

    def contains(self, xy):
        if xy is None:
            return False
        k = (int(xy[0]), int(xy[1]))
        return k in self.ttl

    def __len__(self):
        return len(self.ttl)

def pick_frontier_target_dt(
    geometric_map,
    planner,
    agent_pose,
    min_dist_px=10,
    prefer_far=True,
    blacklist=None,
    topk_check=300,
    thr_obs=0.5,
    thr_exp=0.5,
):
    """
    返回 goal_xy (x,y) in planner map coords, 或 None
    """
    obs_bin = geometric_map[:, :, 0] > thr_obs
    exp_bin = geometric_map[:, :, 1] > thr_exp

    free = exp_bin & (~obs_bin)
    unknown = (~exp_bin) & (~obs_bin)

    if not np.any(unknown):
        return None

    # frontier = unknown 且邻近 free
    near_free = cv2.dilate(free.astype(np.uint8), np.ones((3, 3), np.uint8), 1).astype(bool)
    frontier = unknown & near_free
    if not np.any(frontier):
        return None

    # distance transform：越大表示越“深入未知/离free越远”
    # cv2.distanceTransform 输入：0 表示障碍/背景，非0 表示前景。
    # 我们要算到 free 的距离：令 free=0, 非free=1
    inv_free = (~free).astype(np.uint8)
    dt = cv2.distanceTransform(inv_free, distanceType=cv2.DIST_L2, maskSize=3)

    ax, ay = planner.mapper.world_to_map(agent_pose[0], agent_pose[1])
    ax, ay = int(ax), int(ay)

    fy, fx = np.where(frontier)
    if fy.size == 0:
        return None

    d = np.sqrt((fx - ax) ** 2 + (fy - ay) ** 2)
    keep = d >= float(min_dist_px)
    if np.any(keep):
        fy, fx, d = fy[keep], fx[keep], d[keep]
    if fy.size == 0:
        return None

    score = dt[fy, fx].astype(np.float32)
    if prefer_far:
        score = score + 0.02 * d.astype(np.float32)  # 轻微鼓励走远，覆盖更快

    order = np.argsort(-score)  # 大->小

    # 无黑名单：直接取最好
    if blacklist is None or len(blacklist) == 0:
        idx = int(order[0])
        return np.array([int(fx[idx]), int(fy[idx])], dtype=np.int32)

    # 有黑名单：在 topk 中找第一个不在黑名单的
    K = min(int(topk_check), int(order.size))
    for j in range(K):
        idx = int(order[j])
        x, y = int(fx[idx]), int(fy[idx])
        if not blacklist.contains((x, y)):
            return np.array([x, y], dtype=np.int32)

    return None

def need_repick_goal(
    planner,
    agent_pose,
    goal_world,
    reached_eps_m=0.5,
    stuck_eps_m=0.05,
    stuck_steps=20,
    state=None,
):
    """
    state: dict，用于跨 step 保存 last_dist / stuck_count / fail_count
    返回: (need_repick: bool, reached: bool, failed: bool, dist: float)
    """
    if state is None:
        state = {}

    # --- dist / reached ---
    dist = float(np.linalg.norm(np.array(goal_world, dtype=np.float32) - np.array(agent_pose, dtype=np.float32)))
    reached = (dist <= reached_eps_m)

    if reached:
        state["last_dist"] = None
        state["stuck_count"] = 0
        state["fail_count"] = 0
        return True, True, False, dist

    # --- failed: not navigable ---
    # 如果你没有 check_navigability，可换成 planner.plan_world 的失败信号
    navigable = True
    if hasattr(planner, "check_navigability"):
        navigable = bool(planner.check_navigability(goal_world))

    if not navigable:
        state["fail_count"] = int(state.get("fail_count", 0)) + 1
        # 连续多次不可达才判失败（防抖）
        if state["fail_count"] >= 2:
            state["last_dist"] = None
            state["stuck_count"] = 0
            state["fail_count"] = 0
            return True, False, True, dist
    else:
        state["fail_count"] = 0

    # --- stuck: dist 没下降 ---
    last_dist = state.get("last_dist", None)
    if last_dist is None:
        state["last_dist"] = dist
        state["stuck_count"] = 0
        return False, False, False, dist

    if (last_dist - dist) < stuck_eps_m:
        state["stuck_count"] = int(state.get("stuck_count", 0)) + 1
    else:
        state["stuck_count"] = 0

    state["last_dist"] = dist

    if state["stuck_count"] >= stuck_steps:
        state["last_dist"] = None
        state["stuck_count"] = 0
        return True, False, True, dist

    return False, False, False, dist

class FastFrontierExplorer:
    """
    用法：
      explorer = FastFrontierExplorer()
      每个 step：
        world_goal, goal_map, stop = explorer.step(geometric_map, planner, agent_pose, enable=True)
    """
    def __init__(self):
        self.goal_map = None  # (x,y) in planner map
        self.blacklist = FrontierBlacklist()
        self._repick_state = {"last_dist": None, "stuck_count": 0, "fail_count": 0}

    def reset(self):
        self.goal_map = None
        self._repick_state = {"last_dist": None, "stuck_count": 0, "fail_count": 0}
        # blacklist 不一定要清，通常保留更好

    def step(
        self,
        geometric_map,
        planner,
        agent_pose,
        enable=True,
        reached_eps_m=0.5,
        min_dist_px=10,
        prefer_far=True,
        blacklist_ttl=80,
    ):
        """
        返回：
          world_goal: np.array([x,z], float32) or None
          goal_map: np.array([x,y], int32) or None
          stop: bool
        """
        self.blacklist.tick()

        if not enable:
            # 不启用探索：不提供目标
            return None, None, False

        # 如果没有目标：pick一次
        if self.goal_map is None:
            self.goal_map = pick_frontier_target_dt(
                geometric_map, planner, agent_pose,
                min_dist_px=min_dist_px,
                prefer_far=prefer_far,
                blacklist=self.blacklist
            )
            if self.goal_map is None:
                return np.array(agent_pose, dtype=np.float32), None, True  # 没有frontier了，停

        # 将 map goal 转到 world goal
        world_goal = planner.mapper.map_to_world(int(self.goal_map[0]), int(self.goal_map[1]))
        world_goal = np.array(world_goal, dtype=np.float32)

        need_repick, reached, failed, dist = need_repick_goal(
            planner=planner,
            agent_pose=agent_pose,
            goal_world=world_goal,
            reached_eps_m=reached_eps_m,
            state=self._repick_state,
        )

        if need_repick:
            if failed:
                # 失败点进黑名单，避免反复选同一失败入口
                self.blacklist.add(self.goal_map, ttl=blacklist_ttl)

            # 重新 pick（只在到达/失败/卡住发生）
            self.goal_map = pick_frontier_target_dt(
                geometric_map, planner, agent_pose,
                min_dist_px=min_dist_px,
                prefer_far=prefer_far,
                blacklist=self.blacklist
            )

            if self.goal_map is None:
                # 没有frontier了：探索完成或被障碍隔绝
                return np.array(agent_pose, dtype=np.float32), None, True

            world_goal = planner.mapper.map_to_world(int(self.goal_map[0]), int(self.goal_map[1]))
            world_goal = np.array(world_goal, dtype=np.float32)

        stop = (float(np.linalg.norm(world_goal - np.array(agent_pose, dtype=np.float32))) <= reached_eps_m)
        return world_goal, self.goal_map, bool(stop)


def snap_to_multiple(x, step=10):
    return int(round(x / step) * step)

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
        self._explorer = FastFrontierExplorer()
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
        geometric_map, _,x,y,_ = self.planner.mapper.get_maps_and_agent_pose()
        obs = geometric_map[:, :, 0] > 0.5   # obstacle
        obs_bin = geometric_map[:, :, 0] > 0.5
        exp_bin = geometric_map[:, :, 1] > 0.5
        free_or_unknown = ~obs_bin
        unexplored = (~exp_bin) & free_or_unknown

        free      = exp_bin & (~obs_bin)
        occupied  = exp_bin & obs_bin
        unknown   = ~exp_bin
        frontier = unknown & binary_dilation(free)


        refiner     = kwargs["refiner"]
        vis_fuser   = kwargs["vis_fuser"]
        agent_pose  = kwargs["agent_pos"]
        use_visual  = kwargs["use_visual"]
        audio_intensity = kwargs["audio_intensity"]
        id_name         = kwargs["id_name"]
        save_vis       = kwargs["save_vis"]
        step           = kwargs["step"]
        goal_is_set    = False
        sound_map      = refiner.P

        if use_visual:
            r = 10
            vis_map = vis_fuser.P
            vis_map_gaussian            =  cv2.dilate(vis_map, np.ones((2*r+1, 2*r+1), np.uint8), 1).astype(np.float32)
            weights_visual = 0.6
            if step>=250:
                weights_visual = 0.1
            if step>=400:
                weights_visual = 0
            if audio_intensity==0:
                if not self._visual_only_mode:
                    fused = ((vis_map_gaussian + 1e-6)*0.4 + weights_visual*(sound_map + 1e-6))
                    refiner.P          = fused
                else:
                    if vis_map.sum() == 0:
                        world_goal, goal, stop = self._explorer.step(
                            geometric_map=geometric_map,
                            planner=self.planner,
                            agent_pose=agent_pose,
                            reached_eps_m=0.5,
                            min_dist_px=10,
                            prefer_far=True,
                        )
                        goal_is_set = True
                        goal = np.array([
                            snap_to_multiple(goal[0], 10),
                            snap_to_multiple(goal[1], 10),
                        ], dtype=np.int32)

                        print(goal)
                    else:
                        refiner.P          = vis_map
                        goal_is_set        = False
            else:
                refiner.P          = sound_map

        if not goal_is_set:
            world_goal,map_goal      = refiner._readout(agent_pose[0],agent_pose[1])
            goal = self.planner.mapper.world_to_map(world_goal[0],world_goal[1])
            refiner.P                = sound_map
        else:
            world_goal = self.planner.mapper.map_to_world(goal[0],goal[1])
            map_goal   = refiner.map_to_vu(world_goal[0],world_goal[1])
            # refiner.P                = sound_map

        if use_visual:
            near_vis_ok = any_one_in_window(vis_map, cy=map_goal[0], cx=map_goal[1], r=20, require_all_one=False)
            distance_ok = np.linalg.norm(world_goal - agent_pose) <=0.5
            if (not near_vis_ok) and distance_ok:
                self._visual_only_mode = True
        else:
            near_vis_ok = True

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
            if self._env.sim._episode_step_count==500:
                action = HabitatSimActions.STOP
            else:
                action = self.planner.plan_world(observation, goal_world=world_goal, stop=stop,id_name=id_name,save_vis=save_vis,source=target_goal)
            # action = self.planner.plan_world(observation, goal_world=world_goal, stop=stop,id_name=id_name,save_vis=save_vis,source=target_goal,near_vis_ok=near_vis_ok)
            if action ==HabitatSimActions.STOP:
                print(f"{id_name} final distance to goal: {np.linalg.norm(target_goal - agent_pose)}")
                
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

                if self._env.sim._episode_step_count==500:
                    action = HabitatSimActions.STOP
                else:
                    action = self.planner.plan_world(observation, goal_world=world_goal, stop=stop,id_name=id_name,save_vis=save_vis,source=target_goal)
    
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


