#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

import logging
import networkx as nx
import numpy as np
import torch
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import os
import matplotlib.pyplot as plt
from ss_baselines.av_wan.models.mapper import Mapper, to_array


class Planner:
    """
    Online planner that ONLY uses the accumulated occupancy map (Mapper.geometric_map),
    without requiring NavMesh.

    Supports:
      - plan(observation, goal=(gx,gy), stop=bool)
      - plan_world(observation, goal_world=(wx,wz), stop=bool)
    """

    def __init__(self, task_config=None, use_acoustic_map=False, model_dir=None, masking=True):
        self.mapper = Mapper(
            gm_config=task_config.TASK.GEOMETRIC_MAP,
            am_config=task_config.TASK.ACOUSTIC_MAP,
            action_map_config=task_config.TASK.ACTION_MAP,
            use_acoustic_map=use_acoustic_map
        )

        self._action_map_res = task_config.TASK.ACTION_MAP.MAP_RESOLUTION
        self._action_map_size = task_config.TASK.ACTION_MAP.MAP_SIZE

        self._prev_next_node = None
        self._prev_action = None

        self._obstacles = []
        self._obstacle_threshold = 0.5

        # navigable grid points in internal geometric map
        self._navigable_xs, self._navigable_ys = self.mapper.compute_navigable_xys()

        # initial graph from current geometric map
        self._graph = self._map_to_graph(self.mapper.get_maps_and_agent_pose()[0])

        # incrementally removed nodes/edges (for reset)
        self._removed_edges = list()
        self._removed_nodes = list()

        self._model_dir = model_dir
        self._masking = masking

        # --- stabilizers for 90-degree action space ---
        self._turn_bias = HabitatSimActions.TURN_LEFT  # when rotation==180, do not random
        self._lookahead = 8                            # 5~10 works well for 90-degree turning


        self.reset()

        self._seg_dir = None
        self._seg_steps_left = 0
        self._seg_goal = None
        self._seg_start = None
        self._prev_depth = None

        self._last_collided = False
        self._block_forward_steps = 0

        self._debug_dir = "/home/Disk/yyz/sound-spaces/debug_plan_test"
        os.makedirs(self._debug_dir, exist_ok=True)
        self._debug_step = 0
        self._debug_every = 1  # 每步都画；想降低开销可设 5/10

    # -----------------------------
    # lifecycle
    # -----------------------------
    def reset(self):
        self._prev_depth = None
        self._prev_next_node = None
        self._prev_action = None
        self._obstacles = []
        self.mapper.reset()

        # restore removed nodes/edges if any
        # NOTE: removed_nodes stores tuples (node, attr_dict)
        if len(self._removed_nodes) > 0:
            for n, attr in self._removed_nodes:
                self._graph.add_node(n, **attr)
        if len(self._removed_edges) > 0:
            self._graph.add_edges_from(self._removed_edges)

        self._removed_nodes.clear()
        self._removed_edges.clear()

        self._turn_bias = HabitatSimActions.TURN_LEFT

    # -----------------------------
    # map & graph update
    # -----------------------------
    def update_map_and_graph(self, observation):
        ego_map = to_array(observation['ego_map'])
        depth = to_array(observation['depth'])
        collided = to_array(observation['collision'][0])

        intensity = to_array(observation['intensity'][0]) if 'intensity' in observation else None

        geometric_map, acoustic_map, x, y, orientation = self.mapper.get_maps_and_agent_pose()

        if not collided:
            non_navigable_points, blocked_paths = self.mapper.update(self._prev_action, ego_map, intensity)
            self._update_graph(non_navigable_points, blocked_paths)
        elif self._prev_next_node in self._graph.nodes:
            # only the edge to the previous next node should be removed
            current_node = self._map_index_to_graph_nodes([(x, y)])[0]
            if self._graph.has_edge(self._prev_next_node, current_node):
                self._graph.remove_edge(self._prev_next_node, current_node)
                self._removed_edges.append((self._prev_next_node, current_node))

        elif collided:
            xg, yg = self._snap_to_navigable_grid(x, y)
            s = self.mapper._stride
            fx = xg + int(round(s * np.cos(np.deg2rad(orientation))))
            fy = yg + int(round(s * np.sin(np.deg2rad(orientation))))
            fx, fy = self._snap_to_navigable_grid(fx, fy)

            cur_node = self._map_index_to_graph_nodes([(xg, yg)])[0]
            fwd_node = self._map_index_to_graph_nodes([(fx, fy)])[0]

            if cur_node in self._graph and fwd_node in self._graph and self._graph.has_edge(cur_node, fwd_node):
                self._graph.remove_edge(cur_node, fwd_node)
                self._removed_edges.append((cur_node, fwd_node))


        self._prev_depth = depth

        if logging.root.level == logging.DEBUG:
            geometric_map, acoustic_map, x, y, orientation = self.mapper.get_maps_and_agent_pose()
            assert not geometric_map[y, x, 0]
            for node, attr in self._removed_nodes:
                index = attr['map_index']
                assert self.mapper._geometric_map[index[1], index[0]][0]

    def add_maps_to_observation(self, observation):
        if 'gm' in observation:
            observation['gm'] = self.mapper.get_egocentric_geometric_map().astype(np.float32)
        if 'am' in observation:
            observation['am'] = self.mapper.get_egocentric_acoustic_map().astype(np.float32)
        if 'action_map' in observation:
            observation['action_map'] = np.expand_dims(
                self.mapper.get_egocentric_occupancy_map(
                    size=self._action_map_size,
                    action_map_res=self._action_map_res
                ),
                -1
            ).astype(np.float32)

    # -----------------------------
    # core planning (GLOBAL MAP GOAL)
    # -----------------------------
    def plan(self, observation: dict, goal, stop, distribution=None,id_name='exp') -> torch.Tensor:
        """
        goal: (gx, gy) in GLOBAL MAP INDEX (internal geometric map coordinates)
        stop: bool
        """
        geometric_map, acoustic_map, x, y, orientation = self.mapper.get_maps_and_agent_pose()

        if stop:
            action = HabitatSimActions.STOP
            self._prev_next_node = None
            self._prev_action = action
            return action

        # 1) snap current & goal to the nearest navigable grid point (CRITICAL)
        xg, yg = self._snap_to_navigable_grid(x, y)
        gx, gy = int(goal[0]), int(goal[1])
        ggx, ggy = self._snap_to_navigable_grid(gx, gy)

        cur_node = self._node_id(xg, yg)
        tgt_node = self._node_id(ggx, ggy)

        # print(xg,yg,gx,gy)
        # if cur_node in self._graph and tgt_node in self._graph:
        #     print("has_path:", nx.has_path(self._graph, cur_node, tgt_node))

        # 2) ensure current node exists
        if cur_node not in self._graph:
            action = HabitatSimActions.TURN_LEFT
            self._prev_next_node = None
            self._prev_action = action
            return action

        # 3) if goal node not in graph / not reachable -> project to nearest reachable node
        if tgt_node not in self._graph or not nx.has_path(self._graph, cur_node, tgt_node):
            nn = self._nearest_reachable_node((ggx, ggy), cur_node)
            if nn is None:
                action = HabitatSimActions.TURN_LEFT
                self._prev_next_node = None
                self._prev_action = action
                return action
            tgt_node = nn

        # 4) shortest path
        try:
            path = nx.shortest_path(self._graph, source=cur_node, target=tgt_node)
        except (nx.exception.NetworkXNoPath, nx.exception.NodeNotFound):
            action = HabitatSimActions.TURN_LEFT
            self._prev_next_node = None
            self._prev_action = action
            return action

        if len(path) <= 1:
            action = HabitatSimActions.STOP
            self._prev_next_node = None
            self._prev_action = action
            return action

        lookahead = 10
        idx = min(lookahead, len(path) - 1)
        if idx < 1:
            idx = 1
        inter_node = path[idx]
        inter_xy = self._graph.nodes[inter_node]["map_index"]
        # print( self._graph.nodes[path[1]]["map_index"],x,y)

        # ===== debug draw =====
        if (self._debug_step % self._debug_every) == 0:
            self._plot_plan_debug(
                geometric_map=geometric_map,
                cur_xy=(xg, yg),
                goal_xy=(gx, gy),
                goal_snap_xy=(ggx, ggy),
                inter_xy=inter_xy,
                path_nodes=path,
                orientation = orientation,
                save_prefix=id_name,
            )
        self._debug_step += 1

        def graph_dist_from_mapxy_to_inter(x_map, y_map):
            n = self._map_index_to_graph_nodes([(x_map, y_map)])[0]
            if n not in self._graph:
                return 10**9
            try:
                return nx.shortest_path_length(self._graph, n, inter_node)
            except Exception:
                return 10**9


        cur_d = graph_dist_from_mapxy_to_inter(xg, yg)

        xg, yg = self._snap_to_navigable_grid(x, y)

        # 前进一步坐标
        s = self.mapper._stride
        fx = xg + int(round(s * np.cos(np.deg2rad(orientation))))
        fy = yg + int(round(s * np.sin(np.deg2rad(orientation))))

        can_fwd = self._can_move_to(xg, yg, fx, fy)
        # can_fwd = self.check_navigability((fx,fy))

        f_d = graph_dist_from_mapxy_to_inter(fx, fy) if can_fwd else 10**9
        #
        if can_fwd and (f_d < cur_d):
            action = HabitatSimActions.MOVE_FORWARD
            self._prev_action = action
            return action

        next_node_idx = self._graph.nodes[path[1]]['map_index']
        self._prev_next_node = path[1]
        desired_orientation = np.round(
            np.rad2deg(np.arctan2(next_node_idx[1] - y, next_node_idx[0] - x))) % 360
        rotation = (desired_orientation - orientation) % 360

        if rotation == 0:
            action = HabitatSimActions.MOVE_FORWARD
        elif rotation == 90:
            action = HabitatSimActions.TURN_RIGHT
        elif rotation == 180:
            action = np.random.choice([HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT])
        elif rotation == 270:
            action = HabitatSimActions.TURN_LEFT
        else:
            raise ValueError('Invalid rotation')

        if action == HabitatSimActions.TURN_LEFT:
            self._turn_bias = HabitatSimActions.TURN_LEFT
        elif action == HabitatSimActions.TURN_RIGHT:
            self._turn_bias = HabitatSimActions.TURN_RIGHT

        self._prev_action = action
        self._prev_next_node = None
        return action


    # -----------------------------
    # planning with WORLD GOAL
    # -----------------------------
    def plan_world(self, observation: dict, goal_world, stop, distribution=None,id_name='exp'):
        """
        goal_world: (wx, wz)
        """
        gx, gy = self.mapper.world_to_map(goal_world[0], goal_world[1])
        return self.plan(observation, goal=(gx, gy), stop=stop, distribution=distribution,id_name=id_name)

    # -----------------------------
    # helper: goal <-> intermediate (kept)
    # -----------------------------
    def get_map_coordinates(self, relative_goal):
        map_size = self._action_map_size
        _, _, x, y, _ = self.mapper.get_maps_and_agent_pose()
        pg_y, pg_x = np.unravel_index(relative_goal, (map_size, map_size))
        pg_x = int(pg_x - map_size // 2)
        pg_y = int(pg_y - map_size // 2)
        delta_x, delta_y = self.mapper.egocentric_to_allocentric(pg_x, pg_y, action_map_res=self._action_map_res)
        return x + delta_x, y + delta_y

    def goal_to_intermediate_goal(self, goal_xy):
        map_size = self._action_map_size
        c = map_size // 2
        _, _, x, y, _ = self.mapper.get_maps_and_agent_pose()

        x_goal, y_goal = int(goal_xy[0]), int(goal_xy[1])
        dx = x_goal - x
        dy = y_goal - y
        ex, ey = self.mapper.allocentric_to_egocentric(dx, dy, action_map_res=self._action_map_res)
        ex = int(np.round(ex))
        ey = int(np.round(ey))
        pg_x = int(np.clip(c + ex, 0, map_size - 1))
        pg_y = int(np.clip(c + ey, 0, map_size - 1))
        return int(pg_y * map_size + pg_x)

    def global_goal_to_intermediate_goal(self, goal_xy):
        x_goal, y_goal = int(goal_xy[0]), int(goal_xy[1])
        map_size = self._action_map_size
        c = map_size // 2
        ex, ey = self.mapper.global_to_egocentric(x_goal, y_goal)
        ex = int(np.round(ex))
        ey = int(np.round(ey))
        pg_x = int(np.clip(c + ex, 0, map_size - 1))
        pg_y = int(np.clip(c + ey, 0, map_size - 1))
        return int(pg_y * map_size + pg_x)

    def world_goal_to_intermediate_goal(self, goal_world):
        wx, wz = float(goal_world[0]), float(goal_world[1])
        x_goal, y_goal = self.mapper.world_to_map(wx, wz)
        return self.global_goal_to_intermediate_goal((x_goal, y_goal))

    # -----------------------------
    # navigability check (kept)
    # -----------------------------
    def check_navigability(self, goal):
        _, _, x, y, _ = self.mapper.get_maps_and_agent_pose()
        graph_nodes = self._map_index_to_graph_nodes([(x, y), goal])
        if graph_nodes[0] not in self._graph:
            return False
        return (graph_nodes[1] in self._graph) and nx.has_path(self._graph, source=graph_nodes[0], target=graph_nodes[1])

    # -----------------------------
    # incremental graph update (kept)
    # -----------------------------
    def _update_graph(self, non_navigable_points, blocked_paths):
        non_navigable_nodes = self._map_index_to_graph_nodes(non_navigable_points)
        blocked_edges = [self._map_index_to_graph_nodes([a, b]) for a, b in blocked_paths]

        for node in non_navigable_nodes:
            if node in self._graph.nodes:
                # store node attrs for reset
                self._removed_nodes.append((node, dict(self._graph.nodes[node])))
                # store edges for reset
                self._removed_edges += [(node, neighbor) for neighbor in self._graph[node]]

        self._removed_edges += blocked_edges

        self._graph.remove_nodes_from(non_navigable_nodes)
        self._graph.remove_edges_from(blocked_edges)

    # -----------------------------
    # mapping between map index and graph nodes (kept)
    # -----------------------------
    def _map_index_to_graph_nodes(self, map_indices: list) -> list:
        graph_nodes = list()
        for map_index in map_indices:
            graph_nodes.append(map_index[1] * len(self._navigable_ys) + map_index[0])
        return graph_nodes

    # -----------------------------
    # build graph from occupancy map
    # -----------------------------
    def _map_to_graph(self, geometric_map: np.array) -> nx.Graph:
        """
        Build grid graph from geometric_map.
        occupancy_map==1 means obstacle.
        NOTE: we REMOVE the "keep only largest component" trimming to avoid goal disappearing.
        """
        occupancy_map = np.bitwise_and(
            geometric_map[:, :, 0] >= self._obstacle_threshold,
            geometric_map[:, :, 1] >= self._obstacle_threshold
        )

        graph = nx.Graph()
        for idx_y, y in enumerate(self._navigable_ys):
            for idx_x, x in enumerate(self._navigable_xs):
                node_index = y * len(self._navigable_ys) + x

                if occupancy_map[y][x]:
                    continue

                if node_index not in graph:
                    graph.add_node(node_index, map_index=(x, y))

                # +Y direction
                if idx_y < len(self._navigable_ys) - 1:
                    next_y = self._navigable_ys[idx_y + 1]
                    if not occupancy_map[y: next_y + 1, x].any():
                        next_node_index = next_y * len(self._navigable_ys) + x
                        if next_node_index not in graph:
                            graph.add_node(next_node_index, map_index=(x, next_y))
                        graph.add_edge(node_index, next_node_index)

                # +X direction
                if idx_x < len(self._navigable_xs) - 1:
                    next_x = self._navigable_xs[idx_x + 1]
                    if not occupancy_map[y, x: next_x + 1].any():
                        next_node_index = y * len(self._navigable_ys) + next_x
                        if next_node_index not in graph:
                            graph.add_node(next_node_index, map_index=(next_x, y))
                        graph.add_edge(node_index, next_node_index)

        return graph

    # =========================================================
    #  NEW helpers for stable 90-degree planning
    # =========================================================
    def _snap_to_navigable_grid(self, x, y):
        xs = np.asarray(self._navigable_xs)
        ys = np.asarray(self._navigable_ys)
        xg = int(xs[np.argmin((xs - x) ** 2)])
        yg = int(ys[np.argmin((ys - y) ** 2)])
        return xg, yg

    def _node_id(self, x, y):
        return y * len(self._navigable_ys) + x

    def _nearest_reachable_node(self, goal_xy, cur_node):
        gx, gy = int(goal_xy[0]), int(goal_xy[1])
        try:
            comp = nx.node_connected_component(self._graph, cur_node)
        except Exception:
            comp = self._graph.nodes

        best = None
        best_d2 = 1e18
        for n in comp:
            mi = self._graph.nodes[n].get("map_index", None)
            if mi is None:
                continue
            x, y = mi
            d2 = (x - gx) ** 2 + (y - gy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = n
        return best

    def _choose_lookahead_node(self, path):
        """
        For 90-degree turning: pick a node after the first corner,
        or a far node if no corner in lookahead.
        """
        kmax = min(self._lookahead, len(path) - 1)

        def axis(p, q):
            x1, y1 = self._graph.nodes[p]["map_index"]
            x2, y2 = self._graph.nodes[q]["map_index"]
            return 0 if x2 != x1 else 1  # 0: x changes, 1: y changes

        chosen = path[1]
        prev_axis = axis(path[0], path[1])
        chosen = path[min(1, kmax)]

        for i in range(1, kmax):
            a = axis(path[i - 1], path[i])
            b = axis(path[i], path[i + 1])
            if b != a:
                chosen = path[i + 1]  # node after corner
                break
            chosen = path[i + 1]

        return chosen


    def _dir_from_delta(self, dx, dy, s):
        if dx == s and dy == 0:   return 0
        if dx == 0 and dy == s:   return 90
        if dx == -s and dy == 0:  return 180
        if dx == 0 and dy == -s:  return 270
        return None

    def _compress_path_to_first_segment(self, path, xg, yg):
        """
        输入 path(node ids)，输出第一段的方向(dir)和长度(steps)
        """
        s = self.mapper._stride
        if len(path) < 2:
            return None, 0

        # 第一步方向
        x1, y1 = self._graph.nodes[path[1]]["map_index"]
        dx1, dy1 = x1 - xg, y1 - yg
        d = self._dir_from_delta(dx1, dy1, s)
        if d is None:
            return None, 0

        # 往后数同方向能走多少步
        steps = 1
        prev_x, prev_y = x1, y1
        for i in range(2, len(path)):
            xi, yi = self._graph.nodes[path[i]]["map_index"]
            dx, dy = xi - prev_x, yi - prev_y
            di = self._dir_from_delta(dx, dy, s)
            if di != d:
                break
            steps += 1
            prev_x, prev_y = xi, yi

        return d, steps


    def _can_move_to(self, x_from, y_from, x_to, y_to):
        """
        判断从(from)到(to)这条 stride 边是否可走：
        1) to 节点在图里
        2) (from,to) 边在图里
        3) geometric_map 上 to 不是障碍（更强的兜底）
        """
        # snap 到网格
        x_from, y_from = self._snap_to_navigable_grid(x_from, y_from)
        x_to, y_to     = self._snap_to_navigable_grid(x_to, y_to)

        n_from = self._map_index_to_graph_nodes([(x_from, y_from)])[0]
        n_to   = self._map_index_to_graph_nodes([(x_to, y_to)])[0]

        if n_from not in self._graph or n_to not in self._graph:
            return False
        if not self._graph.has_edge(n_from, n_to):
            return False

        geometric_map, _, _, _, _ = self.mapper.get_maps_and_agent_pose()
        # geometric_map[y,x,0] = obstacle channel; 1 means obstacle
        if geometric_map[y_to, x_to, 0] > 0.5:
            return False

        return True

    def _turn_toward_intermediate(self, xg, yg, orientation, inter_xy):
        tx, ty = inter_xy

        # 目标向量 v = inter - agent
        vx = tx - xg
        vy = ty - yg

        # 如果目标就在当前位置，不用转
        if vx == 0 and vy == 0:
            return None  # caller 决定 forward/stop

        # 当前朝向单位向量 f（用你 Mapper 的定义：x += cos, y += sin）
        fx = int(round(np.cos(np.deg2rad(orientation))))
        fy = int(round(np.sin(np.deg2rad(orientation))))

        # 叉积符号：cross = f x v（2D）
        cross = fx * vy - fy * vx

        if cross < 0:
            return HabitatSimActions.TURN_LEFT
        elif cross >= 0:
            return HabitatSimActions.TURN_RIGHT
        else:
            # 共线：在正前或正后。用 bias 防抖
            return getattr(self, "_turn_bias", HabitatSimActions.TURN_LEFT)

    def _plot_plan_debug(
        self,
        geometric_map,
        cur_xy,
        goal_xy,
        goal_snap_xy,
        inter_xy,
        path_nodes,
        orientation,
        save_prefix="plan",
    ):
        H, W, _ = geometric_map.shape

        obs = geometric_map[:, :, 0] > 0.5   # obstacle
        exp = geometric_map[:, :, 1] > 0.5   # explored

        # ===== 背景 RGB 图（固定颜色）=====
        bg = np.zeros((H, W, 3), dtype=np.uint8)
        bg[:, :, :] = 60  # 未探索：深灰
        bg[exp & (~obs)] = np.array([255, 255, 255], dtype=np.uint8)  # explored free
        bg[exp & (obs)]  = np.array([0, 0, 0], dtype=np.uint8)        # explored obstacle

        # ===== path nodes -> (x,y) =====
        path_xy = []
        if path_nodes is not None and len(path_nodes) > 0:
            for n in path_nodes:
                if n in self._graph:
                    path_xy.append(self._graph.nodes[n]["map_index"])  # (x,y)

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(bg, origin="upper")
        plt.title(f"{save_prefix} step={self._debug_step}")

        # ===== 最短路径：蓝色折线 =====
        if len(path_xy) >= 2:
            xs = [p[0] for p in path_xy]
            ys = [p[1] for p in path_xy]
            plt.plot(xs, ys, linewidth=2.5, color="dodgerblue", alpha=0.95,
                     label=f"shortest path (len={len(path_xy)})")

        # ===== Agent：红圆点 =====
        cx, cy = cur_xy
        plt.scatter([cx], [cy], s=90, c="red", marker="o",
                    edgecolors="white", linewidths=1.0,
                    label=f"agent (x,y)=({cx},{cy})  ori={int(orientation)}°")

        # ===== 朝向箭头：黄色 =====
        theta = np.deg2rad(orientation)
        dx = np.cos(theta)
        dy = np.sin(theta)

        arrow_len = max(3, int(self.mapper._stride * 0.8))
        ax = dx * arrow_len
        ay = dy * arrow_len

        plt.arrow(
            cx, cy, ax, ay,
            head_width=6,
            head_length=8,
            fc="yellow",
            ec="yellow",
            linewidth=2.0,
            length_includes_head=True,
            alpha=0.95
        )

        # ===== Goal (snapped)：绿叉 =====
        if goal_snap_xy is not None:
            gx, gy = goal_snap_xy
            plt.scatter([gx], [gy], s=120, c="limegreen", marker="x",
                        linewidths=2.5,
                        label=f"goal(snap) (x,y)=({gx},{gy})")

        # ===== Intermediate：橙三角 =====
        if inter_xy is not None:
            ix, iy = inter_xy
            plt.scatter([ix], [iy], s=120, c="orange", marker="^",
                        edgecolors="black", linewidths=0.8,
                        label=f"intermediate (x,y)=({ix},{iy})")

        # ===== Raw goal：紫加号（可选）=====
        if goal_xy is not None:
            rgx, rgy = int(goal_xy[0]), int(goal_xy[1])
            if 0 <= rgx < W and 0 <= rgy < H:
                plt.scatter([rgx], [rgy], s=80, c="magenta", marker="+",
                            linewidths=2.0,
                            label=f"goal(raw) (x,y)=({rgx},{rgy})")

        plt.xlim([0, W - 1])
        plt.ylim([H - 1, 0])

        # 让 legend 更紧凑一点
        plt.legend(loc="lower right", fontsize=9, framealpha=0.85)
        plt.tight_layout()

        out_png = os.path.join(self._debug_dir, f"{save_prefix}_{self._debug_step:03d}.png")
        plt.savefig(out_png, dpi=150)
        plt.close(fig)

        # out_np = os.path.join(self._debug_dir, f"{save_prefix}_{self._debug_step:06d}_path.npy")
        # np.save(out_np, np.array(path_xy, dtype=np.int32))


    def _pick_frontier_node(self, geometric_map, cur_node, goal_xy, max_samples=3000, step=4,
                            w_goal=1.0, w_agent=0.2):
        gx, gy = goal_xy

        frontier = self._get_frontier_mask(geometric_map)

        # 当前连通块（保证可达）
        comp = nx.node_connected_component(self._graph, cur_node)

        # 从 comp 里抽取 frontier 节点（用 graph 的 node->map_index，不用全图扫，快很多）
        cand = []
        for n in comp:
            x2, y2 = self._graph.nodes[n]["map_index"]
            if frontier[y2, x2]:
                cand.append(n)

        if len(cand) == 0:
            return None

        # 下采样，避免太多 frontier
        if len(cand) > max_samples:
            cand = cand[::step]

        # 当前点
        cx, cy = self._graph.nodes[cur_node]["map_index"]

        best_n, best_score = None, 1e18
        for n in cand:
            x2, y2 = self._graph.nodes[n]["map_index"]
            d_goal  = (x2 - gx) ** 2 + (y2 - gy) ** 2
            d_agent = (x2 - cx) ** 2 + (y2 - cy) ** 2
            score = w_goal * d_goal + w_agent * d_agent
            if score < best_score:
                best_score = score
                best_n = n

        return best_n


    def _forward_xy(self, xg, yg, orientation):
        s = self.mapper._stride
        fx = xg + int(round(s * np.cos(np.deg2rad(orientation))))
        fy = yg + int(round(s * np.sin(np.deg2rad(orientation))))
        return self._snap_to_navigable_grid(fx, fy)

