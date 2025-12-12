import numpy as np
import torch
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import math

def heatmap_argmax_direction(hm: torch.Tensor):
    """
    hm: (1, H, W) 或 (H, W)，值可以是 logits 或 prob
    返回:
        dx, dy: 最大值位置相对中心的位移（右正、前正）
    """
    # 保证形状是 (H, W)
    if hm.ndim == 3:
        hm = hm[0]      # (H, W)
    hm_np = hm.detach().cpu().numpy()

    H, W = hm_np.shape
    cx, cy = W // 2, H // 2

    # 如果全 0，没有信息
    if np.allclose(hm_np, 0):
        return None, None

    # 找到最大值像素位置 (y_max, x_max)
    y_max, x_max = np.unravel_index(hm_np.argmax(), hm_np.shape)

    # 相对位移：右正、前正
    dx = x_max - cx          # x 往右为正
    dy = cy - y_max          # y 往上为正（前方）

    return dx, dy

def policy_from_heatmap(
    hm: torch.Tensor,
    angle_thresh_deg: float = 45.0,
    min_dist: float = 1.0,
):
    """
    输入:
        hm: (1, H, W) 的 heatmap（一般是 logits 经过 sigmoid 之后）
    输出:
        一个 HabitatSimActions.* 动作
    """
    dx, dy = heatmap_argmax_direction(hm)

    # 没信息：默认向前
    if dx is None:
        return HabitatSimActions.MOVE_FORWARD

    # 半径（像素距离）
    r = np.sqrt(dx**2 + dy**2)

    # 角度：atan2(dx, dy) 约定 dy 是前后, dx 是左右
    #   angle > 0  → 在右边
    #   angle < 0  → 在左边
    #   angle ≈ 0 → 在正前方
    angle = np.arctan2(dx, dy)
    angle_deg = np.degrees(angle)

    # 1. 如果目标点离中心很近：认为已经到了附近
    if r < min_dist:
        return HabitatSimActions.STOP

    # 2. 根据角度决定转向 or 前进
    if abs(angle_deg) < angle_thresh_deg:
        # 热力图最大值基本在正前方
        return HabitatSimActions.MOVE_FORWARD
    elif angle_deg > 0:
        # 最大值在右侧
        return HabitatSimActions.TURN_RIGHT
    else:
        # 最大值在左侧
        return HabitatSimActions.TURN_LEFT

def euclidean_distance(a, b):
    """计算两个 list 的欧式距离"""
    if len(a) != len(b):
        raise ValueError("两个 list 长度必须相同")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def is_front_1m_free_point(ego_map, map_res=0.1, dist=1.0):
    """
    根据 EgoMap 检测 agent 正前方 dist 米是否可行
    - ego_map: (MAP_SIZE, MAP_SIZE, 2)，最后一维 [obstacle, explored]
    - map_res: 每个栅格的尺寸（米）
    - dist:    要检查的前方距离（米）
    返回值：True = 前方 dist 米内可行，False = 有障碍或未知区域
    """
    V = ego_map.shape[0]         # MAP_SIZE
    center_x = V // 2            # 正前方列索引

    # 前方 dist 米对应的格子数
    n_forward = int(dist / map_res)   # 例如 1.0 / 0.1 = 10

    # y 范围：从离 agent 最近的那行，到 dist 米处的那行
    y_max = V - 1                   # 最靠近 agent 的一行
    y_min = max(0, V - n_forward)   # 距离 dist 处那一行

    # 取出正前方一条竖线
    obstacle_col = ego_map[y_min:y_max + 1, center_x, 0]
    explore_col  = ego_map[y_min:y_max + 1, center_x, 1]

    # 条件1：这条线上所有格子都被探索过
    all_explored = np.all(explore_col == 1)

    # 条件2：这条线上所有格子都没有障碍物
    no_obstacle = np.all(obstacle_col == 0)

    return True

