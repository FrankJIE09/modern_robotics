"""
RRT路径规划算法实现 - 机械臂版本
基于《现代机器人学》第10章算法10.3
使用Robotics Toolbox进行运动学计算
"""

import numpy as np
import random
from manipulator_model import Manipulator2R


class RRTPlannerManipulator:
    """RRT路径规划器 - 用于机械臂"""
    
    def __init__(self, manipulator, obstacles, q_start, q_goal, 
                 step_size=0.1, goal_bias=0.1, max_iterations=5000):
        """
        初始化RRT规划器
        
        参数:
            manipulator: Manipulator2R实例
            obstacles: 障碍物列表
            q_start: 起始配置 [theta1, theta2]
            q_goal: 目标配置 [theta1, theta2] 或目标区域
            step_size: 扩展步长（在配置空间中）
            goal_bias: 目标偏向概率（0-1之间）
            max_iterations: 最大迭代次数
        """
        self.manipulator = manipulator
        self.obstacles = obstacles
        self.q_start = np.array(q_start)
        self.q_goal = np.array(q_goal) if isinstance(q_goal, (list, np.ndarray)) else q_goal
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.max_iterations = max_iterations
        
        # RRT树结构：{节点: 父节点}
        self.tree = {tuple(self.q_start): None}
        self.nodes = [self.q_start.copy()]
        
        # 可视化数据
        self.explored_nodes = []
        self.path = []
    
    def _sample_config(self):
        """
        在配置空间中采样
        
        返回:
            q_sample: 采样配置
        """
        # 目标偏向采样
        if random.random() < self.goal_bias:
            if isinstance(self.q_goal, dict):
                # 目标区域：在区域内随机采样
                return self._sample_goal_region()
            else:
                return self.q_goal.copy()
        else:
            # 均匀随机采样
            q_sample = np.array([
                random.uniform(-np.pi, np.pi),  # theta1
                random.uniform(-np.pi, np.pi)   # theta2
            ])
            return q_sample
    
    def _sample_goal_region(self):
        """在目标区域内采样"""
        # 简化：如果目标是区域，在区域内随机采样
        # 这里假设目标是单个配置
        return self.q_goal.copy()
    
    def _nearest_node(self, q_sample):
        """
        找到树中距离q_sample最近的节点
        
        参数:
            q_sample: 采样配置
        
        返回:
            q_near: 最近的节点配置
        """
        min_dist = float('inf')
        q_near = None
        
        for q_node in self.nodes:
            dist = self._distance(q_node, q_sample)
            if dist < min_dist:
                min_dist = dist
                q_near = q_node
        
        return q_near
    
    def _distance(self, q1, q2):
        """
        计算两个配置之间的距离
        
        参数:
            q1, q2: 配置向量
        
        返回:
            距离（考虑角度的周期性）
        """
        # 对于角度，需要考虑周期性（2π等价）
        diff = q2 - q1
        # 将角度差归一化到[-π, π]
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        return np.linalg.norm(diff)
    
    def _steer(self, q_near, q_sample):
        """
        从q_near向q_sample方向扩展步长
        
        参数:
            q_near: 最近节点
            q_sample: 采样点
        
        返回:
            q_new: 新配置
        """
        # 计算方向
        direction = q_sample - q_near
        dist = self._distance(q_near, q_sample)
        
        if dist < self.step_size:
            # 如果距离小于步长，直接返回采样点
            return q_sample.copy()
        
        # 沿方向扩展步长
        q_new = q_near + (direction / dist) * self.step_size
        
        # 确保角度在[-π, π]范围内
        q_new = np.arctan2(np.sin(q_new), np.cos(q_new))
        
        return q_new
    
    def _is_collision_free(self, q):
        """
        检查配置是否无碰撞
        
        参数:
            q: 配置
        
        返回:
            是否无碰撞
        """
        is_collision, _ = self.manipulator.check_collision(q, self.obstacles)
        return not is_collision
    
    def _is_path_collision_free(self, q1, q2, num_checks=15):
        """
        检查从q1到q2的路径是否无碰撞
        
        参数:
            q1, q2: 起始和结束配置
            num_checks: 检查点的数量
        
        返回:
            是否无碰撞
        """
        # 在路径上采样多个点进行检查
        # 增加检查点数量，更精确的碰撞检测
        for i in range(num_checks + 1):
            alpha = i / num_checks
            q_interp = q1 + alpha * (q2 - q1)
            # 归一化角度
            q_interp = np.arctan2(np.sin(q_interp), np.cos(q_interp))
            
            if not self._is_collision_free(q_interp):
                return False
        
        return True
    
    def _is_goal(self, q):
        """
        检查是否到达目标
        
        参数:
            q: 配置
        
        返回:
            是否在目标集内
        """
        if isinstance(self.q_goal, dict):
            # 目标区域：检查是否在区域内
            # 简化实现
            return False
        else:
            # 单个目标配置
            dist = self._distance(q, self.q_goal)
            return dist < 0.2  # 增加容差，更容易到达目标
    
    def plan(self):
        """
        执行RRT搜索
        
        返回:
            success: 是否找到路径
            path: 路径（配置序列）
        """
        print("开始RRT搜索...")
        
        for iteration in range(self.max_iterations):
            # 采样
            q_sample = self._sample_config()
            
            # 找最近节点
            q_near = self._nearest_node(q_sample)
            
            # 扩展
            q_new = self._steer(q_near, q_sample)
            
            # 检查碰撞
            if self._is_collision_free(q_new):
                # 检查路径是否无碰撞
                if self._is_path_collision_free(q_near, q_new):
                    # 添加到树
                    self.tree[tuple(q_new)] = tuple(q_near)
                    self.nodes.append(q_new.copy())
                    self.explored_nodes.append(q_new.copy())
                    
                    # 检查是否到达目标
                    if self._is_goal(q_new):
                        print(f"✓ 找到路径！迭代次数: {iteration + 1}")
                        self.path = self._reconstruct_path(q_new)
                        return True, self.path
            
            # 每500次迭代打印进度，并检查是否接近目标
            if (iteration + 1) % 500 == 0:
                # 检查当前树中是否有节点接近目标
                if self.nodes:
                    min_dist_to_goal = min([self._distance(node, self.q_goal) 
                                           for node in self.nodes])
                    print(f"  迭代: {iteration + 1}/{self.max_iterations}, "
                          f"树大小: {len(self.nodes)}, "
                          f"最近距离目标: {min_dist_to_goal:.3f}")
                else:
                    print(f"  迭代: {iteration + 1}/{self.max_iterations}, 树大小: {len(self.nodes)}")
        
        print("✗ 未找到路径（达到最大迭代次数）")
        return False, []
    
    def _reconstruct_path(self, q_goal):
        """
        从目标节点重构路径
        
        参数:
            q_goal: 目标配置
        
        返回:
            path: 从起点到终点的路径
        """
        path = []
        current = tuple(q_goal)
        
        while current is not None:
            path.append(np.array(current))
            current = self.tree.get(current, None)
        
        path.reverse()
        return path
    
    def get_tree_edges(self):
        """
        获取RRT树的所有边（用于可视化）
        
        返回:
            edges: 边列表 [(q1, q2), ...]
        """
        edges = []
        for q_node, q_parent in self.tree.items():
            if q_parent is not None:
                edges.append((np.array(q_parent), np.array(q_node)))
        return edges

