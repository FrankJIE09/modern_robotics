"""
PRM路径规划算法实现 - 机械臂版本
基于《现代机器人学》第10章算法10.4
使用Robotics Toolbox进行运动学计算
"""

import numpy as np
import random
from collections import defaultdict
from manipulator_model import Manipulator2R


class PRMPlannerManipulator:
    """PRM路径规划器 - 用于机械臂"""
    
    def __init__(self, manipulator, obstacles, num_samples=500, k_neighbors=10):
        """
        初始化PRM规划器
        
        参数:
            manipulator: Manipulator2R实例
            obstacles: 障碍物列表
            num_samples: 采样点数量
            k_neighbors: 每个节点的邻居数量
        """
        self.manipulator = manipulator
        self.obstacles = obstacles
        self.num_samples = num_samples
        self.k_neighbors = k_neighbors
        
        # PRM路线图：无向图
        self.nodes = []  # 节点列表
        self.edges = []  # 边列表 [(node_i, node_j), ...]
        self.graph = defaultdict(list)  # 邻接表 {node_idx: [neighbor_idx, ...]}
        
        # 可视化数据
        self.path = []
    
    def _sample_config(self):
        """
        在配置空间中采样
        
        返回:
            q_sample: 采样配置
        """
        q_sample = np.array([
            random.uniform(-np.pi, np.pi),  # theta1
            random.uniform(-np.pi, np.pi)   # theta2
        ])
        return q_sample
    
    def _is_collision_free(self, q):
        """检查配置是否无碰撞"""
        is_collision, _ = self.manipulator.check_collision(q, self.obstacles)
        return not is_collision
    
    def _is_path_collision_free(self, q1, q2, num_checks=10):
        """检查从q1到q2的路径是否无碰撞"""
        for i in range(num_checks + 1):
            alpha = i / num_checks
            q_interp = q1 + alpha * (q2 - q1)
            q_interp = np.arctan2(np.sin(q_interp), np.cos(q_interp))
            
            if not self._is_collision_free(q_interp):
                return False
        return True
    
    def _distance(self, q1, q2):
        """计算两个配置之间的距离（考虑角度周期性）"""
        diff = q2 - q1
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        return np.linalg.norm(diff)
    
    def _k_nearest_neighbors(self, q, k):
        """
        找到k个最近邻居
        
        参数:
            q: 查询配置
            k: 邻居数量
        
        返回:
            neighbors: 最近邻居的索引列表
        """
        distances = []
        for i, node in enumerate(self.nodes):
            dist = self._distance(q, node)
            distances.append((dist, i))
        
        # 排序并取前k个
        distances.sort(key=lambda x: x[0])
        neighbors = [idx for _, idx in distances[:k]]
        return neighbors
    
    def build_roadmap(self):
        """
        构建PRM路线图（算法10.4）
        
        返回:
            success: 是否成功构建
        """
        print("构建PRM路线图...")
        print(f"  采样点数: {self.num_samples}")
        print(f"  每个节点的邻居数: {self.k_neighbors}")
        
        # 第一阶段：采样自由配置
        print("  阶段1: 采样自由配置...")
        valid_samples = 0
        
        while len(self.nodes) < self.num_samples:
            q_sample = self._sample_config()
            
            if self._is_collision_free(q_sample):
                self.nodes.append(q_sample.copy())
                valid_samples += 1
                
                if valid_samples % 100 == 0:
                    print(f"    已采样: {valid_samples}/{self.num_samples}")
        
        print(f"  ✓ 采样完成，共 {len(self.nodes)} 个节点")
        
        # 第二阶段：连接邻居
        print("  阶段2: 连接邻居节点...")
        edges_added = 0
        
        for i, q_i in enumerate(self.nodes):
            # 找到k个最近邻居
            neighbors = self._k_nearest_neighbors(q_i, self.k_neighbors)
            
            for j in neighbors:
                if i == j:
                    continue
                
                q_j = self.nodes[j]
                
                # 检查是否已有边
                if (i, j) in self.edges or (j, i) in self.edges:
                    continue
                
                # 检查路径是否无碰撞
                if self._is_path_collision_free(q_i, q_j):
                    # 添加边
                    self.edges.append((i, j))
                    self.graph[i].append(j)
                    self.graph[j].append(i)
                    edges_added += 1
            
            if (i + 1) % 100 == 0:
                print(f"    处理节点: {i + 1}/{len(self.nodes)}, 已添加边: {edges_added}")
        
        print(f"  ✓ 路线图构建完成，共 {edges_added} 条边")
        return True
    
    def query(self, q_start, q_goal):
        """
        查询路径（使用A*搜索路线图）
        
        参数:
            q_start: 起始配置
            q_goal: 目标配置
        
        返回:
            success: 是否找到路径
            path: 路径（配置序列）
        """
        print("查询路径...")
        
        # 将起点和目标连接到路线图
        start_idx = self._connect_to_roadmap(q_start)
        goal_idx = self._connect_to_roadmap(q_goal)
        
        if start_idx is None or goal_idx is None:
            print("  ✗ 无法将起点或目标连接到路线图")
            return False, []
        
        print(f"  起点连接到节点 {start_idx}")
        print(f"  目标连接到节点 {goal_idx}")
        
        # 使用A*搜索路径
        path_indices = self._astar_search(start_idx, goal_idx)
        
        if path_indices is None:
            print("  ✗ 在路线图中未找到路径")
            return False, []
        
        # 构建完整路径
        self.path = [q_start]
        for idx in path_indices[1:-1]:  # 跳过起点和终点（已包含）
            self.path.append(self.nodes[idx].copy())
        self.path.append(q_goal)
        
        print(f"  ✓ 找到路径，长度: {len(self.path)} 个配置")
        return True, self.path
    
    def _connect_to_roadmap(self, q, max_attempts=20):
        """
        将配置连接到路线图
        
        参数:
            q: 配置
            max_attempts: 最大尝试次数
        
        返回:
            node_idx: 连接的节点索引，如果失败返回None
        """
        # 如果配置本身碰撞，返回None
        if not self._is_collision_free(q):
            return None
        
        # 找到最近邻居
        neighbors = self._k_nearest_neighbors(q, min(self.k_neighbors, len(self.nodes)))
        
        # 尝试连接到最近的邻居
        for neighbor_idx in neighbors:
            q_neighbor = self.nodes[neighbor_idx]
            
            if self._is_path_collision_free(q, q_neighbor):
                return neighbor_idx
        
        return None
    
    def _astar_search(self, start_idx, goal_idx):
        """
        使用A*算法在路线图中搜索路径
        
        参数:
            start_idx: 起始节点索引
            goal_idx: 目标节点索引
        
        返回:
            path_indices: 路径节点索引列表，如果失败返回None
        """
        import heapq
        
        # 初始化
        open_set = [(0, start_idx)]
        came_from = {}
        g_score = {start_idx: 0}
        f_score = {start_idx: self._distance(self.nodes[start_idx], self.nodes[goal_idx])}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal_idx:
                # 重构路径
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from.get(current, None)
                path.reverse()
                return path
            
            # 探索邻居
            for neighbor in self.graph[current]:
                tentative_g = g_score[current] + self._distance(
                    self.nodes[current], self.nodes[neighbor]
                )
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score = self._distance(self.nodes[neighbor], self.nodes[goal_idx])
                    f_score[neighbor] = tentative_g + h_score
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def get_roadmap_data(self):
        """
        获取路线图数据（用于可视化）
        
        返回:
            nodes: 节点列表
            edges: 边列表
        """
        return self.nodes, self.edges

