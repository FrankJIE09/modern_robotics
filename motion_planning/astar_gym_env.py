"""
使用Gymnasium环境展示A*算法
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gymnasium as gym
from gymnasium import spaces
from astar_planner import AStarPlanner

# 导入中文字体配置
try:
    from font_config import init_chinese_font
    init_chinese_font(verbose=False)
except ImportError:
    # 如果font_config不存在，使用备用方案
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False


class GridPathPlanningEnv(gym.Env):
    """
    2D网格路径规划环境（Gymnasium兼容）
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, width=20, height=20, obstacle_ratio=0.2, render_mode=None):
        super().__init__()
        
        self.width = width
        self.height = height
        self.obstacle_ratio = obstacle_ratio
        self.render_mode = render_mode
        
        # 创建网格
        self.grid = np.zeros((height, width), dtype=int)
        self._generate_obstacles()
        
        # 设置起点和终点
        self.start = (0, 0)
        self.goal = (height - 1, width - 1)
        self.grid[self.start] = 0
        self.grid[self.goal] = 0
        
        # 当前状态
        self.current_pos = self.start
        self.path = []
        self.explored = []
        
        # 动作空间：8个方向
        self.action_space = spaces.Discrete(8)
        
        # 观察空间：当前位置 + 局部视野
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(height, width), dtype=np.float32
        )
        
        # 渲染相关
        self.fig = None
        self.ax = None
    
    def _generate_obstacles(self):
        """生成随机障碍物"""
        num_obstacles = int(self.width * self.height * self.obstacle_ratio)
        for _ in range(num_obstacles):
            row = np.random.randint(0, self.height)
            col = np.random.randint(0, self.width)
            self.grid[row, col] = 1
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        self.current_pos = self.start
        self.path = []
        self.explored = []
        
        observation = self._get_observation()
        info = {"start": self.start, "goal": self.goal}
        
        return observation, info
    
    def _get_observation(self):
        """获取当前观察"""
        obs = self.grid.copy().astype(np.float32)
        # 标记当前位置
        obs[self.current_pos] = 0.5
        return obs
    
    def step(self, action):
        """执行动作"""
        # 动作映射到8个方向
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 对角线
        ]
        
        dr, dc = directions[action]
        new_row = self.current_pos[0] + dr
        new_col = self.current_pos[1] + dc
        
        # 检查边界和障碍物
        if (0 <= new_row < self.height and 
            0 <= new_col < self.width and 
            self.grid[new_row, new_col] == 0):
            self.current_pos = (new_row, new_col)
            self.path.append(self.current_pos)
        
        # 计算奖励
        reward = -0.1  # 每步小惩罚
        if self.current_pos == self.goal:
            reward = 100.0  # 到达目标大奖励
        
        # 检查是否完成
        terminated = self.current_pos == self.goal
        truncated = len(self.path) > self.width * self.height  # 防止无限循环
        
        observation = self._get_observation()
        info = {"path_length": len(self.path)}
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """渲染环境"""
        if self.render_mode is None:
            return
        
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            plt.ion()
        
        self.ax.clear()
        
        # 绘制网格
        self.ax.imshow(self.grid, cmap='Greys', origin='upper', alpha=0.3)
        
        # 绘制起点和终点
        start_row, start_col = self.start
        goal_row, goal_col = self.goal
        self.ax.plot(start_col, start_row, 'go', markersize=20, 
                    label='起点', zorder=5)
        self.ax.plot(goal_col, goal_row, 'ro', markersize=20, 
                    label='终点', zorder=5)
        
        # 绘制路径
        if self.path:
            path_x = [col for row, col in self.path]
            path_y = [row for row, col in self.path]
            self.ax.plot(path_x, path_y, 'b-', linewidth=2, 
                        label='路径', zorder=4)
            self.ax.plot(path_x, path_y, 'bo', markersize=6, zorder=4)
        
        # 绘制当前位置
        if self.current_pos:
            curr_row, curr_col = self.current_pos
            self.ax.plot(curr_col, curr_row, 'yo', markersize=15, 
                        label='当前位置', zorder=6)
        
        # 网格线
        self.ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', 
                    linewidth=0.5, alpha=0.3)
        
        self.ax.set_xlim(-0.5, self.width - 0.5)
        self.ax.set_ylim(-0.5, self.height - 0.5)
        self.ax.set_aspect('equal')
        self.ax.legend(loc='upper right')
        self.ax.set_title('A*路径规划 - Gym环境', fontsize=14, fontweight='bold')
        
        plt.draw()
        plt.pause(0.1)
    
    def close(self):
        """关闭环境"""
        if self.fig:
            plt.close(self.fig)


def demo_astar_with_gym():
    """使用Gym环境演示A*算法"""
    print("=" * 60)
    print("A*算法 + Gymnasium环境演示")
    print("=" * 60)
    
    # 创建环境
    env = GridPathPlanningEnv(width=25, height=25, obstacle_ratio=0.25, 
                              render_mode="human")
    
    # 重置环境
    obs, info = env.reset()
    print(f"环境大小: {env.height} x {env.width}")
    print(f"起点: {env.start}")
    print(f"终点: {env.goal}")
    print()
    
    # 创建A*规划器
    planner = AStarPlanner(
        grid=env.grid,
        start=env.start,
        goal=env.goal,
        heuristic_type='euclidean'
    )
    
    print("执行A*搜索...")
    success, path = planner.plan()
    
    if success:
        print(f"✓ 找到路径！")
        print(f"  路径长度: {len(path)} 个节点")
        print(f"  探索节点数: {len(planner.explored_nodes)}")
        print()
        
        # 可视化A*结果
        print("显示A*搜索结果...")
        visualize_astar_gym(planner, env)
        
        # 在环境中执行路径
        print("在Gym环境中执行路径...")
        env.reset()
        env.render()
        
        for i, pos in enumerate(path[1:], 1):  # 跳过起点
            # 计算动作（从当前位置到下一个位置）
            current = path[i-1] if i > 0 else env.start
            next_pos = pos
            
            dr = next_pos[0] - current[0]
            dc = next_pos[1] - current[1]
            
            # 映射到动作
            action_map = {
                (-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3,
                (-1, -1): 4, (-1, 1): 5, (1, -1): 6, (1, 1): 7
            }
            
            action = action_map.get((dr, dc), 0)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated:
                print(f"到达目标！总步数: {i}")
                break
            
            plt.pause(0.1)
        
        plt.pause(2)
    else:
        print("✗ 未找到路径")
    
    env.close()
    return planner, env


def visualize_astar_gym(planner, env):
    """在Gym环境中可视化A*结果"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制网格
    ax.imshow(env.grid, cmap='Greys', origin='upper', alpha=0.3)
    
    # 起点和终点
    start_row, start_col = env.start
    goal_row, goal_col = env.goal
    ax.plot(start_col, start_row, 'go', markersize=20, label='起点', zorder=5)
    ax.plot(goal_col, goal_row, 'ro', markersize=20, label='终点', zorder=5)
    
    # 探索的节点
    explored_x = [col for row, col in planner.explored_nodes]
    explored_y = [row for row, col in planner.explored_nodes]
    ax.scatter(explored_x, explored_y, c='yellow', s=20, alpha=0.5, 
               label='探索节点', zorder=2)
    
    # 最优路径
    if planner.path:
        path_x = [col for row, col in planner.path]
        path_y = [row for row, col in planner.path]
        ax.plot(path_x, path_y, 'b-', linewidth=3, label='最优路径', zorder=4)
        ax.plot(path_x, path_y, 'bo', markersize=8, zorder=4)
    
    # 网格线
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title('A*路径规划结果 (Gym环境)', fontsize=14, fontweight='bold')
    ax.set_xlabel('列', fontsize=12)
    ax.set_ylabel('行', fontsize=12)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    planner, env = demo_astar_with_gym()

