"""
A*路径规划算法实现
基于《现代机器人学》第10章的描述
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from collections import deque
import heapq

# 导入中文字体配置
try:
    from font_config import init_chinese_font
    init_chinese_font(verbose=False)
except ImportError:
    # 如果font_config不存在，使用备用方案
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False


class AStarPlanner:
    """A*路径规划器"""
    
    def __init__(self, grid, start, goal, heuristic_type='euclidean'):
        """
        初始化A*规划器
        
        参数:
            grid: 2D数组，0表示自由空间，1表示障碍物
            start: 起始位置 (row, col)
            goal: 目标位置 (row, col) 或目标区域
            heuristic_type: 启发式类型 ('euclidean', 'manhattan', 'diagonal')
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.start = start
        self.goal = goal
        self.heuristic_type = heuristic_type
        
        # 数据结构
        self.OPEN = []  # 优先队列：(估计总成本, 节点)
        self.CLOSED = set()
        self.past_cost = {}  # 从起点到节点的最小成本
        self.parent = {}  # 父节点映射
        self.cost_matrix = self._build_cost_matrix()
        
        # 可视化数据
        self.explored_nodes = []
        self.path = []
        
    def _build_cost_matrix(self):
        """构建成本矩阵（这里简化为邻接矩阵）"""
        # 对于网格，我们使用8连通（包括对角线）
        # 实际成本在搜索时计算
        return None
    
    def _heuristic_cost_to_go(self, node):
        """计算从节点到目标的启发式成本（乐观估计）"""
        row, col = node
        goal_row, goal_col = self.goal
        
        if self.heuristic_type == 'euclidean':
            # 欧几里得距离
            return np.sqrt((row - goal_row)**2 + (col - goal_col)**2)
        elif self.heuristic_type == 'manhattan':
            # 曼哈顿距离
            return abs(row - goal_row) + abs(col - goal_col)
        elif self.heuristic_type == 'diagonal':
            # 对角线距离（允许对角线移动）
            dx = abs(row - goal_row)
            dy = abs(col - goal_col)
            return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)
        else:
            return 0
    
    def _get_neighbors(self, node):
        """获取节点的邻居（8连通）"""
        row, col = node
        neighbors = []
        
        # 8个方向：上、下、左、右、左上、右上、左下、右下
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 4连通
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 对角线
        ]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # 检查边界
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                # 检查是否为障碍物
                if self.grid[new_row, new_col] == 0:
                    neighbors.append((new_row, new_col))
        
        return neighbors
    
    def _edge_cost(self, node1, node2):
        """计算从node1到node2的边成本"""
        r1, c1 = node1
        r2, c2 = node2
        
        # 如果是对角线移动，成本为sqrt(2)
        if abs(r1 - r2) == 1 and abs(c1 - c2) == 1:
            return np.sqrt(2)
        else:
            return 1.0
    
    def _is_goal(self, node):
        """检查节点是否在目标集中"""
        if isinstance(self.goal, tuple):
            return node == self.goal
        else:
            # 如果是目标区域，检查是否在区域内
            return node in self.goal
    
    def plan(self):
        """执行A*搜索"""
        # 初始化
        self.past_cost[self.start] = 0
        h_start = self._heuristic_cost_to_go(self.start)
        heapq.heappush(self.OPEN, (h_start, self.start))
        
        while self.OPEN:
            # 从OPEN中取出估计总成本最小的节点
            est_total_cost, current = heapq.heappop(self.OPEN)
            
            # 如果已经在CLOSED中，跳过
            if current in self.CLOSED:
                continue
            
            # 添加到CLOSED
            self.CLOSED.add(current)
            self.explored_nodes.append(current)
            
            # 检查是否到达目标
            if self._is_goal(current):
                # 重构路径
                self.path = self._reconstruct_path(current)
                return True, self.path
            
            # 探索邻居
            for nbr in self._get_neighbors(current):
                if nbr in self.CLOSED:
                    continue
                
                # 计算到达邻居的暂定成本
                edge_cost = self._edge_cost(current, nbr)
                tentative_past_cost = self.past_cost[current] + edge_cost
                
                # 如果找到更低的成本，更新
                if nbr not in self.past_cost or tentative_past_cost < self.past_cost[nbr]:
                    self.past_cost[nbr] = tentative_past_cost
                    self.parent[nbr] = current
                    
                    # 计算估计总成本
                    h_nbr = self._heuristic_cost_to_go(nbr)
                    est_total_cost_nbr = tentative_past_cost + h_nbr
                    
                    # 添加到OPEN（如果不在OPEN中或成本更低）
                    heapq.heappush(self.OPEN, (est_total_cost_nbr, nbr))
        
        # 没有找到路径
        return False, []
    
    def _reconstruct_path(self, goal_node):
        """从目标节点重构路径"""
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current)
            current = self.parent.get(current, None)
        
        path.reverse()
        return path


class GridWorld:
    """2D网格世界环境"""
    
    def __init__(self, width=20, height=20, obstacle_ratio=0.3):
        """
        创建网格世界
        
        参数:
            width: 网格宽度
            height: 网格高度
            obstacle_ratio: 障碍物比例
        """
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        
        # 随机生成障碍物
        num_obstacles = int(width * height * obstacle_ratio)
        for _ in range(num_obstacles):
            row = np.random.randint(0, height)
            col = np.random.randint(0, width)
            self.grid[row, col] = 1
        
        # 确保起点和终点不是障碍物
        self.start = (0, 0)
        self.goal = (height - 1, width - 1)
        self.grid[self.start] = 0
        self.grid[self.goal] = 0
    
    def set_obstacles(self, obstacles):
        """手动设置障碍物"""
        for row, col in obstacles:
            if 0 <= row < self.height and 0 <= col < self.width:
                self.grid[row, col] = 1
    
    def set_start_goal(self, start, goal):
        """设置起点和终点"""
        self.start = start
        self.goal = goal
        self.grid[start] = 0
        self.grid[goal] = 0


def visualize_astar(planner, grid, start, goal, save_animation=False):
    """可视化A*搜索过程"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制网格
    ax.imshow(grid, cmap='Greys', origin='upper', alpha=0.3)
    
    # 绘制起点和终点
    start_row, start_col = start
    goal_row, goal_col = goal
    ax.plot(start_col, start_row, 'go', markersize=15, label='起点', zorder=5)
    ax.plot(goal_col, goal_row, 'ro', markersize=15, label='终点', zorder=5)
    
    # 绘制探索的节点
    explored_x = [col for row, col in planner.explored_nodes]
    explored_y = [row for row, col in planner.explored_nodes]
    ax.scatter(explored_x, explored_y, c='yellow', s=30, alpha=0.6, 
               label='探索节点', zorder=3)
    
    # 绘制路径
    if planner.path:
        path_x = [col for row, col in planner.path]
        path_y = [row for row, col in planner.path]
        ax.plot(path_x, path_y, 'b-', linewidth=3, label='最优路径', zorder=4)
        ax.plot(path_x, path_y, 'bo', markersize=8, zorder=4)
    
    # 添加网格线
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title('A*路径规划结果', fontsize=14, fontweight='bold')
    ax.set_xlabel('列', fontsize=12)
    ax.set_ylabel('行', fontsize=12)
    
    plt.tight_layout()
    if save_animation:
        plt.savefig('astar_result.png', dpi=150, bbox_inches='tight')
    plt.show()


def create_animation(planner, grid, start, goal, interval=50):
    """创建A*搜索过程的动画"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 初始化绘图
    ax.imshow(grid, cmap='Greys', origin='upper', alpha=0.3)
    start_row, start_col = start
    goal_row, goal_col = goal
    start_point = ax.plot(start_col, start_row, 'go', markersize=15, 
                          label='起点', zorder=5)[0]
    goal_point = ax.plot(goal_col, goal_row, 'ro', markersize=15, 
                         label='终点', zorder=5)[0]
    
    explored_scatter = ax.scatter([], [], c='yellow', s=30, alpha=0.6, 
                                   label='探索节点', zorder=3)
    path_line = ax.plot([], [], 'b-', linewidth=3, label='当前路径', zorder=4)[0]
    path_points = ax.plot([], [], 'bo', markersize=8, zorder=4)[0]
    
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title('A*路径规划过程', fontsize=14, fontweight='bold')
    ax.set_xlabel('列', fontsize=12)
    ax.set_ylabel('行', fontsize=12)
    
    explored_x = []
    explored_y = []
    current_path = []
    
    def animate(frame):
        nonlocal explored_x, explored_y, current_path
        
        if frame < len(planner.explored_nodes):
            # 添加新探索的节点
            node = planner.explored_nodes[frame]
            explored_x.append(node[1])
            explored_y.append(node[0])
            explored_scatter.set_offsets(np.column_stack([explored_x, explored_y]))
            
            # 更新当前路径（如果已找到）
            if frame == len(planner.explored_nodes) - 1 and planner.path:
                path_x = [col for row, col in planner.path]
                path_y = [row for row, col in planner.path]
                path_line.set_data(path_x, path_y)
                path_points.set_data(path_x, path_y)
        
        return explored_scatter, path_line, path_points
    
    anim = FuncAnimation(fig, animate, frames=len(planner.explored_nodes) + 10,
                        interval=interval, blit=True, repeat=False)
    
    plt.tight_layout()
    return anim


def main():
    """主函数：演示A*算法"""
    print("=" * 60)
    print("A*路径规划算法演示")
    print("=" * 60)
    
    # 创建网格世界
    world = GridWorld(width=30, height=30, obstacle_ratio=0.25)
    
    # 可以手动设置一些障碍物（可选）
    world.set_obstacles([(5, 5), (5, 6), (6, 5), (10, 10), (15, 15)])
    
    print(f"网格大小: {world.height} x {world.width}")
    print(f"起点: {world.start}")
    print(f"终点: {world.goal}")
    print(f"障碍物数量: {np.sum(world.grid == 1)}")
    print()
    
    # 创建A*规划器
    planner = AStarPlanner(
        grid=world.grid,
        start=world.start,
        goal=world.goal,
        heuristic_type='euclidean'  # 可选: 'euclidean', 'manhattan', 'diagonal'
    )
    
    print("开始A*搜索...")
    success, path = planner.plan()
    
    if success:
        print(f"✓ 找到路径！")
        print(f"  路径长度: {len(path)} 个节点")
        print(f"  总成本: {planner.past_cost[world.goal]:.2f}")
        print(f"  探索节点数: {len(planner.explored_nodes)}")
        print(f"  路径效率: {len(path) / len(planner.explored_nodes) * 100:.1f}%")
    else:
        print("✗ 未找到路径")
    
    print()
    
    # 可视化结果
    print("生成可视化...")
    visualize_astar(planner, world.grid, world.start, world.goal, save_animation=True)
    
    # 可选：创建动画
    print("创建动画...")
    anim = create_animation(planner, world.grid, world.start, world.goal, interval=30)
    plt.show()
    
    return planner, world


if __name__ == "__main__":
    planner, world = main()

