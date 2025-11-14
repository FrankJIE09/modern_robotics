"""
Dijkstra路径规划算法实现
基于《现代机器人学》第10章的描述
如果启发式成本到目标总是估计为零，则A*变为Dijkstra算法
"""

import numpy as np
import matplotlib.pyplot as plt
import heapq

# 导入中文字体配置和GridWorld
try:
    from font_config import init_chinese_font
    init_chinese_font(verbose=False)
except ImportError:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

from astar_planner import GridWorld, visualize_astar, create_animation


class DijkstraPlanner:
    """Dijkstra路径规划器（启发式成本为0的A*）"""
    
    def __init__(self, grid, start, goal):
        """
        初始化Dijkstra规划器
        
        参数:
            grid: 2D数组，0表示自由空间，1表示障碍物
            start: 起始位置 (row, col)
            goal: 目标位置 (row, col) 或目标区域
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.start = start
        self.goal = goal
        
        # 数据结构
        self.OPEN = []  # 优先队列：(过去成本, 节点)
        self.CLOSED = set()
        self.past_cost = {}  # 从起点到节点的最小成本
        self.parent = {}  # 父节点映射
        
        # 可视化数据
        self.explored_nodes = []
        self.path = []
        self.visualization_steps = []
    
    def _get_neighbors(self, node):
        """获取节点的邻居（4连通）"""
        row, col = node
        neighbors = []
        
        # 4个方向：上、下、左、右
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # 检查边界
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                # 检查是否为障碍物
                if self.grid[new_row, new_col] == 0:
                    neighbors.append((new_row, new_col))
        
        return neighbors
    
    def _edge_cost(self, node1, node2, parent_node=None):
        """计算从node1到node2的边成本（包括转弯成本）"""
        r1, c1 = node1
        r2, c2 = node2
        
        # 基础移动成本（直线移动）
        base_cost = 1.0
        
        # 计算转弯成本
        turn_cost = 0.0
        if parent_node is not None:
            pr, pc = parent_node
            dir1 = (r1 - pr, c1 - pc)
            dir2 = (r2 - r1, c2 - c1)
            
            if dir1 != dir2 and dir1 != (0, 0):
                turn_cost = 2.0
        
        return base_cost + turn_cost
    
    def _is_goal(self, node):
        """检查节点是否在目标集中"""
        if isinstance(self.goal, tuple):
            return node == self.goal
        else:
            return node in self.goal
    
    def plan(self, visualize=False, fig=None, ax=None):
        """
        执行Dijkstra搜索（启发式成本为0的A*）
        
        参数:
            visualize: 是否实时可视化搜索过程
            fig, ax: matplotlib图形对象（如果visualize=True）
        """
        # 初始化
        self.past_cost[self.start] = 0
        # Dijkstra: 启发式成本为0，所以只使用过去成本
        heapq.heappush(self.OPEN, (0, self.start))
        
        step = 0
        while self.OPEN:
            # 从OPEN中取出过去成本最小的节点
            past_cost_current, current = heapq.heappop(self.OPEN)
            
            # 如果已经在CLOSED中，跳过
            if current in self.CLOSED:
                continue
            
            # 添加到CLOSED
            self.CLOSED.add(current)
            self.explored_nodes.append(current)
            
            # 记录可视化状态
            if visualize:
                open_nodes = [node for _, node in self.OPEN]
                self.visualization_steps.append({
                    'step': step,
                    'current': current,
                    'open': open_nodes.copy(),
                    'closed': list(self.CLOSED.copy()),
                    'past_cost_current': self.past_cost.get(current, float('inf')),
                    'est_total_cost': past_cost_current  # Dijkstra中等于过去成本
                })
                self._plot_step(fig, ax)
            
            # 检查是否到达目标
            if self._is_goal(current):
                self.path = self._reconstruct_path(current)
                if visualize:
                    self._plot_final_path(fig, ax)
                return True, self.path
            
            # 探索邻居
            for nbr in self._get_neighbors(current):
                if nbr in self.CLOSED:
                    continue
                
                # 计算到达邻居的暂定成本
                parent_of_current = self.parent.get(current, None)
                edge_cost = self._edge_cost(current, nbr, parent_of_current)
                tentative_past_cost = self.past_cost[current] + edge_cost
                
                # 如果找到更低的成本，更新
                if nbr not in self.past_cost or tentative_past_cost < self.past_cost[nbr]:
                    self.past_cost[nbr] = tentative_past_cost
                    self.parent[nbr] = current
                    
                    # Dijkstra: 只使用过去成本（启发式成本为0）
                    heapq.heappush(self.OPEN, (tentative_past_cost, nbr))
            
            step += 1
        
        # 没有找到路径
        if visualize:
            self._plot_final_path(fig, ax)
        return False, []
    
    def _plot_step(self, fig, ax):
        """绘制当前搜索步骤"""
        if fig is None or ax is None:
            return
        
        ax.clear()
        ax.imshow(self.grid, cmap='Greys', origin='upper', alpha=0.3, vmin=0, vmax=1)
        
        start_row, start_col = self.start
        goal_row, goal_col = self.goal
        ax.plot(start_col, start_row, 'go', markersize=20, label='起点', 
                markeredgecolor='darkgreen', markeredgewidth=2, zorder=10)
        ax.plot(goal_col, goal_row, 'ro', markersize=20, label='终点',
                markeredgecolor='darkred', markeredgewidth=2, zorder=10)
        
        if self.visualization_steps:
            step_data = self.visualization_steps[-1]
            
            if step_data['closed']:
                closed_x = [col for row, col in step_data['closed']]
                closed_y = [row for row, col in step_data['closed']]
                ax.scatter(closed_x, closed_y, c='lightblue', s=100, alpha=0.7,
                          edgecolors='blue', linewidths=1.5, label='CLOSED (已探索)', zorder=3)
            
            if step_data['open']:
                open_x = [col for row, col in step_data['open']]
                open_y = [row for row, col in step_data['open']]
                ax.scatter(open_x, open_y, c='yellow', s=100, alpha=0.8,
                          edgecolors='orange', linewidths=1.5, label='OPEN (待探索)', zorder=4)
            
            current = step_data['current']
            ax.plot(current[1], current[0], 'rs', markersize=25, 
                   markeredgecolor='darkred', markeredgewidth=3, 
                   label=f'当前节点 (步骤 {step_data["step"]})', zorder=5)
            
            info_text = f'步骤: {step_data["step"]}\n'
            info_text += f'当前节点: {current}\n'
            info_text += f'过去成本: {step_data["past_cost_current"]:.2f}\n'
            info_text += f'OPEN大小: {len(step_data["open"])}\n'
            info_text += f'CLOSED大小: {len(step_data["closed"])}'
            ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(-0.5, self.rows - 0.5)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title('Dijkstra 搜索过程可视化', fontsize=14, fontweight='bold')
        ax.set_xlabel('列', fontsize=12)
        ax.set_ylabel('行', fontsize=12)
        plt.pause(0.1)
    
    def _plot_final_path(self, fig, ax):
        """绘制最终路径"""
        if fig is None or ax is None or not self.path:
            return
        path_x = [col for row, col in self.path]
        path_y = [row for row, col in self.path]
        ax.plot(path_x, path_y, 'b-', linewidth=4, label='最优路径', zorder=6)
        ax.plot(path_x, path_y, 'bo', markersize=10, zorder=6)
        plt.draw()
    
    def _reconstruct_path(self, goal_node):
        """从目标节点重构路径"""
        path = []
        current = goal_node
        while current is not None:
            path.append(current)
            current = self.parent.get(current, None)
        path.reverse()
        return path


def main():
    """主函数：演示Dijkstra算法"""
    print("=" * 60)
    print("Dijkstra路径规划算法演示")
    print("=" * 60)
    
    world = GridWorld(width=30, height=30, obstacle_ratio=0.25)
    world.set_obstacles([(5, 5), (5, 6), (6, 5), (10, 10), (15, 15)])
    
    print(f"网格大小: {world.height} x {world.width}")
    print(f"起点: {world.start}")
    print(f"终点: {world.goal}")
    print(f"障碍物数量: {np.sum(world.grid == 1)}")
    print()
    
    planner = DijkstraPlanner(
        grid=world.grid,
        start=world.start,
        goal=world.goal
    )
    
    print("开始Dijkstra搜索...")
    
    fig, ax = plt.subplots(figsize=(14, 12))
    plt.ion()
    
    success, path = planner.plan(visualize=True, fig=fig, ax=ax)
    
    plt.ioff()
    
    if success:
        print(f"✓ 找到路径！")
        print(f"  路径长度: {len(path)} 个节点")
        print(f"  总成本: {planner.past_cost[world.goal]:.2f}")
        print(f"  探索节点数: {len(planner.explored_nodes)}")
        print(f"  路径效率: {len(path) / len(planner.explored_nodes) * 100:.1f}%")
    else:
        print("✗ 未找到路径")
    
    print()
    print("生成可视化...")
    visualize_astar(planner, world.grid, world.start, world.goal, save_animation=True)
    
    print("创建动画...")
    anim = create_animation(planner, world.grid, world.start, world.goal, interval=30)
    plt.show()
    
    return planner, world


if __name__ == "__main__":
    planner, world = main()

