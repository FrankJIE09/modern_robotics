"""
3D可视化类 - 使用Robotics Toolbox的Swift后端
用于机械臂路径规划的可视化
"""

import numpy as np

# 尝试导入Swift
try:
    from roboticstoolbox.backends.swift import Swift
    SWIFT_AVAILABLE = True
except ImportError:
    SWIFT_AVAILABLE = False
    Swift = None

from spatialmath import SE3
import time


class ManipulatorVisualizer:
    """机械臂路径规划3D可视化器"""
    
    def __init__(self, manipulator, obstacles=None):
        """
        初始化可视化器
        
        参数:
            manipulator: Manipulator2R实例
            obstacles: 障碍物列表
        """
        self.manipulator = manipulator
        self.obstacles = obstacles or []
        
        # 创建Swift环境（如果失败则使用matplotlib）
        # 注意：Swift在某些环境下可能有事件循环问题，直接使用matplotlib更稳定
        self.swift_available = False
        self.env = None
        
        # 由于Swift在后台线程中的事件循环问题，我们直接使用matplotlib
        # 如果需要Swift，可以在有图形界面的环境中单独测试
        use_swift = False  # 设置为False以禁用Swift，直接使用matplotlib
        
        if use_swift and SWIFT_AVAILABLE:
            try:
                self.env = Swift()
                # 尝试启动Swift（可能需要图形界面）
                self.env.launch(realtime=True)
                self.swift_available = True
            except Exception as e:
                # Swift启动失败，使用matplotlib
                print(f"注意：Swift可视化不可用 ({type(e).__name__})")
                print("将使用matplotlib进行3D可视化")
                self.swift_available = False
                self.env = None
        else:
            if not use_swift:
                print("使用matplotlib进行3D可视化（Swift已禁用以避免事件循环问题）")
            else:
                print("注意：Swift后端未安装，将使用matplotlib进行可视化")
        
        if self.swift_available:
            # 添加机械臂到环境
            self.env.add(manipulator.robot)
            
            # 添加障碍物到环境
            self._add_obstacles()
            
            # 更新环境
            self.env.step()
        else:
            # 使用matplotlib作为备选
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('机械臂路径规划可视化')
            plt.ion()
            plt.show()
    
    def _add_obstacles(self):
        """添加障碍物到Swift环境"""
        try:
            from swift import Box, Sphere
        except ImportError:
            # 如果swift不可用，尝试使用RTB的Shape类
            try:
                from roboticstoolbox import Box as RTBBox, Sphere as RTBSphere
                Box = RTBBox
                Sphere = RTBSphere
            except ImportError:
                print("警告：无法导入障碍物类，跳过障碍物可视化")
                return
        
        for i, obs in enumerate(self.obstacles):
            obs_type = obs.get('type', 'box')
            
            try:
                if obs_type == 'box':
                    center = obs['center']
                    size = obs['size']
                    
                    # 创建立方体
                    box = Box(size=size)
                    box.T = SE3(center)
                    self.env.add(box, name=f'obstacle_{i}')
                
                elif obs_type == 'sphere':
                    center = obs['center']
                    radius = obs['radius']
                    
                    # 创建球体
                    sphere = Sphere(radius=radius)
                    sphere.T = SE3(center)
                    self.env.add(sphere, name=f'obstacle_{i}')
            except Exception as e:
                print(f"警告：无法添加障碍物 {i}: {e}")
    
    def show_config(self, q, duration=0.1):
        """
        显示机械臂配置
        
        参数:
            q: 关节角度
            duration: 显示持续时间（秒）
        """
        if self.swift_available and self.env is not None:
            self.manipulator.robot.q = q
            self.env.step(duration)
        else:
            # 使用matplotlib可视化
            self._plot_config_2d(q)
    
    def animate_path(self, path, duration=2.0, show_tree=None, tree_edges=None):
        """
        播放路径动画
        
        参数:
            path: 路径（配置序列）
            duration: 总动画时长（秒）
            show_tree: 是否显示搜索树（RRT）
            tree_edges: 搜索树的边（用于RRT可视化）
        """
        if not path:
            print("路径为空，无法播放动画")
            return
        
        print(f"播放路径动画，共 {len(path)} 个配置点...")
        
        # 计算每步的时间
        step_time = duration / len(path)
        
        # 播放路径
        for i, q in enumerate(path):
            self.show_config(q, duration=step_time)
            
            if (i + 1) % 10 == 0:
                print(f"  进度: {i + 1}/{len(path)}")
        
        print("✓ 动画播放完成")
    
    def show_rrt_tree(self, tree_nodes, tree_edges, q_current=None):
        """
        显示RRT搜索树（简化版：只显示节点）
        
        参数:
            tree_nodes: 树节点列表
            tree_edges: 树边列表
            q_current: 当前正在扩展的节点
        """
        # 显示所有节点
        for q in tree_nodes:
            self.show_config(q, duration=0.01)
        
        # 高亮当前节点
        if q_current is not None:
            self.show_config(q_current, duration=0.1)
    
    def show_prm_roadmap(self, roadmap_nodes, roadmap_edges):
        """
        显示PRM路线图
        
        参数:
            roadmap_nodes: 路线图节点列表
            roadmap_edges: 路线图边列表
        """
        # 显示所有节点
        for q in roadmap_nodes:
            self.show_config(q, duration=0.01)
    
    def _plot_config_2d(self, q):
        """使用matplotlib绘制2D配置（平面机械臂）"""
        import matplotlib.pyplot as plt
        
        # 获取关节位置
        joint_positions, end_pos, _ = self.manipulator.forward_kinematics(q)
        
        # 清除之前的绘图
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('机械臂路径规划可视化')
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-0.5, 0.5])
        
        # 绘制障碍物
        for obs in self.obstacles:
            obs_type = obs.get('type', 'box')
            center = np.array(obs['center'])
            
            if obs_type == 'box':
                size = np.array(obs['size'])
                # 绘制立方体（简化）
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                # 创建立方体的8个顶点
                x, y, z = center
                dx, dy, dz = size / 2
                vertices = [
                    [x-dx, y-dy, z-dz], [x+dx, y-dy, z-dz],
                    [x+dx, y+dy, z-dz], [x-dx, y+dy, z-dz],
                    [x-dx, y-dy, z+dz], [x+dx, y-dy, z+dz],
                    [x+dx, y+dy, z+dz], [x-dx, y+dy, z+dz]
                ]
                faces = [
                    [0,1,2,3], [4,5,6,7], [0,1,5,4],
                    [2,3,7,6], [0,3,7,4], [1,2,6,5]
                ]
                cube = Poly3DCollection([[vertices[i] for i in face] for face in faces],
                                      alpha=0.3, facecolor='red', edgecolor='black')
                self.ax.add_collection3d(cube)
            elif obs_type == 'sphere':
                radius = obs['radius']
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                self.ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='red')
        
        # 绘制机械臂
        for i in range(len(joint_positions) - 1):
            p1 = joint_positions[i]
            p2 = joint_positions[i + 1]
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                         'b-', linewidth=3)
        
        # 绘制关节
        for i, pos in enumerate(joint_positions):
            self.ax.scatter(pos[0], pos[1], pos[2], s=100, c='blue', marker='o')
        
        # 绘制末端执行器
        self.ax.scatter(end_pos[0], end_pos[1], end_pos[2], s=150, 
                       c='green', marker='*', label='末端执行器')
        
        plt.draw()
        plt.pause(0.01)
    
    def close(self):
        """关闭可视化窗口"""
        if self.swift_available and self.env is not None:
            self.env.close()
        else:
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.close(self.fig)


def create_visualizer(manipulator, obstacles=None):
    """
    创建可视化器的便捷函数
    
    参数:
        manipulator: Manipulator2R实例
        obstacles: 障碍物列表
    
    返回:
        ManipulatorVisualizer实例
    """
    return ManipulatorVisualizer(manipulator, obstacles)

