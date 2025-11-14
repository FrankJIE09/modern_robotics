"""
机械臂模型类 - 使用 Robotics Toolbox
基于《现代机器人学》第10章
"""

import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
import matplotlib.pyplot as plt

# 导入中文字体配置
try:
    from font_config import init_chinese_font
    init_chinese_font(verbose=False)
except ImportError:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False


class Manipulator2R:
    """2R平面机械臂模型（使用Robotics Toolbox）"""
    
    def __init__(self, L1=1.0, L2=0.8, base_pos=[0, 0, 0]):
        """
        初始化2R机械臂
        
        参数:
            L1: 第一个连杆长度
            L2: 第二个连杆长度
            base_pos: 基座位置 [x, y, z]
        """
        self.L1 = L1
        self.L2 = L2
        self.base_pos = np.array(base_pos)
        
        # 使用RTB创建机械臂模型
        # RevoluteDH参数: a(连杆长度), d(偏移), alpha(扭转角)
        self.robot = DHRobot([
            RevoluteDH(a=L1, alpha=0),   # 第一个关节：绕z轴旋转
            RevoluteDH(a=L2, alpha=0)    # 第二个关节：绕z轴旋转
        ], name='2R Manipulator')
        
        # 设置关节限位（可选）
        self.robot.qlim = np.array([
            [-np.pi, np.pi],  # 第一个关节：-180°到180°
            [-np.pi, np.pi]   # 第二个关节：-180°到180°
        ])
    
    def forward_kinematics(self, q):
        """
        计算正向运动学
        
        参数:
            q: 关节角度 [theta1, theta2] (弧度)
        
        返回:
            joint_positions: 所有关节位置列表
            end_effector_pos: 末端执行器位置 [x, y, z]
            end_effector_pose: 末端执行器位姿矩阵
        """
        q = np.array(q)
        
        # 使用RTB计算正向运动学
        T = self.robot.fkine(q)
        
        # 提取末端执行器位置
        end_effector_pos = T.t
        
        # 计算所有关节位置（用于可视化）
        joint_positions = []
        
        # 基座位置
        joint_positions.append(self.base_pos.copy())
        
        # 第一个关节位置（在基座）
        joint_positions.append(self.base_pos.copy())
        
        # 第二个关节位置（第一个连杆末端）
        # 使用RTB的A矩阵计算
        T1 = self.robot.A(0, q[0])
        joint2_pos = T1.t + self.base_pos
        joint_positions.append(joint2_pos)
        
        # 末端执行器位置
        end_pos = T.t + self.base_pos
        joint_positions.append(end_pos)
        
        return joint_positions, end_pos, T
    
    def check_collision(self, q, obstacles):
        """
        检查配置是否与障碍物碰撞
        
        参数:
            q: 关节角度 [theta1, theta2]
            obstacles: 障碍物列表（每个障碍物是字典，包含'type'和参数）
        
        返回:
            is_collision: 是否碰撞
            collision_info: 碰撞信息
        """
        # 获取所有关节位置
        joint_positions, _, _ = self.forward_kinematics(q)
        
        # 检查每个连杆是否与障碍物碰撞
        for i in range(len(joint_positions) - 1):
            link_start = joint_positions[i]
            link_end = joint_positions[i + 1]
            
            # 检查连杆与每个障碍物的碰撞
            for obs in obstacles:
                if self._check_link_obstacle_collision(link_start, link_end, obs):
                    return True, {'link': i, 'obstacle': obs}
        
        return False, None
    
    def _check_link_obstacle_collision(self, start, end, obstacle):
        """
        检查连杆（线段）是否与障碍物碰撞
        
        参数:
            start: 线段起点 [x, y, z]
            end: 线段终点 [x, y, z]
            obstacle: 障碍物字典
        
        返回:
            是否碰撞
        """
        obs_type = obstacle.get('type', 'box')
        
        if obs_type == 'box':
            # 立方体障碍物
            center = np.array(obstacle['center'])
            size = np.array(obstacle['size'])
            
            # 简化的碰撞检测：检查线段是否与立方体相交
            # 这里使用AABB（轴对齐包围盒）检测
            return self._line_aabb_intersection(start, end, center, size)
        
        elif obs_type == 'sphere':
            # 球体障碍物
            center = np.array(obstacle['center'])
            radius = obstacle['radius']
            
            # 检查线段到球心的最短距离
            return self._line_sphere_intersection(start, end, center, radius)
        
        return False
    
    def _line_aabb_intersection(self, start, end, box_center, box_size):
        """线段与轴对齐包围盒的碰撞检测"""
        # 简化实现：检查线段端点是否在包围盒内
        # 或线段是否与包围盒相交
        
        # 计算包围盒边界
        min_bound = box_center - box_size / 2
        max_bound = box_center + box_size / 2
        
        # 检查端点是否在包围盒内
        if (np.all(start >= min_bound) and np.all(start <= max_bound)) or \
           (np.all(end >= min_bound) and np.all(end <= max_bound)):
            return True
        
        # 简化的线段-包围盒相交检测
        # 这里使用简化的方法，实际应该使用更精确的算法
        t_min = 0.0
        t_max = 1.0
        
        for i in range(3):
            if abs(end[i] - start[i]) < 1e-6:
                # 线段平行于该轴
                if start[i] < min_bound[i] or start[i] > max_bound[i]:
                    return False
            else:
                t1 = (min_bound[i] - start[i]) / (end[i] - start[i])
                t2 = (max_bound[i] - start[i]) / (end[i] - start[i])
                
                if t1 > t2:
                    t1, t2 = t2, t1
                
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                
                if t_min > t_max:
                    return False
        
        return True
    
    def _line_sphere_intersection(self, start, end, sphere_center, radius):
        """线段与球体的碰撞检测"""
        # 计算线段到球心的最短距离
        line_dir = end - start
        line_len = np.linalg.norm(line_dir)
        
        if line_len < 1e-6:
            # 线段退化为点
            dist = np.linalg.norm(start - sphere_center)
            return dist < radius
        
        line_dir = line_dir / line_len
        
        # 从球心到线段起点的向量
        to_start = start - sphere_center
        
        # 投影到线段方向
        proj = np.dot(to_start, line_dir)
        
        # 最近点
        if proj < 0:
            closest = start
        elif proj > line_len:
            closest = end
        else:
            closest = start + proj * line_dir
        
        # 检查距离
        dist = np.linalg.norm(closest - sphere_center)
        return dist < radius
    
    def get_workspace_bounds(self):
        """
        获取工作空间边界
        
        返回:
            bounds: 工作空间边界 [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        # 2R机械臂的工作空间是环形
        r_inner = abs(self.L1 - self.L2)
        r_outer = self.L1 + self.L2
        
        return np.array([
            -r_outer, r_outer,  # x范围
            -r_outer, r_outer,  # y范围
            -0.1, 0.1           # z范围（平面机械臂）
        ]) + np.array([self.base_pos[0], self.base_pos[0],
                       self.base_pos[1], self.base_pos[1],
                       self.base_pos[2], self.base_pos[2]])


def create_2r_manipulator(L1=1.0, L2=0.8):
    """
    创建2R机械臂的便捷函数
    
    参数:
        L1: 第一个连杆长度
        L2: 第二个连杆长度
    
    返回:
        Manipulator2R实例
    """
    return Manipulator2R(L1=L1, L2=L2)


if __name__ == "__main__":
    # 测试机械臂模型
    print("=" * 60)
    print("测试2R机械臂模型")
    print("=" * 60)
    
    robot = create_2r_manipulator(L1=1.0, L2=0.8)
    
    # 测试正向运动学
    q_test = [np.pi/4, np.pi/6]
    print(f"\n测试配置: q = {q_test}")
    
    joint_pos, end_pos, T = robot.forward_kinematics(q_test)
    print(f"关节位置数量: {len(joint_pos)}")
    print(f"末端执行器位置: {end_pos}")
    print(f"末端执行器位姿矩阵:\n{T}")
    
    # 测试碰撞检测
    obstacles = [
        {'type': 'box', 'center': [0.5, 0.5, 0], 'size': [0.3, 0.3, 0.2]},
        {'type': 'sphere', 'center': [-0.5, 0.3, 0], 'radius': 0.2}
    ]
    
    is_collision, info = robot.check_collision(q_test, obstacles)
    print(f"\n碰撞检测: {'碰撞' if is_collision else '无碰撞'}")
    if is_collision:
        print(f"碰撞信息: {info}")
    
    print("\n✓ 机械臂模型测试完成！")

