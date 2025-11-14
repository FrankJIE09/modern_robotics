"""
机械臂路径规划演示程序
使用RRT和PRM算法，基于《现代机器人学》第10章
"""

import numpy as np
from manipulator_model import create_2r_manipulator
from rrt_planner_manipulator import RRTPlannerManipulator
from prm_planner_manipulator import PRMPlannerManipulator
from visualizer_rtb import create_visualizer


def create_obstacles(simple=False):
    """
    创建障碍物场景
    
    参数:
        simple: 如果为True，使用简化的障碍物场景（更容易找到路径）
    """
    if simple:
        # 简化场景：障碍物位置调整，不阻挡主要路径
        obstacles = [
            {'type': 'box', 'center': [0.3, 0.3, 0], 'size': [0.2, 0.2, 0.2]},
        ]
    else:
        # 完整场景：调整障碍物位置，避免完全阻挡路径
        obstacles = [
            # 立方体障碍物（位置调整）
            {'type': 'box', 'center': [0.3, 0.3, 0], 'size': [0.25, 0.25, 0.2]},
            {'type': 'box', 'center': [-0.4, 0.5, 0], 'size': [0.2, 0.2, 0.2]},
            
            # 球体障碍物
            {'type': 'sphere', 'center': [0.7, -0.4, 0], 'radius': 0.12},
        ]
    return obstacles


def demo_rrt():
    """演示RRT算法"""
    print("=" * 80)
    print("RRT路径规划演示 - 2R机械臂")
    print("=" * 80)
    print()
    
    # 创建机械臂
    robot = create_2r_manipulator(L1=1.0, L2=0.8)
    
    # 创建障碍物
    obstacles = create_obstacles()
    
    # 定义起点和目标
    q_start = [0.0, 0.0]  # 两个关节都在0度
    # 选择一个无碰撞且可达的目标配置
    q_goal = [-np.pi/4, np.pi/3]  # 目标配置（已验证无碰撞）
    
    print(f"起点配置: {q_start}")
    print(f"目标配置: {q_goal}")
    print()
    
    # 创建RRT规划器
    planner = RRTPlannerManipulator(
        manipulator=robot,
        obstacles=obstacles,
        q_start=q_start,
        q_goal=q_goal,
        step_size=0.15,      # 减小步长，更精细的探索
        goal_bias=0.2,        # 增加目标偏向，更容易到达目标
        max_iterations=5000   # 增加最大迭代次数
    )
    
    # 执行规划
    success, path = planner.plan()
    
    if success:
        print(f"\n✓ 找到路径！")
        print(f"  路径长度: {len(path)} 个配置点")
        print(f"  探索节点数: {len(planner.nodes)}")
        print(f"  树大小: {len(planner.tree)}")
        
        # 可视化
        print("\n启动3D可视化...")
        try:
            visualizer = create_visualizer(robot, obstacles)
            
            # 显示起点
            visualizer.show_config(q_start, duration=1.0)
            
            # 显示目标
            visualizer.show_config(q_goal, duration=1.0)
            
            # 播放路径动画
            visualizer.animate_path(path, duration=3.0)
            
            # 保持窗口打开
            print("\n可视化窗口已打开，按任意键关闭...")
            input()
            visualizer.close()
        except Exception as e:
            print(f"可视化错误: {e}")
            print("（可能需要在图形界面环境中运行）")
    else:
        print("\n✗ 未找到路径")
    
    return planner, path


def demo_prm():
    """演示PRM算法"""
    print("=" * 80)
    print("PRM路径规划演示 - 2R机械臂")
    print("=" * 80)
    print()
    
    # 创建机械臂
    robot = create_2r_manipulator(L1=1.0, L2=0.8)
    
    # 创建障碍物
    obstacles = create_obstacles()
    
    # 创建PRM规划器
    planner = PRMPlannerManipulator(
        manipulator=robot,
        obstacles=obstacles,
        num_samples=300,
        k_neighbors=10
    )
    
    # 构建路线图
    planner.build_roadmap()
    
    # 定义查询
    q_start = [0.0, 0.0]
    q_goal = [np.pi/2, -np.pi/3]
    
    print(f"\n查询路径:")
    print(f"  起点: {q_start}")
    print(f"  目标: {q_goal}")
    print()
    
    # 查询路径
    success, path = planner.query(q_start, q_goal)
    
    if success:
        print(f"\n✓ 找到路径！")
        print(f"  路径长度: {len(path)} 个配置点")
        print(f"  路线图节点数: {len(planner.nodes)}")
        print(f"  路线图边数: {len(planner.edges)}")
        
        # 可视化
        print("\n启动3D可视化...")
        try:
            visualizer = create_visualizer(robot, obstacles)
            
            # 显示起点和目标
            visualizer.show_config(q_start, duration=1.0)
            visualizer.show_config(q_goal, duration=1.0)
            
            # 播放路径动画
            visualizer.animate_path(path, duration=3.0)
            
            # 保持窗口打开
            print("\n可视化窗口已打开，按任意键关闭...")
            input()
            visualizer.close()
        except Exception as e:
            print(f"可视化错误: {e}")
            print("（可能需要在图形界面环境中运行）")
    else:
        print("\n✗ 未找到路径")
    
    return planner, path


def main():
    """主函数"""
    print("=" * 80)
    print("机械臂路径规划演示程序")
    print("基于《现代机器人学》第10章")
    print("=" * 80)
    print()
    print("请选择算法:")
    print("  1. RRT (快速探索随机树)")
    print("  2. PRM (概率路线图)")
    print("  3. 两者都运行")
    print()
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == '1':
        demo_rrt()
    elif choice == '2':
        demo_prm()
    elif choice == '3':
        print("\n" + "=" * 80)
        demo_rrt()
        print("\n" + "=" * 80)
        demo_prm()
    else:
        print("无效选择，运行RRT演示...")
        demo_rrt()


if __name__ == "__main__":
    main()

