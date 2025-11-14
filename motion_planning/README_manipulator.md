# 机械臂路径规划实现

基于《现代机器人学》第10章的路径规划算法实现，使用Robotics Toolbox进行运动学计算和3D可视化。

## 文件说明

### 核心模块

1. **`manipulator_model.py`** - 机械臂模型类
   - 使用Robotics Toolbox (RTB) 定义2R平面机械臂
   - 提供正向运动学计算
   - 碰撞检测功能

2. **`rrt_planner_manipulator.py`** - RRT路径规划算法
   - 实现算法10.3（RRT算法）
   - 在配置空间中搜索路径
   - 使用RTB进行碰撞检测

3. **`prm_planner_manipulator.py`** - PRM路径规划算法
   - 实现算法10.4（PRM路线图构建）
   - 两阶段：采样 + 连接
   - 使用A*在路线图中查询路径

4. **`visualizer_rtb.py`** - 3D可视化类
   - 使用RTB的Swift后端进行3D可视化
   - 支持路径动画播放
   - 支持搜索树/路线图可视化

5. **`demo_manipulator.py`** - 演示程序
   - 交互式演示RRT和PRM算法
   - 3D可视化展示

## 安装依赖

```bash
conda activate path_planning
pip install roboticstoolbox-python spatialmath-python swift-sim
```

或使用环境文件：
```bash
conda env update -f ../environment.yml
```

**注意**：需要 `numpy<2.0.0`（RTB目前不支持numpy 2.x）

## 使用方法

### 运行演示程序

```bash
cd motion_planning
python demo_manipulator.py
```

然后选择：
- `1` - 运行RRT算法演示
- `2` - 运行PRM算法演示
- `3` - 运行两者

### 单独使用模块

```python
from manipulator_model import create_2r_manipulator
from rrt_planner_manipulator import RRTPlannerManipulator

# 创建机械臂
robot = create_2r_manipulator(L1=1.0, L2=0.8)

# 定义障碍物
obstacles = [
    {'type': 'box', 'center': [0.5, 0.5, 0], 'size': [0.3, 0.3, 0.2]}
]

# 创建规划器
planner = RRTPlannerManipulator(
    manipulator=robot,
    obstacles=obstacles,
    q_start=[0.0, 0.0],
    q_goal=[np.pi/2, -np.pi/3],
    step_size=0.2,
    max_iterations=3000
)

# 执行规划
success, path = planner.plan()
```

## 算法说明

### RRT (Rapidly-exploring Random Tree)

- **适用场景**：单次查询路径规划
- **特点**：快速探索配置空间
- **参数**：
  - `step_size`: 扩展步长（配置空间）
  - `goal_bias`: 目标偏向概率（0-1）
  - `max_iterations`: 最大迭代次数

### PRM (Probabilistic Roadmap)

- **适用场景**：多次查询路径规划
- **特点**：预先构建路线图，查询快速
- **参数**：
  - `num_samples`: 采样点数量
  - `k_neighbors`: 每个节点的邻居数量

## 可视化

使用Robotics Toolbox的Swift后端进行3D可视化：

- 实时显示机械臂姿态
- 显示障碍物
- 播放路径动画
- 支持交互式旋转、缩放

**注意**：可视化需要在图形界面环境中运行。

## 技术细节

### 配置空间

- 2R机械臂的配置空间是2维：`q = [θ₁, θ₂]`
- 角度范围：`[-π, π]`
- 需要考虑角度的周期性（2π等价）

### 碰撞检测

- 使用RTB计算正向运动学
- 检查每个连杆（线段）与障碍物的碰撞
- 支持立方体和球体障碍物

### 路径插值

- 在配置空间中线性插值
- 检查插值路径上的多个点
- 确保路径无碰撞

## 扩展

### 添加更多自由度

修改 `manipulator_model.py` 中的 `Manipulator2R` 类，添加更多 `RevoluteDH` 关节。

### 添加更多障碍物类型

在 `manipulator_model.py` 的 `_check_link_obstacle_collision` 方法中添加新的障碍物类型。

### 优化算法

- RRT*：渐进最优RRT
- PRM*：渐进最优PRM
- 双向RRT：从起点和目标同时扩展

## 参考

- 《现代机器人学》第10章：运动规划
- Robotics Toolbox文档：https://petercorke.github.io/robotics-toolbox-python/

