# 运动规划 (Motion Planning) 学习资料

本文件夹包含《现代机器人学》第10章"运动规划"相关的学习资料和笔记。

## 文件说明

### 学习笔记
- `chapter10_motion_planning.tex` - 第10章"运动规划"的完整 LaTeX 格式总结
  - 包含所有小节内容
  - 包含算法伪代码（A* 算法等）
  - 包含数学公式和概念解释

- `modern_robotics_motion_planning.md` - 第10章"运动规划"的 Markdown 格式摘要
  - 章节主要内容概述
  - 关键概念总结

- `review_papers.md` - 运动规划领域的相关综述论文推荐
  - 经典综述论文
  - 最新研究进展
  - 重要研究方向

## 主要内容

### 第10章：运动规划 (Motion Planning)

#### 核心概念
- **配置空间 (C-space)**：机器人所有可能配置的集合
- **C-障碍物 (C-obstacles)**：配置空间中与物理障碍物对应的区域
- **自由 C-空间 (Free C-space)**：机器人可以安全移动的配置空间区域

#### 规划方法
1. **图搜索算法**
   - A* 算法
   - Dijkstra 算法
   - 广度优先搜索

2. **采样方法**
   - RRT (Rapidly-exploring Random Tree)
   - PRM (Probabilistic Roadmap Method)

3. **虚拟势场法**
   - 导航函数
   - 工作空间势场

4. **优化方法**
   - 非线性优化
   - 平滑处理

#### 规划问题类型
- 在线 vs. 离线规划
- 最优 vs. 满足性规划
- 精确 vs. 近似规划
- 有障碍物 vs. 无障碍物规划

## 相关实现

本项目的 A* 算法实现位于项目根目录：
- `astar_planner.py` - A* 算法实现（Matplotlib 可视化）
- `astar_gym_env.py` - A* 算法实现（Gymnasium 环境）
- `README_AStar.md` - A* 算法使用说明

## 参考书籍

- **Modern Robotics: Mechanics, Planning, and Control** (Kevin M. Lynch, Frank C. Park)
- PDF 文件位于项目根目录

