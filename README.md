# 路径规划学习项目

本项目主要用于学习《现代机器人学》(Modern Robotics) 一书，特别是第10章"运动规划"的内容。

## 项目结构

```
path_planning/
├── MR-tablet.pdf                    # 英文原版 PDF
├── 现代机器人学 (林奇) (Z-Library).pdf  # 中文版 PDF（>50M，已排除在 Git 外）
│
├── motion_planning/                 # 运动规划学习资料
│   ├── README.md                    # 学习资料说明
│   ├── chapter10_motion_planning.tex # 第10章 LaTeX 总结
│   ├── modern_robotics_motion_planning.md  # 第10章 Markdown 摘要
│   └── review_papers.md             # 相关综述论文
│
├── astar_planner.py                 # A* 算法实现（Matplotlib 可视化）
├── astar_gym_env.py                 # A* 算法实现（Gymnasium 环境）
├── font_config.py                   # 中文字体配置
├── README_AStar.md                  # A* 算法使用说明
│
├── requirements.txt                 # Python 依赖
├── environment.yml                  # Conda 环境配置
├── setup_conda.sh                  # Conda 环境设置脚本
└── CONDA_SETUP.md                  # Conda 设置说明
```

## 快速开始

### 1. 设置环境

```bash
# 使用 Conda（推荐）
bash setup_conda.sh
conda activate path_planning

# 或使用 pip
pip install -r requirements.txt
```

### 2. 运行 A* 算法演示

```bash
# Matplotlib 版本
python astar_planner.py

# Gymnasium 版本
python astar_gym_env.py
```

## 主要内容

### A* 路径规划算法
- 2D 网格地图路径规划
- 多种启发式函数（曼哈顿、欧几里得、切比雪夫）
- 实时可视化
- Gymnasium 环境集成

### 现代机器人学学习
- 第10章：运动规划理论总结
- 配置空间 (C-space) 概念
- 图搜索算法详解
- 相关论文推荐

## 依赖

- Python 3.10+
- NumPy
- Matplotlib
- Gymnasium

详细依赖列表见 `requirements.txt` 或 `environment.yml`。

## 学习资料

- **PDF 书籍**：位于项目根目录
  - `MR-tablet.pdf` - 英文原版
  - `现代机器人学 (林奇) (Z-Library).pdf` - 中文版（>50M，已排除在 Git 外）

- **运动规划学习资料**：位于 `motion_planning/` 文件夹
  - 第10章完整总结（LaTeX 和 Markdown 格式）
  - 相关综述论文推荐
  - 详见 `motion_planning/README.md`

## 许可证

本项目仅用于学习目的。

