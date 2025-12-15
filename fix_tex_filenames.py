#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复一些标题提取不完整的文件
"""

import os
import re
from pathlib import Path

# 需要手动修复的文件映射
manual_fixes = {
    'chapter8_dynamics/为什么v.tex': '为什么v0等于g重力加速度.tex',
    'chapter8_dynamics/ad.tex': 'ad算子记号说明.tex',
    'chapter8_dynamics/为什么0.tex': '为什么积分项初始值为0.tex',
    'chapter8_dynamics/固定在坐标系中的向量_为什么v.tex': '固定在坐标系中的向量为什么v等于0.tex',
}

def fix_filenames(directory):
    """修复文件名"""
    directory = Path(directory)
    fixed_count = 0
    
    for old_name, new_name in manual_fixes.items():
        old_path = Path(old_name)
        if old_path.exists():
            new_path = old_path.parent / new_name
            if new_path.exists():
                print(f"目标文件已存在，跳过: {new_path}")
                continue
            try:
                old_path.rename(new_path)
                print(f"修复: {old_path.name} -> {new_path.name}")
                fixed_count += 1
            except Exception as e:
                print(f"修复失败 {old_path.name}: {e}")
    
    print(f"\n修复完成！共修复 {fixed_count} 个文件")

if __name__ == '__main__':
    fix_filenames('.')

