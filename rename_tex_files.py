#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重命名所有 .tex 文件，使用它们的中文标题作为文件名
"""

import os
import re
from pathlib import Path

def extract_title_from_tex(file_path):
    """从 .tex 文件中提取标题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 查找 \title{...} 模式
            match = re.search(r'\\title\{([^}]+)\}', content)
            if match:
                title = match.group(1)
                # 移除 LaTeX 命令和数学符号，只保留中文
                # 移除 $...$ 数学模式
                title = re.sub(r'\$[^$]+\$', '', title)
                # 移除 \bm{...} 等命令
                title = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', title)
                # 移除单独的 LaTeX 命令
                title = re.sub(r'\\[a-zA-Z]+', '', title)
                # 移除特殊字符，保留中文、英文、数字、空格
                title = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s：，。、]', '', title)
                # 清理多余空格
                title = ' '.join(title.split())
                return title.strip()
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
    return None

def title_to_filename(title):
    """将标题转换为文件名"""
    if not title:
        return None
    # 将空格替换为下划线
    filename = title.replace(' ', '_')
    # 移除冒号、逗号等标点
    filename = filename.replace('：', '_').replace('，', '_').replace('。', '_').replace('、', '_')
    # 移除多余的下划线
    filename = re.sub(r'_+', '_', filename)
    filename = filename.strip('_')
    return filename

def rename_tex_files(directory):
    """重命名目录中的所有 .tex 文件"""
    directory = Path(directory)
    renamed_count = 0
    skipped_count = 0
    
    # 获取所有 .tex 文件
    tex_files = list(directory.glob('*.tex'))
    
    print(f"在 {directory} 中找到 {len(tex_files)} 个 .tex 文件\n")
    
    for tex_file in tex_files:
        # 跳过主文件（通常包含章节名）
        if tex_file.name.startswith('chapter'):
            print(f"跳过主文件: {tex_file.name}")
            skipped_count += 1
            continue
            
        title = extract_title_from_tex(tex_file)
        if not title:
            print(f"无法提取标题: {tex_file.name}")
            skipped_count += 1
            continue
        
        new_filename = title_to_filename(title)
        if not new_filename:
            print(f"无法生成文件名: {tex_file.name} (标题: {title})")
            skipped_count += 1
            continue
        
        new_filepath = tex_file.parent / f"{new_filename}.tex"
        
        # 如果新文件名与旧文件名相同，跳过
        if new_filepath == tex_file:
            print(f"文件名已正确: {tex_file.name}")
            skipped_count += 1
            continue
        
        # 如果目标文件已存在，添加序号
        counter = 1
        original_new_filepath = new_filepath
        while new_filepath.exists():
            new_filepath = tex_file.parent / f"{new_filename}_{counter}.tex"
            counter += 1
        
        try:
            tex_file.rename(new_filepath)
            print(f"重命名: {tex_file.name} -> {new_filepath.name}")
            print(f"  标题: {title}\n")
            renamed_count += 1
        except Exception as e:
            print(f"重命名失败 {tex_file.name}: {e}\n")
            skipped_count += 1
    
    print(f"\n完成！")
    print(f"重命名: {renamed_count} 个文件")
    print(f"跳过: {skipped_count} 个文件")

if __name__ == '__main__':
    import sys
    
    # 要处理的目录列表
    directories = [
        'chapter8_dynamics',
        'chapter11_robot_control',
        'motion_planning'
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"\n{'='*60}")
            print(f"处理目录: {directory}")
            print(f"{'='*60}\n")
            rename_tex_files(directory)
        else:
            print(f"目录不存在: {directory}")

