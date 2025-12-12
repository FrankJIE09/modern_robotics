#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将第8章英文文本转换为中文LaTeX格式
"""

import re
import sys

def clean_text(text):
    """清理文本，移除特殊字符"""
    # 移除换页符
    text = text.replace('\f', '')
    # 移除多余的空白行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def format_equation(text):
    """格式化数学公式"""
    # 将行内公式用 $ 包围
    # 这里需要根据实际情况调整
    return text

def format_section(text):
    """格式化章节标题"""
    # 匹配章节标题，如 "8.1       Lagrangian Formulation"
    pattern = r'^(\d+\.\d+(?:\.\d+)?)\s+(.+)$'
    def replace(match):
        level = len(match.group(1).split('.')) - 1
        title = match.group(2).strip()
        if level == 1:
            return f'\\section{{{title}}}'
        elif level == 2:
            return f'\\subsection{{{title}}}'
        elif level == 3:
            return f'\\subsubsection{{{title}}}'
        return match.group(0)
    return re.sub(pattern, replace, text, flags=re.MULTILINE)

def format_figure(text):
    """格式化图片引用"""
    pattern = r'Figure (\d+\.\d+):\s*(.+)'
    def replace(match):
        num = match.group(1)
        caption = match.group(2).strip()
        return f'\\begin{{figure}}[h]\n\\centering\n\\includegraphics{{figure_{num.replace(".", "_")}.pdf}}\n\\caption{{{caption}}}\n\\label{{fig:{num.replace(".", "_")}}}\n\\end{{figure}}'
    return re.sub(pattern, replace, text)

def translate_key_terms(text):
    """翻译关键术语（这里只是示例，实际需要完整的翻译）"""
    translations = {
        'Chapter 8': '第8章',
        'Dynamics of Open Chains': '开链动力学',
        'Lagrangian Formulation': '拉格朗日公式',
        'Newton–Euler': '牛顿-欧拉',
        'generalized coordinates': '广义坐标',
        'kinetic energy': '动能',
        'potential energy': '势能',
        'mass matrix': '质量矩阵',
        'joint variables': '关节变量',
        'forward dynamics': '正向动力学',
        'inverse dynamics': '逆动力学',
    }
    
    for en, cn in translations.items():
        text = text.replace(en, cn)
    
    return text

def convert_to_latex(input_file, output_file):
    """主转换函数"""
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 清理文本
    content = clean_text(content)
    
    # 创建LaTeX文档头部
    latex_header = """\\documentclass[12pt,a4paper]{article}
\\usepackage[UTF8]{ctex}
\\usepackage{amsmath,amssymb,amsthm}
\\usepackage{geometry}
\\usepackage{graphicx}
\\usepackage{hyperref}

\\geometry{margin=2.5cm}

\\title{第8章：开链动力学}
\\author{Modern Robotics}
\\date{}

\\begin{document}

\\maketitle

"""
    
    # 处理内容
    # 移除页码和页眉
    content = re.sub(r'May 2017 preprint.*?http://modernrobotics\.org', '', content)
    content = re.sub(r'^\d+\s*$', '', content, flags=re.MULTILINE)
    
    # 格式化章节标题
    content = format_section(content)
    
    # 格式化图片
    content = format_figure(content)
    
    # 处理数学公式（简单处理，实际需要更复杂的处理）
    # 将独立的公式行转换为equation环境
    content = re.sub(r'^(\s*)(.+?)\s+\((\d+\.\d+)\)\s*$', 
                     r'\\begin{equation}\n\2\n\\label{eq:\3}\n\\end{equation}', 
                     content, flags=re.MULTILINE)
    
    # 添加LaTeX尾部
    latex_footer = "\n\\end{document}\n"
    
    # 组合完整文档
    full_latex = latex_header + content + latex_footer
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_latex)
    
    print(f"转换完成！输出文件：{output_file}")
    print(f"注意：这是一个初步转换，需要手动翻译文本内容为中文")

if __name__ == '__main__':
    input_file = '/tmp/chapter8_raw.txt'
    output_file = '/home/lenovo/Frank/doc/modern_robotics/chapter8_dynamics/chapter8_dynamics.tex'
    convert_to_latex(input_file, output_file)
