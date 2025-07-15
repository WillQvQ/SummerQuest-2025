#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hw3_checker.py - 检查 hw3_2.json 格式数据的 Output 字段

检查项目：
1. 是否包含 think 部分
2. 除think外展示给用户的部分，是否含有特殊词符 <|EDIT|> 和 <|AGENT|> 之一
3. <|AGENT|> 后是否正确调用函数 python
4. <|EDIT|> 后是否调用函数 editor

使用方法：
    python hw3_checker.py [文件路径]
    
    如果不指定文件路径，默认检查 hw3_2.json
"""

import json
import re
import sys
import argparse
from typing import Dict, List, Tuple

def extract_think_content(output: str) -> Tuple[str, str]:
    """
    提取 think 部分和非 think 部分的内容
    
    Args:
        output: 完整的输出字符串
        
    Returns:
        tuple: (think_content, non_think_content)
    """
    # 匹配 <think>...</think> 标签
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, output, re.DOTALL)
    
    # 提取 think 内容
    think_content = '\n'.join(think_matches) if think_matches else ''
    
    # 移除 think 部分，得到非 think 内容
    non_think_content = re.sub(think_pattern, '', output, flags=re.DOTALL).strip()
    
    return think_content, non_think_content

def check_special_markers(non_think_content: str) -> Tuple[bool, str]:
    """
    检查是否包含特殊词符 <|EDIT|> 或 <|AGENT|>
    
    Args:
        non_think_content: 非think部分的内容
        
    Returns:
        tuple: (has_marker, marker_type)
    """
    if '<|EDIT|>' in non_think_content:
        return True, 'EDIT'
    elif '<|AGENT|>' in non_think_content:
        return True, 'AGENT'
    else:
        return False, 'NONE'

def check_function_call(content: str, expected_function: str) -> Tuple[bool, str]:
    """
    检查是否正确调用了指定的函数
    
    Args:
        content: 要检查的内容
        expected_function: 期望的函数名 ('python' 或 'editor')
        
    Returns:
        tuple: (has_correct_call, details)
    """
    # 匹配 JSON 格式的函数调用
    function_call_pattern = r'{\s*"name"\s*:\s*"([^"]+)"'
    matches = re.findall(function_call_pattern, content)
    
    if matches:
        for match in matches:
            if match == expected_function:
                return True, f"找到正确的{expected_function}函数调用"
        return False, f"找到函数调用但不是{expected_function}: {matches}"
    else:
        return False, f"未找到{expected_function}函数调用"

def check_single_output(output: str, index: int) -> Dict:
    """
    检查单个输出项
    
    Args:
        output: 输出字符串
        index: 项目索引
        
    Returns:
        dict: 检查结果
    """
    result = {
        'index': index,
        'has_think': False,
        'has_special_marker': False,
        'marker_type': 'NONE',
        'correct_function_call': False,
        'function_call_details': '',
        'issues': []
    }
    
    # 1. 检查是否包含 think 部分
    think_content, non_think_content = extract_think_content(output)
    result['has_think'] = bool(think_content.strip())
    
    if not result['has_think']:
        result['issues'].append('缺少 <think> 部分')
    
    # 2. 检查特殊词符
    has_marker, marker_type = check_special_markers(non_think_content)
    result['has_special_marker'] = has_marker
    result['marker_type'] = marker_type
    
    if not has_marker:
        result['issues'].append('缺少特殊词符 <|EDIT|> 或 <|AGENT|>')
    
    # 3. 根据标记类型检查函数调用
    if marker_type == 'AGENT':
        # 检查是否调用了 python 函数
        has_correct_call, details = check_function_call(non_think_content, 'python')
        result['correct_function_call'] = has_correct_call
        result['function_call_details'] = details
        
        if not has_correct_call:
            result['issues'].append('<|AGENT|> 后未正确调用 python 函数')
            
    elif marker_type == 'EDIT':
        # 检查是否调用了 editor 函数
        has_correct_call, details = check_function_call(non_think_content, 'editor')
        result['correct_function_call'] = has_correct_call
        result['function_call_details'] = details
        
        if not has_correct_call:
            result['issues'].append('<|EDIT|> 后未正确调用 editor 函数')
    
    return result

def check_query_output_file(file_path: str) -> Dict:
    """
    检查整个 hw3_2.json 文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        dict: 完整的检查结果
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return {'error': f'文件未找到: {file_path}'}
    except json.JSONDecodeError as e:
        return {'error': f'JSON 解析错误: {e}'}
    
    if not isinstance(data, list):
        return {'error': '数据格式错误：应该是一个列表'}
    
    results = {
        'total_items': len(data),
        'passed_items': 0,
        'failed_items': 0,
        'details': [],
        'summary': {
            'missing_think': 0,
            'missing_markers': 0,
            'wrong_function_calls': 0
        }
    }
    
    for i, item in enumerate(data):
        if not isinstance(item, dict) or 'Output' not in item:
            results['details'].append({
                'index': i,
                'error': '项目格式错误：缺少 Output 字段'
            })
            results['failed_items'] += 1
            continue
        
        output = item['Output']
        check_result = check_single_output(output, i)
        results['details'].append(check_result)
        
        # 统计
        if check_result['issues']:
            results['failed_items'] += 1
            if not check_result['has_think']:
                results['summary']['missing_think'] += 1
            if not check_result['has_special_marker']:
                results['summary']['missing_markers'] += 1
            if not check_result['correct_function_call'] and check_result['marker_type'] != 'NONE':
                results['summary']['wrong_function_calls'] += 1
        else:
            results['passed_items'] += 1
    
    return results

def print_results(results: Dict, verbose: bool = False):
    """
    打印检查结果

    Args:
        results: 检查结果字典
        verbose: 是否显示详细信息
    """
    if 'error' in results:
        print(f"❌ 错误: {results['error']}")
        return
    
    print("=" * 60)
    print("📋 hw3_checker.py 检查结果")
    print("=" * 60)
    
    # 总体统计
    print(f"📊 总体统计:")
    print(f"   总项目数: {results['total_items']}")
    print(f"   ✅ 通过: {results['passed_items']}")
    print(f"   ❌ 失败: {results['failed_items']}")
    print(f"   📈 通过率: {results['passed_items']/results['total_items']*100:.1f}%")
    print()
    
    # 问题统计
    summary = results['summary']
    if any(summary.values()):
        print(f"🔍 问题统计:")
        if summary['missing_think'] > 0:
            print(f"   缺少 <think> 部分: {summary['missing_think']} 项")
        if summary['missing_markers'] > 0:
            print(f"   缺少特殊词符: {summary['missing_markers']} 项")
        if summary['wrong_function_calls'] > 0:
            print(f"   函数调用错误: {summary['wrong_function_calls']} 项")
        print()
    
    # 详细结果
    print("📝 详细检查结果:")
    for detail in results['details']:
        if 'error' in detail:
            print(f"   项目 {detail['index']}: ❌ {detail['error']}")
        elif detail['issues']:
            issues_str = ', '.join(detail['issues'])
            print(f"   项目 {detail['index']}: ❌ {issues_str}")
            
            # verbose模式下显示更多详细信息
            if verbose:
                print(f"      - 标记类型: {detail['marker_type']}")
                if detail['function_call_details']:
                    print(f"      - 函数调用: {detail['function_call_details']}")
        else:
            print(f"   项目 {detail['index']}: ✅ 通过所有检查")
            
            # verbose模式下显示通过项目的详细信息
            if verbose:
                print(f"      - 标记类型: {detail['marker_type']}")
                if detail['function_call_details']:
                    print(f"      - 函数调用: {detail['function_call_details']}")
    
    print("=" * 60)

def main():
    """
    主函数
    """
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description='检查 hw3_2.json 格式数据的 Output 字段',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""检查项目：
1. 是否包含 think 部分
2. 除think外展示给用户的部分，是否含有特殊词符 <|EDIT|> 和 <|AGENT|> 之一
3. <|AGENT|> 后是否正确调用函数 python
4. <|EDIT|> 后是否调用函数 editor

示例：
    python hw3_checker.py                           # 检查默认文件
    python hw3_checker.py data.json                 # 检查指定文件
    python hw3_checker.py /path/to/your/file.json   # 检查指定路径的文件"""
    )
    
    parser.add_argument(
        'file_path',
        nargs='?',
        default='hw3_2.json',
        help='要检查的JSON文件路径 (默认: hw3_2.json)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细的检查信息'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    file_path = args.file_path
    
    print("🚀 开始检查文件...")
    print(f"📁 文件路径: {file_path}")
    print()
    
    results = check_query_output_file(file_path)
    print_results(results, verbose=args.verbose)

if __name__ == '__main__':
    main()