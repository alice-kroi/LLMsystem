#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境测试脚本
用于检查项目所需的所有依赖库是否已正确安装
"""

import sys
import subprocess
from typing import Dict, List, Tuple

# 定义项目所需的库及其可选性
REQUIRED_LIBRARIES = [
    ('yaml', False, 'PyYAML', '用于解析YAML配置文件'),
    ('langgraph', True, 'langgraph', '用于创建和管理Agent工作流'),
    ('logging', False, 'logging', 'Python标准库，用于日志记录'),
    ('os', False, 'os', 'Python标准库，用于文件路径操作'),
    ('typing', False, 'typing', 'Python标准库，用于类型注解'),
    ('sys', False, 'sys', 'Python标准库，用于系统操作'),
]


def check_library_installed(lib_name: str) -> Tuple[bool, str]:
    """
    检查指定的库是否已安装
    
    Args:
        lib_name: 要检查的库名称
        
    Returns:
        Tuple[bool, str]: (是否安装, 错误信息)
    """
    try:
        __import__(lib_name)
        return True, ""
    except ImportError as e:
        return False, str(e)


def check_pip_installed() -> bool:
    """
    检查pip是否可用
    
    Returns:
        bool: pip是否可用
    """
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '--version'], 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def get_python_version() -> str:
    """
    获取Python版本信息
    
    Returns:
        str: Python版本字符串
    """
    return f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def main() -> None:
    """
    主函数，执行所有环境检查
    """
    print("=" * 60)
    print("环境检查工具".center(50))
    print("=" * 60)
    
    # 打印Python版本信息
    print(f"Python版本: {get_python_version()}")
    
    # 检查pip是否可用
    pip_available = check_pip_installed()
    print(f"pip可用: {'✓' if pip_available else '✗'}")
    
    print("\n依赖库检查:")
    print("-" * 60)
    
    # 统计信息
    installed_count = 0
    not_installed_count = 0
    optional_not_installed = 0
    
    # 检查每个库
    for lib_name, is_optional, package_name, description in REQUIRED_LIBRARIES:
        status, error_msg = check_library_installed(lib_name)
        
        if status:
            installed_count += 1
            status_str = "✓ 已安装"
            color_code = "\033[92m"  # 绿色
        else:
            if is_optional:
                optional_not_installed += 1
                status_str = "⚠ 可选库未安装"
                color_code = "\033[93m"  # 黄色
            else:
                not_installed_count += 1
                status_str = "✗ 未安装 - 必需"
                color_code = "\033[91m"  # 红色
        
        # Windows终端可能不支持颜色，这里使用简单的输出
        print(f"{lib_name:<15} | {status_str:<20} | {description}")
        if not status:
            print(f"    错误信息: {error_msg}")
            print(f"    安装命令: pip install {package_name}")
    
    print("-" * 60)
    print(f"总计: {installed_count} 已安装, {not_installed_count} 必需库未安装, {optional_not_installed} 可选库未安装")
    
    # 显示总结
    print("\n总结:")
    if not_installed_count > 0:
        print("❌ 环境检查失败: 存在必需的库未安装，请安装缺少的库后再试")
        print("  安装命令示例:")
        for lib_name, is_optional, package_name, _ in REQUIRED_LIBRARIES:
            if not is_optional and not check_library_installed(lib_name)[0]:
                print(f"    pip install {package_name}")
    elif optional_not_installed > 0:
        print("⚠️  环境检查警告: 所有必需库已安装，但存在未安装的可选库")
        print("  部分功能可能受限，如果需要完整功能，可考虑安装这些可选库")
    else:
        print("✅ 环境检查成功: 所有必需库已正确安装")
    
    print("\n提示: 运行此脚本后，您可以继续运行项目中的其他组件")
    print("=" * 60)


if __name__ == "__main__":
    main()