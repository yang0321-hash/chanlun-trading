"""
缠论交易系统 - 安装和运行脚本
在Windows中双击运行此文件
"""
import subprocess
import sys
import os

def install_requirements():
    """安装依赖"""
    print("=" * 50)
    print("正在安装依赖包...")
    print("=" * 50)

    packages = [
        "akshare>=1.12.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.18.0",
        "loguru>=0.7.0",
        "python-dotenv>=1.0.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0"
    ]

    for package in packages:
        print(f"安装 {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
            print(f"  ✓ {package}")
        except Exception as e:
            print(f"  ✗ {package}: {e}")

    print("\n依赖安装完成！")

def run_example():
    """运行示例"""
    print("\n" + "=" * 50)
    print("运行回测示例...")
    print("=" * 50)

    example_file = os.path.join(os.path.dirname(__file__), "examples", "basic_usage.py")

    if os.path.exists(example_file):
        try:
            subprocess.call([sys.executable, example_file])
        except Exception as e:
            print(f"运行出错: {e}")
    else:
        print(f"找不到示例文件: {example_file}")

if __name__ == "__main__":
    try:
        install_requirements()
        input("\n按回车键运行回测示例...")
        run_example()
    except Exception as e:
        print(f"错误: {e}")
    finally:
        input("\n按回车键退出...")
