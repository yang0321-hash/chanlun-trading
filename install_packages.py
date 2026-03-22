"""使用国内镜像源安装依赖"""
import sys
import subprocess

# 清华大学镜像源
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

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

print("使用清华大学镜像源安装依赖...")
print(f"镜像: {MIRROR}")
print()

for package in packages:
    print(f"安装 {package}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-i", MIRROR, package, "--trusted-host", "pypi.tuna.tsinghua.edu.cn"
        ])
        print(f"  [OK] {package}\n")
    except Exception as e:
        print(f"  [FAIL] {package}: {e}\n")

print("\n安装完成！")
input("按回车键退出...")
