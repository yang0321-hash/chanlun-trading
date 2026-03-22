"""测试依赖是否安装成功"""
import sys

def test_imports():
    packages = [
        ('akshare', 'AKShare'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('plotly', 'Plotly'),
        ('loguru', 'Loguru'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib')
    ]

    print("=" * 50)
    print("Testing Package Installation")
    print("=" * 50)

    failed = []
    for module, name in packages:
        try:
            __import__(module)
            print(f"[OK] {name}")
        except ImportError as e:
            print(f"[FAIL] {name}: {e}")
            failed.append(name)

    print("=" * 50)

    if failed:
        print(f"\nMissing packages: {', '.join(failed)}")
        print("\nInstalling missing packages...")
        import subprocess
        for module, name in packages:
            if name in failed:
                print(f"Installing {module}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", module, "-q"])
                print(f"  [OK] {name}")
        print("\nAll packages installed!")
    else:
        print("\nAll packages are already installed!")

    print("\nPython version:", sys.version)
    print("=" * 50)

if __name__ == "__main__":
    test_imports()
    input("\nPress Enter to exit...")
