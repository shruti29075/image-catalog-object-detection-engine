import sys
import os

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

packages = ['PIL', 'numpy', 'cv2', 'torch', 'scipy', 'matplotlib', 'streamlit']
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'unknown')
        print(f"  [x] {pkg}: {ver}")
    except ImportError:
        print(f"  [ ] {pkg}: NOT installed")
