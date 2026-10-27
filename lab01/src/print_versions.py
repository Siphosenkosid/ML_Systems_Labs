import sys
from importlib.metadata import version, PackageNotFoundError

# Display the Python version being used
print("Python version:", sys.version.split()[0])
print("\nInstalled package versions:")

packages = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "torch",
    "torchvision",
    "torchinfo",
    "thop",
    "onnx",
    "onnxruntime",
    "mlflow",
    "memory-profiler",
    "psutil",
    "codecarbon",
    "fastapi",
    "uvicorn",
    "pytest",
    "httpx",
    "locust",
    "requests",
    "pyarrow",
    "joblib",
    "tqdm"
]

for package in packages:
    try:
        print(f"{package}: {version(package)}")
    except PackageNotFoundError:
        print(f"{package}: Not installed")
        