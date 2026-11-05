import sys
import platform

import torch
import transformers
import datasets
import evaluate
import sklearn
import pandas


def check_environment():

    print("=" * 55)
    print("LEGAL CLAUSE EXTRACTION - ENVIRONMENT CHECK")
    print("=" * 55)

    print(f"Python Version       : {sys.version.split()[0]}")
    print(f"Platform             : {platform.platform()}")
    print(f"PyTorch Version      : {torch.__version__}")
    print(f"Transformers Version : {transformers.__version__}")
    print(f"Datasets Version     : {datasets.__version__}")
    print(f"Evaluate Version     : {evaluate.__version__}")
    print(f"Scikit-learn Version : {sklearn.__version__}")
    print(f"Pandas Version       : {pandas.__version__}")

    print("\nDEVICE INFORMATION")
    print("-" * 55)

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("CUDA Available       : Yes")
        print(f"GPU                  : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version         : {torch.version.cuda}")

    else:

        device = torch.device("cpu")

        print("CUDA Available       : No")
        print("GPU                  : Not detected")
        print("Using Device         : CPU")

    test_tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)

    print(f"\nTensor Device        : {test_tensor.device}")

    print("\nENVIRONMENT STATUS")
    print("-" * 55)
    print("All required packages imported successfully.")
    print("Environment verification completed.")

    print("=" * 55)


if __name__ == "__main__":
    check_environment()