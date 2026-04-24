# research/gpu_test.py
# ---------------------------------------------------------
# GPU / CUDA Environment Test
#
# Run from project root:
#   python research/gpu_test.py
# ---------------------------------------------------------

import torch


def main():

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("CUDA Device Count:", torch.cuda.device_count())
        print("Current Device:", torch.cuda.current_device())
    else:
        print("No GPU detected")


if __name__ == "__main__":
    main()