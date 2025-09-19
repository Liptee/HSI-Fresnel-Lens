from glob import glob
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_DIR = "/Users/mac/Desktop/Hydra/CAVE_drive-download-20250502T181816Z-1-001/Train/HSI"
TARGET_FILES = glob(f"{TARGET_DIR}/*.mat")
print(TARGET_FILES)
QE_MODE = "per_rgb"
DIR_TO_SAVE = "/PER_RGB"
PATH_TO_SPECTRAL_FILTERS = "/Users/mac/Desktop/Hydra/for_git/cmv_400_graph"
PATH_TO_WL = "wl_cave.txt"
PIXEL_SIZE = 10e-6
FOCAL_LENGTH = 0.3
Z1 = 0.6
HIGH_LIMIT_WAVELENGTH = 800
N = 1.62
H = 4.38
M = 6

LAMBDA_FOR_LENS = []
for m in range(1, M + 1):
    cand_nm = ((N - 1) * H / m) * 1e3
    if cand_nm < HIGH_LIMIT_WAVELENGTH:
        LAMBDA_FOR_LENS.append((N - 1) * H / m)