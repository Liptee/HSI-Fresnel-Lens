import os
from pathlib import Path 
import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat, savemat
import torch
import config


def fresnel_transfer_function(z, wavelength_arg, FX, FY):
    return torch.exp(-1j * torch.pi * wavelength_arg * z * (FX**2 + FY**2))


def generate_lens_phase(wavelength_arg_m, X, Y):
    phi = -torch.pi / (wavelength_arg_m * config.FOCAL_LENGTH) * (X**2 + Y**2)
    return torch.exp(1j * phi)


def read_spectral_filters_from_txt(folder_path: str, pattern: str = "*.txt", *, dtype=np.float32):
    paths = sorted(Path(folder_path).glob(pattern))
    filters = []
    for p in paths:
        data = np.loadtxt(p, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        if data.ndim != 2 or data.shape[1] < 2:
            continue
        x, y = data[:, 0], data[:, 1]
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if np.any(np.diff(x) <= 0):
            idx = np.argsort(x)
            x, y = x[idx], y[idx]
        filters.append((x.astype(dtype), y.astype(dtype)))
    return filters


def _interp_on_grid(xy_list, grid):
    Ys = []
    for x, y in xy_list:
        f = interp1d(x.astype(np.float32), y.astype(np.float32),
                    kind="linear", bounds_error=False, fill_value=0.0, assume_sorted=True)
        Ys.append(f(grid))
    return Ys


def build_cmv4000_rgb_interps(filters_xy: list[tuple[np.ndarray, np.ndarray]]):
    """
    Возвращает три интерполятора (fB, fG, fR): nm -> weight (0..1).
    Если файлов меньше 3 — дублируем то, что есть.
    Порядок файлов считаем B, G, R.
    """
    grid = np.arange(300.0, 901.0, 1.0, dtype=np.float32)
    if len(filters_xy) == 0:
        flat = np.ones_like(grid, dtype=np.float32)
        f = interp1d(grid, flat, kind="linear", bounds_error=False, fill_value=0.0, assume_sorted=True)
        return (f, f, f)

    pads = [filters_xy[i % len(filters_xy)] for i in range(3)]
    Ys = _interp_on_grid(pads, grid)  # список из 3 массивов
    fB = interp1d(grid, Ys[0].astype(np.float32), kind="linear",
                bounds_error=False, fill_value=0.0, assume_sorted=True)
    fG = interp1d(grid, Ys[1].astype(np.float32), kind="linear",
                bounds_error=False, fill_value=0.0, assume_sorted=True)
    fR = interp1d(grid, Ys[2].astype(np.float32), kind="linear",
                bounds_error=False, fill_value=0.0, assume_sorted=True)
    return (fB, fG, fR)


def build_cmv4000_mono_interp(filters_xy: list[tuple[np.ndarray, np.ndarray]]):
    if not filters_xy:
        return lambda nm: np.float32(1.0)

    grid = np.arange(300.0, 901.0, 1.0, dtype=np.float32)  # 300..900 нм
    Ys = []
    for x, y in filters_xy:
        f = interp1d(x.astype(np.float32), y.astype(np.float32),
                    kind="linear", bounds_error=False, fill_value=0.0, assume_sorted=True)
        Ys.append(f(grid))
    mono = np.mean(np.stack(Ys, axis=0), axis=0)  # усреднение
    fmono = interp1d(grid, mono.astype(np.float32),
                    kind="linear", bounds_error=False, fill_value=0.0, assume_sorted=True)
    return fmono

def load_mat_cube(mat_path: str):
    m = loadmat(mat_path)
    cube = None
    for k in ['cube','hsi','HSI','data','I','img','X']:
        if k in m:
            cube = m[k]
            break
    if cube is None:
        raise ValueError("В .mat не найден гиперкуб (cube/hsi/HSI/data/I/img/X)")
    cube = np.asarray(cube)
    if cube.ndim == 2:
        cube = cube[..., None]
    if cube.ndim != 3:
        raise ValueError(f"Ожидаю (H,W,B), получил shape={cube.shape}")

    wl = None
    for k in ['wavelengths','wl','lambda','bands']:
        if k in m:
            wl = np.asarray(m[k]).squeeze()
            break
    if wl is None:
        try:
            with open(config.PATH_TO_WL, 'r') as file:
                wl = np.array([float(line.strip()) for line in file.readlines()])
        except:
            raise ValueError("В .mat не найден вектор длин волн (wavelengths/wl/lambda/bands)")
    wl = wl.astype(np.float32).reshape(-1)
    if np.nanmax(wl) < 50.0:
        wl = wl * 1000.0

    cube = cube.astype(np.float32)
    vmax = np.nanmax(cube)
    if vmax > 0:
        cube = cube / vmax
    return cube, wl

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    lens_phases = []
    for lam_l in config.LAMBDA_FOR_LENS:
        lens_phases.append(lam_l * 1e-6)

        filters_xy = read_spectral_filters_from_txt(config.PATH_TO_SPECTRAL_FILTERS, "*.txt")
        if config.QE_MODE == "per_rgb":
            fB, fG, fR = build_cmv4000_rgb_interps(filters_xy)
        elif config.QE_MODE == "mono":
            cmv_interp = build_cmv4000_mono_interp(filters_xy)
        else:
            raise ValueError(f"Неизвестный режим: {config.QE_MODE}")

    for input_mat in config.TARGET_FILES:
        output_mat = input_mat.replace("/HSI", config.DIR_TO_SAVE)
        
        dir_to_save = output_mat.split("/")[:-1]
        dir_to_save = "/".join(dir_to_save)
        os.makedirs(dir_to_save, exist_ok=True)

        cube_np, wl_nm = load_mat_cube(input_mat)
        Hn, Wn, B = cube_np.shape

        y = torch.linspace(-Hn//2, Hn//2 - 1, Hn, device=config.DEVICE) * config.PIXEL_SIZE
        x = torch.linspace(-Wn//2, Wn//2 - 1, Wn, device=config.DEVICE) * config.PIXEL_SIZE
        Y, X = torch.meshgrid(y, x, indexing="ij")

        fy = torch.fft.fftfreq(Hn, d=config.PIXEL_SIZE).to(config.DEVICE)
        fx = torch.fft.fftfreq(Wn, d=config.PIXEL_SIZE).to(config.DEVICE)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")

        lens_P = []
        for lam_m in lens_phases:
            P = generate_lens_phase(lam_m, X, Y).to(torch.complex64)
            lens_P.append(P)

        out_cube = torch.zeros((Hn, Wn, B), dtype=torch.float32, device=config.DEVICE)

        cube_t = torch.from_numpy(cube_np).to(config.DEVICE)  # (H,W,B)
        for k in range(B):
            lam_nm = float(wl_nm[k])
            lam_m  = np.float32(lam_nm * 1e-9)

            # H(λ_k, z)
            Hlam = fresnel_transfer_function(config.Z1, lam_m, FX, FY)

            # Первый проход Френеля
            field0_fft = torch.fft.fft2(cube_t[..., k].to(torch.complex64))
            field1 = torch.fft.ifft2(field0_fft * Hlam)

            if config.QE_MODE == "per_rgb":
                # 2→B, 1→G, 0→R
                lens_to_color = {0: "R", 1: "G", 2: "B"}
                wB = float(fB(np.float32(lam_nm)))
                wG = float(fG(np.float32(lam_nm)))
                wR = float(fR(np.float32(lam_nm)))
                if (wB + wG + wR) == 0.0:
                    continue  # out of QE range

                acc = torch.zeros((Hn, Wn), dtype=torch.float32, device=config.DEVICE)
                for idx, P in enumerate(lens_P):
                    field2 = field1 * P
                    field3 = torch.fft.ifft2(torch.fft.fft2(field2) * Hlam)
                    inten = torch.abs(field3)
                    w = wR if lens_to_color[idx] == "R" else (wG if lens_to_color[idx] == "G" else wB)
                    acc = acc + inten * w
                out_cube[..., k] = acc.square()

            else:
                w_sensor = float(cmv_interp(np.float32(lam_nm)))
                if w_sensor <= 0.0:
                    continue
                acc = torch.zeros((Hn, Wn), dtype=torch.float32, device=config.DEVICE)
                for P in lens_P:
                    field2 = field1 * P
                    field3 = torch.fft.ifft2(torch.fft.fft2(field2) * Hlam)
                    acc = acc + torch.abs(field3)
                out_cube[..., k] = (acc * w_sensor).square()

        out_np = out_cube.detach().cpu().numpy()
        savemat(output_mat, {"cube": out_np, "wavelengths": wl_nm.astype(np.float32)})
        print(f"Готово: {output_mat}  shape={out_np.shape}")