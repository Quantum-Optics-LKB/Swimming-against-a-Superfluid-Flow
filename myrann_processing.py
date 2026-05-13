# %% Imports et blabla
import contrast as contrast
from PIL import Image
from scipy import ndimage
from skimage import restoration
import numpy as np
import matplotlib.pyplot as plt
import pyfftw
import multiprocessing
from cycler import cycler
from matplotlib.animation import FuncAnimation
import matplotlib.colors

# import polarTransform as pT
from matplotlib import colors, patches
import scipy as sp
from scipy.signal import detrend
from skimage.restoration import unwrap_phase
from cmcrameri import cm

# import cupy as cp
from scipy.signal import correlate2d
from scipy.ndimage import center_of_mass
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from numpy.lib.stride_tricks import sliding_window_view
from scipy import interpolate
import tqdm
import time

# import cv2
import os

# import regex as re
import velocity as velocity
import pickle
import faulthandler
import scipy.optimize as opt
from matplotlib.colors import LogNorm
import cmasher as cmr
# import polarTransform

# from findpeaks import findpeaks

faulthandler.enable()

# matplotlib.use("Qt5Agg")

pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
# try to load previous fftw wisdom
try:
    with open("fft.wisdom", "rb") as file:
        wisdom = pickle.load(file)
        pyfftw.import_wisdom(wisdom)
except FileNotFoundError:
    print("No FFT wisdom found, starting over ...")
# for dark theme
# plt.style.use("dark_background")
# plt.rcParams["figure.facecolor"] = "#00000080"
# plt.rcParams["axes.facecolor"] = "#00000080"
# plt.rcParams["savefig.facecolor"] = "#00000080"
# plt.rcParams['savefig.transparent'] = True
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Liberation Sans']
# for plots
tab_colors = [
    "tab:blue",
    "tab:orange",
    "forestgreen",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "teal",
]
fills = [
    "lightsteelblue",
    "navajowhite",
    "darkseagreen",
    "lightcoral",
    "violet",
    "indianred",
    "lavenderblush",
    "lightgray",
    "darkkhaki",
    "darkturquoise",
]
edges = tab_colors
custom_cycler = (
    (cycler(color=tab_colors))
    + (cycler(markeredgecolor=edges))
    + (cycler(markerfacecolor=fills))
)
plt.rc("axes", prop_cycle=custom_cycler)

# %% Paths and parameters
k0 = 2 * np.pi / 780e-9
L = 20e-2
d_slm = 8e-6
d_real = 3.45e-6 * 2
d_def = 1.1e-6
Nx, Ny = 2464, 2056

Nx_def, Ny_def = 2048, 2048

# %% Functions for processing


def remove_outliers_iqr(arr, factor=1.5):
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    # Index des valeurs normales
    mask = (arr >= lower_bound) & (arr <= upper_bound)
    # Index des outliers à supprimer
    outlier_indices = np.where(~mask)[0]

    # Suppression
    cleaned_arr = np.delete(arr, outlier_indices)
    return cleaned_arr, outlier_indices


def grating(m: int, n: int, theta: float = 45, pitch: int = 8) -> np.ndarray:
    grating = np.zeros((m, n), dtype=np.float32)
    c = np.cos(np.pi / 180 * theta)
    s = np.sin(np.pi / 180 * theta)
    for i in range(m):
        for j in range(n):
            grating[i, j] = c * i + s * j
            grating[i, j] %= pitch
            grating[i, j] /= pitch
    return grating


def soliton1d(x: np.ndarray, x_val: float, xi: float, A0: float = 1.0) -> np.ndarray:
    x = x - x_val
    psi = A0 * np.tanh(x / xi) / np.sqrt(1 + (x / xi) ** 2)
    return np.abs(psi)


def grating(m: int, n: int, theta: float = 45, pitch: int = 8) -> np.ndarray:
    grating = np.zeros((m, n), dtype=np.float32)
    c = np.cos(np.pi / 180 * theta)
    s = np.sin(np.pi / 180 * theta)
    for i in range(m):
        for j in range(n):
            grating[i, j] = c * i + s * j
            grating[i, j] %= pitch
            grating[i, j] /= pitch
    return grating


def split_step_propagation(psi0, dx, dy, dz, Nz, R):
    N, M = psi0.shape
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(M, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2

    psi = psi0.copy()
    psi_z = [psi.copy()]

    expK = np.exp(-1j * dz * K2 / (2 * R))

    for _ in range(Nz):
        psi_fft = np.fft.fft2(psi)
        psi_fft *= expK
        psi = np.fft.ifft2(psi_fft)
        psi_z.append(psi.copy())

    return psi_z


def Vortex(x, y, x_val, y_val, ell, xi) -> np.ndarray:
    """Generate a vortex profile.
    Args:
        m (int): Height of the pattern
        n (int): Width of the pattern
        x_val (int): Position of the soliton
        y_val (int): Position of the soliton
        ell (int): Charge of the vortex
        xi (float): xi parameter
        pitch (int, optional): Pixel pitch of the grating in px
        theta (int, optional): Angle of the grating in degrees. Defaults to 90.
    """
    x = x - x_val
    y = y - y_val
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    Psi = r / np.sqrt(r**2 + (xi / 0.83) ** 2) * np.exp(1j * ell * theta)
    Amp = np.abs(Psi)
    Phase = np.angle(Psi)

    return Amp, Phase


def rolling_std(arr: np.ndarray, window: float) -> np.ndarray:
    """
    Computes the rolling standard deviation of an array
    Args:
        arr (np.ndarray): The array
        window (float): The window in which to compute the standard deviation
    Returns:
        np.ndarray: The rolling standard deviation
    """
    arr = np.pad(arr, window // 2, mode="edge")
    # arr = arr[0:-1]
    aw = np.lib.stride_tricks.sliding_window_view(arr, window)
    avar = np.std(aw, axis=-1)
    return avar


def gaussian_fit(x, A0, x0, w0):
    temp = np.empty_like(x)
    for i in range(x.size):
        temp[i] = A0 * np.exp(-2 * (x[i] - x0) ** 2 / w0**2)
    return temp


def bigaussian_fit(x, A0, A1, x0, x1, w0, w1, offset):
    temp = np.empty_like(x)
    for i in range(x.size):
        temp[i] = offset + A0 * np.exp(-2 * (x[i] - x0) ** 2 / w0**2)
        temp[i] += A1 * np.exp(-2 * (x[i] - x1) ** 2 / w1**2)
    return temp


# %% Functions for the figures


def tf_defect(scan: str, plot: bool = False):
    """Streamplot of the velocity field.
    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """
    print("Loading data fields...")
    fields = np.load(f"{scan}/field.npy")

    print("Loading data fields ref...")
    fields_ref = np.load(f"{scan}/field_ref.npy")

    kx_fluid = np.load(f"{scan}/kx_fluid.npy")
    cs = np.load(f"{scan}/cs_quench.npy")
    xi_cs = 1 / (k0 * cs)
    betas = 2 * np.pi * kx_fluid * xi_cs

    exponent = 9
    fact = 1
    xi = 48e-6
    rad = 1
    roi_x = 512 // fact
    roi_y = roi_x
    # roi_y = roi_x - roi_x // 2
    roi_kx = 512 // 3
    roi_ky = roi_kx
    # roi_ky = roi_kx - roi_kx // 2
    N_times = fields_ref.shape[0]

    x00 = int(fields_ref.shape[-1] // 2)
    y00 = int(fields_ref.shape[-2] // 2)
    x00_ok = x00 - 150
    # x00 = x00 - 200
    # y00 = y00

    # which = np.arange(0, N_times, 10)
    which = np.array([50, 72, 99])
    # which = np.array([50, 71, 99])
    for i in tqdm.tqdm(which):
        beta = betas[i]
        for j in np.array([0]):
            field = fields[i, :, :]
            field_ref = fields_ref[i, :, :]
            field_ref_0 = fields_ref[0, :, :]

            amp0 = np.abs(field_ref_0)
            # compute amp0 barycenter
            frame_sum_x = np.sum(amp0, axis=0)
            frame_sum_x /= np.sum(frame_sum_x)
            x0 = np.arange(amp0.shape[1])
            x0_bary = np.sum(x0 * frame_sum_x)

            amp = np.abs(field)
            # compute amp barycenter
            frame_sum_x = np.sum(amp, axis=0)
            frame_sum_x /= np.sum(frame_sum_x)
            x = np.arange(amp.shape[1])
            x_bary = np.sum(x * frame_sum_x)

            shift_x = int(np.abs(x_bary - x0_bary))
            print(f"Shift x: {shift_x}")
            field_ref_0 = np.roll(field_ref_0, shift=shift_x, axis=1)
            rho_ref_0 = (
                field_ref_0.real * field_ref_0.real
                + field_ref_0.imag * field_ref_0.imag
            )

            # contrast.exp_angle_fast(field, field_ref_0)
            contrast.exp_angle_fast(field, field_ref)
            field = ndimage.gaussian_filter(field, rad)
            # field = np.roll(field, shift=150, axis=1)
            # field_tf = np.fft.fftshift(np.fft.fft2(field))

            field[np.abs(field) < 0.08 * np.max(np.abs(field))] = 0

            amp_tot = np.abs(field)
            amp_tot = amp_tot / np.max(amp_tot)
            phase_tot = np.angle(field)
            amp_tot_zoom = amp_tot[
                y00 - roi_y : y00 + roi_y,
                x00 - roi_x : x00 + roi_x,
            ]
            # amp_tot_zoom[amp_tot_zoom <= 0.08] = 0
            phase_tot_zoom = phase_tot[
                y00 - roi_y : y00 + roi_y,
                x00 - roi_x : x00 + roi_x,
            ]
            field_zoom = amp_tot_zoom * np.exp(1j * phase_tot_zoom)
            field_zoom[:, x00_ok:] = 1e-6
            amp_tot_zoom[:, x00_ok:] = 1e-6
            phase_tot_zoom[:, x00_ok:] = 1e-6
            field_tf = np.fft.fftshift(np.fft.fft2(field_zoom))

            (
                u_tot,
                u_inc,
                u_comp,
            ) = velocity.helmholtz_decomp(
                amp_tot * np.exp(1j * phase_tot), dx=d_real / xi, plot=False
            )
            u_tot_zoom_0 = u_tot[
                0,
                y00 - roi_y : y00 + roi_y,
                x00 - roi_x : x00 + roi_x,
            ]
            u_tot_zoom_1 = u_tot[
                1,
                y00 - roi_y : y00 + roi_y,
                x00 - roi_x : x00 + roi_x,
            ]
            u_inc_zomm_0 = u_inc[
                0,
                y00 - roi_y : y00 + roi_y,
                x00 - roi_x : x00 + roi_x,
            ]
            u_inc_zomm_1 = u_inc[
                1,
                y00 - roi_y : y00 + roi_y,
                x00 - roi_x : x00 + roi_x,
            ]
            u_comp_zoom_0 = u_comp[
                0,
                y00 - roi_y : y00 + roi_y,
                x00 - roi_x : x00 + roi_x,
            ]
            u_comp_zoom_1 = u_comp[
                1,
                y00 - roi_y : y00 + roi_y,
                x00 - roi_x : x00 + roi_x,
            ]

            # plt.figure()
            # plt.imshow(
            #     phase_tot_zoom,
            #     cmap="twilight_shifted",
            #     vmin=-np.pi,
            #     vmax=np.pi,
            #     interpolation="none",
            # )
            # plt.show()

            flow_tot = np.hypot(u_tot_zoom_0, u_tot_zoom_1)
            x = np.arange(amp_tot_zoom.shape[-1])
            y = np.arange(amp_tot_zoom.shape[-2])
            XX, YY = np.meshgrid(x, y)

            fig, ax = plt.subplots(1, 3, figsize=[12, 4])
            # set title
            fig.suptitle(rf"$\beta$ = {beta:.1f}")
            ax[0].imshow(
                np.abs(amp_tot_zoom),
                cmap="gray",
                interpolation="none",
                extent=[
                    -amp_tot_zoom.shape[-1] * d_real / 2 / xi,
                    amp_tot_zoom.shape[-1] * d_real / 2 / xi,
                    -amp_tot_zoom.shape[-2] * d_real / 2 / xi,
                    amp_tot_zoom.shape[-2] * d_real / 2 / xi,
                ],
            )
            ax[0].set_title("Density")
            ax[0].set_xlabel(r"$x/\xi$")
            ax[0].set_ylabel(r"$y/\xi$")
            ax[1].imshow(
                phase_tot_zoom,
                cmap="twilight_shifted",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="none",
            )
            ax[1].set_title("Velocity")
            ax[1].set_xlabel(r"$x/\xi$")
            ax[1].set_ylabel(r"$y/\xi$")
            # ax[1].streamplot(
            #     XX,
            #     YY,
            #     u_tot_zoom_0,
            #     u_tot_zoom_1,
            #     color="white",
            #     linewidth=1,
            #     density=0.5,
            # )
            # ax[0].quiver(
            #     XX[::10, ::10],
            #     YY[::10, ::10],
            #     u_tot_zoom_0[::10, ::10],
            #     u_tot_zoom_1[::10, ::10],
            #     color="black",
            #     scale=1 / (d_real / xi),
            #     width=0.0025,
            #     headwidth=3,
            # )
            field_zoom_tf = field_tf[
                field_tf.shape[-2] // 2 - roi_ky : field_tf.shape[-2] // 2 + roi_ky,
                field_tf.shape[-1] // 2 - roi_kx : field_tf.shape[-1] // 2 + roi_kx,
            ]
            kx_0 = (
                2
                * np.pi
                * np.fft.fftshift(np.fft.fftfreq(field_tf.shape[-1], d=d_real))
            ) * xi

            kx_zoom = kx_0[
                field_tf.shape[-1] // 2 - roi_kx : field_tf.shape[-1] // 2 + roi_kx
            ]

            # plt.figure()
            # plt.plot(kx_zoom, np.abs(field_zoom_tf[field_zoom_tf.shape[-2] // 2, :]))
            # plt.yscale("log")
            # plt.show()

            kx = (
                2
                * np.pi
                * np.fft.fftshift(np.fft.fftfreq(field_zoom_tf.shape[-1], d=d_real))
            ) * xi

            ky = (
                2
                * np.pi
                * np.fft.fftshift(np.fft.fftfreq(field_zoom_tf.shape[-2], d=d_real))
            ) * xi
            # extent for the FFT plot, Fourier space 1/x
            extent_k = [
                kx[0],
                kx[-1],
                ky[0],
                ky[-1],
            ]
            ax[2].imshow(
                np.abs(field_zoom_tf),
                cmap="gray",
                interpolation="none",
                extent=extent_k,
                # vmax=3e3,
                # vmin=0.5e1,
                norm=LogNorm(vmin=0.4e1, vmax=3e3),
            )
            ax[2].set_title("FFT of the field")
            ax[2].set_xlabel(r"$k_x\xi$")
            ax[2].set_ylabel(r"$k_y\xi$")
            plt.tight_layout()
            # add colorbars
            cbar0 = plt.colorbar(ax[0].images[0], ax=ax[0])
            cbar1 = plt.colorbar(ax[1].images[0], ax=ax[1])
            cbar2 = plt.colorbar(ax[2].images[0], ax=ax[2])
            plt.show()


def inc_vs_comp_plot(scan: str, plot: bool = False):
    p_tot = np.load(f"{scan}/p_tot.npy")
    p_inc = np.load(f"{scan}/p_inc.npy")
    p_comp = np.load(f"{scan}/p_comp.npy")
    p_tot_ref = np.load(f"{scan}/p_tot_ref.npy")
    p_inc_ref = np.load(f"{scan}/p_inc_ref.npy")
    p_comp_ref = np.load(f"{scan}/p_comp_ref.npy")
    bs = np.load(f"{scan}/betas.npy")

    plt.figure(figsize=[3, 2])
    plt.plot(
        bs,
        p_comp - p_comp_ref,
        "v-",
        label="Comp (f+d) - (f)",
        color="dimgray",
        markerfacecolor="lightgray",
        markeredgecolor="dimgray",
        markersize=8,
        markeredgewidth=1.5,
    )
    plt.plot(
        bs,
        p_tot - p_tot_ref,
        "s-",
        label="Tot (f+d) - (f)",
        color="tab:purple",
        markerfacecolor="thistle",
        markeredgecolor="tab:purple",
        markersize=8,
        markeredgewidth=1.5,
    )
    plt.plot(
        bs,
        p_inc - p_inc_ref,
        "^-",
        label="Inc (f+d) - (f)",
        color="tab:blue",
        markerfacecolor="lightsteelblue",
        markeredgecolor="tab:blue",
        markersize=8,
        markeredgewidth=1.5,
    )
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$p_{f+d} - p_f$ (a.u.)")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()


def fish_plot(scan: str, plot: bool = False):
    """Streamplot of the velocity field.
    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """
    print("Loading data fields...")
    fields = np.load(f"{scan}/field_fluid.npy")

    print("Loading data fields ref...")
    fields_ref = np.load(f"{scan}/field_fluid_ref.npy")

    kx_fluid = np.load(f"{scan}/kx_fluid.npy")
    cs = np.load(f"{scan}/cs_quench.npy")
    xi_cs = 1 / (k0 * cs)
    print("\n", xi_cs, "\n")
    betas = 2 * np.pi * kx_fluid * xi_cs

    exponent = 9
    # fact = 1
    fact = 4
    xi = 48e-6
    rad = 3
    roi_x = 512 // fact
    roi_y = roi_x // 2
    roi_y = roi_x // 2
    N_times = fields_ref.shape[0]

    bs = []

    which = np.arange(0, N_times, 2)
    # which = np.array([0, 20, 30, 40, 50, 60, 71, 80, 90, 99])
    which = np.array([0, 30, 50, 80, 99])
    # which = np.array([99])
    bad_points = [38, 56, 64, 70, 95]
    for i in tqdm.tqdm(which):
        if i in bad_points:
            continue
        beta = betas[i]
        bs += [beta]
        field = fields[i, :, :]
        field_nosmooth = field.copy()

        # field_ref = fields_ref[i, :, :]
        # field_ref_0 = fields_ref[0, :, :]
        # amp0 = np.abs(field_ref_0)
        # # compute amp0 barycenter
        # frame_sum_x = np.sum(amp0, axis=0)
        # frame_sum_x /= np.sum(frame_sum_x)
        # x0 = np.arange(amp0.shape[1])
        # x0_bary = np.sum(x0 * frame_sum_x)
        # amp = np.abs(field)
        # # compute amp barycenter
        # frame_sum_x = np.sum(amp, axis=0)
        # frame_sum_x /= np.sum(frame_sum_x)
        # x = np.arange(amp.shape[1])
        # x_bary = np.sum(x * frame_sum_x)
        # shift_x = int(np.abs(x_bary - x0_bary))
        # field_ref_0 = np.roll(field_ref_0, shift=shift_x, axis=1)
        # # field_ref_0 = field_ref
        # field_ref_copy = field_ref.copy()
        # contrast.exp_angle_fast(field, field_ref_0)
        # contrast.exp_angle_fast(field, field_ref)

        field = ndimage.gaussian_filter(field, rad)
        amp_tot = np.abs(field)
        amp_tot = amp_tot / np.max(amp_tot)
        phase_tot = np.angle(field)

        (
            u_tot,
            u_inc,
            u_comp,
        ) = velocity.helmholtz_decomp(
            amp_tot * np.exp(1j * phase_tot), dx=d_real / xi_cs, plot=False
        )

        amp_tot = np.abs(field_nosmooth)
        amp_tot = amp_tot / np.max(amp_tot)
        amp_tot_zoom = amp_tot[
            field.shape[-2] // 2 - roi_y : field.shape[-2] // 2 + roi_y,
            field.shape[-1] // 2 - roi_x : field.shape[-1] // 2 + roi_x,
        ]

        phase_tot_zoom = phase_tot[
            field.shape[-2] // 2 - roi_y : field.shape[-2] // 2 + roi_y,
            field.shape[-1] // 2 - roi_x : field.shape[-1] // 2 + roi_x,
        ]
        u_tot_zoom_x = u_tot[
            0,
            field.shape[-2] // 2 - roi_y : field.shape[-2] // 2 + roi_y,
            field.shape[-1] // 2 - roi_x : field.shape[-1] // 2 + roi_x,
        ]
        u_tot_zoom_y = u_tot[
            1,
            field.shape[-2] // 2 - roi_y : field.shape[-2] // 2 + roi_y,
            field.shape[-1] // 2 - roi_x : field.shape[-1] // 2 + roi_x,
        ]
        u_inc_zoom_x = u_inc[
            0,
            field.shape[-2] // 2 - roi_y : field.shape[-2] // 2 + roi_y,
            field.shape[-1] // 2 - roi_x : field.shape[-1] // 2 + roi_x,
        ]
        u_inc_zoom_y = u_inc[
            1,
            field.shape[-2] // 2 - roi_y : field.shape[-2] // 2 + roi_y,
            field.shape[-1] // 2 - roi_x : field.shape[-1] // 2 + roi_x,
        ]

        flow_inc = np.hypot(u_inc_zoom_x, u_inc_zoom_y)

        vorticity = np.gradient(u_tot_zoom_x, d_real / xi_cs, axis=-2) - np.gradient(
            u_tot_zoom_y, d_real / xi_cs, axis=-1
        )

        xx = np.linspace(
            0,
            phase_tot_zoom.shape[-1],
            phase_tot_zoom.shape[-1],
        )
        yy = np.linspace(
            0,
            phase_tot_zoom.shape[-2],
            phase_tot_zoom.shape[-2],
        )

        XX, YY = np.meshgrid(xx, yy)

        extent_roi = [
            -phase_tot_zoom.shape[-1] * d_real / (2 * xi),
            phase_tot_zoom.shape[-1] * d_real / (2 * xi),
            -phase_tot_zoom.shape[-2] * d_real / (2 * xi),
            phase_tot_zoom.shape[-2] * d_real / (2 * xi),
        ]

        fig, ax = plt.subplots(1, 2, figsize=[10, 5])
        fig.suptitle(
            rf"$\beta$ = {beta:.1f}",
            fontsize=16,
        )
        ax[0].imshow(
            amp_tot_zoom,
            cmap="gray",
        )
        ax[0].set_title("Density")
        ax[0].set_xlabel(r"$x/\xi$")
        ax[0].set_ylabel(r"$y/\xi$")
        cbar0 = plt.colorbar(ax[0].images[0], ax=ax[0])
        ax[1].imshow(
            vorticity,
            interpolation="none",
            vmin=-0.1,
            vmax=0.1,
            cmap=cm.vik,
        )
        ax[1].streamplot(
            xx,
            yy,
            u_inc_zoom_x,
            u_inc_zoom_y,
            linewidth=1.5,
            density=0.5,
            color="black",
        )
        # cv = 10
        # ax[1].quiver(
        #     XX[::cv, ::cv],
        #     YY[::cv, ::cv],
        #     u_inc_zoom_x[::cv, ::cv],
        #     u_inc_zoom_y[::cv, ::cv],
        #     color="black",
        #     pivot="mid",
        #     units="inches",
        # )
        ax[1].set_title("Inc")
        ax[1].set_xlabel(r"$x$")
        ax[1].set_ylabel(r"$y$")
        cbar1 = plt.colorbar(ax[1].images[0], ax=ax[1])
        plt.show()


def fish_crit_velo(scan):
    """Streamplot of the velocity field.
    Args:
        scan (str): File path.
        plot (bool, optional): Whether to plot. Defaults to False.
    """

    print("Loading data fields...")
    fields = np.load(f"{scan}/field.npy")

    print("Loading data fields ref...")
    fields_ref = np.load(f"{scan}/field_ref.npy")

    print("Loading data defect...")
    fields_def = np.load(f"{scan}/field_def.npy")

    x_bary_def_px = 640
    y_bary_def_px = 517
    x_bary_px = 641
    y_bary_px = 504
    shift_x = x_bary_px - x_bary_def_px
    shift_y = y_bary_px - y_bary_def_px

    fields_def = np.roll(fields_def, (shift_y, shift_x), axis=(-2, -1))

    kx_fluid = np.load(f"{scan}/kx_fluid.npy")
    cs = np.load(f"{scan}/cs_quench.npy")
    xi_cs = 1 / (k0 * cs)
    print("\n", xi_cs, "\n")
    betas = 2 * np.pi * kx_fluid * xi_cs

    # fact = 1
    fact = 4
    xi = 48e-6
    rad = 3
    roi_x = 512 // fact
    # roi_y = roi_x // 1
    roi_y = roi_x // 2
    N_times = fields_ref.shape[0]

    line = -5

    velo_x = []
    velo_d = []
    flows = []
    bs = []

    which = np.arange(0, N_times, 1)
    # which = np.array([0, 20, 30, 40, 50, 60, 71, 80, 90, 99])
    # which = np.array([0, 30, 50, 60, 71, 80, 99])
    # which = np.array([26, 33, 39])
    # which = np.array([1, 15, 26, 30, 33, 39, 50, 80, 99])
    bad_points = [38, 56, 64, 70, 95]
    for i in tqdm.tqdm(which):
        if i in bad_points:
            continue
        beta = betas[i]
        bs += [beta]

        field = fields[i, :, :]
        field_ref = fields_ref[i, :, :]
        field_ref_0 = fields_ref[0, :, :]
        field_def = fields_def[i, :, :]
        field_def = np.flip(field_def)

        amp0 = np.abs(field_ref_0)
        # compute amp0 barycenter
        frame_sum_x = np.sum(amp0, axis=0)
        frame_sum_x /= np.sum(frame_sum_x)
        x0 = np.arange(amp0.shape[1])
        x0_bary = np.sum(x0 * frame_sum_x)
        amp = np.abs(field)
        # compute amp barycenter
        frame_sum_x = np.sum(amp, axis=0)
        frame_sum_x /= np.sum(frame_sum_x)
        x = np.arange(amp.shape[1])
        x_bary = np.sum(x * frame_sum_x)
        shift_xx = int(np.abs(x_bary - x0_bary))
        field_ref_0 = np.roll(field_ref_0, shift=shift_xx, axis=1)

        # field_ref_0 = field_ref

        field_ref_copy = field_ref.copy()

        contrast.exp_angle_fast(field, field_ref_0)
        # contrast.exp_angle_fast(field, field_ref)
        field = ndimage.gaussian_filter(field, rad)
        field_ref_copy = ndimage.gaussian_filter(field_ref_copy, rad)

        amp_tot = np.abs(field)
        amp_tot = amp_tot / np.max(amp_tot)
        phase_tot = np.angle(field)

        amp_tot_ref = np.abs(field_ref_copy)
        amp_tot_ref = amp_tot_ref / np.max(amp_tot_ref)
        phase_tot_ref = np.angle(field_ref_copy)

        ampd = np.abs(field_def)
        ampd = ampd / np.max(ampd)

        (
            u_tot,
            u_inc,
            u_comp,
        ) = velocity.helmholtz_decomp(
            amp_tot * np.exp(1j * phase_tot), dx=d_real / xi_cs, plot=False
        )

        amp_tot_zoom = amp_tot[
            field.shape[-2] // 2 - roi_y : field.shape[-2] // 2 + roi_y,
            field.shape[-1] // 2 - roi_x : field.shape[-1] // 2 + roi_x,
        ]
        ampd_zoom = ampd[
            field.shape[-2] // 2 - roi_y : field.shape[-2] // 2 + roi_y,
            field.shape[-1] // 2 - roi_x : field.shape[-1] // 2 + roi_x,
        ]
        phase_tot_zoom = phase_tot[
            field.shape[-2] // 2 - roi_y : field.shape[-2] // 2 + roi_y,
            field.shape[-1] // 2 - roi_x : field.shape[-1] // 2 + roi_x,
        ]
        u_inc_zoom_x = u_inc[
            0,
            field.shape[-2] // 2 - roi_y : field.shape[-2] // 2 + roi_y,
            field.shape[-1] // 2 - roi_x : field.shape[-1] // 2 + roi_x,
        ]
        u_inc_zoom_y = u_inc[
            1,
            field.shape[-2] // 2 - roi_y : field.shape[-2] // 2 + roi_y,
            field.shape[-1] // 2 - roi_x : field.shape[-1] // 2 + roi_x,
        ]

        # flow_inc = np.hypot(u_inc_zoom_x, u_inc_zoom_y)
        flow_inc = u_inc_zoom_x
        flows += [flow_inc]
        amp_tot_zoom_x = amp_tot_zoom[
            amp_tot_zoom.shape[-2] // 2 + line - 10 : amp_tot_zoom.shape[-2] // 2
            + line
            + 10,
            :,
        ]
        amp_tot_zoom_x = np.mean(amp_tot_zoom_x, axis=0)

        x_min = np.where(ampd_zoom == np.max(ampd_zoom))[1][0]
        # x_min = np.where(amp_tot_zoom_x == np.min(amp_tot_zoom_x))[0][0]

        # fig, ax = plt.subplots(1, 3, figsize=(12, 3), layout="constrained")
        # im0 = ax[0].imshow(
        #     amp_tot_zoom,
        #     # extent=[
        #     #     -roi_x * d_real / xi_cs,
        #     #     roi_x * d_real / xi_cs,
        #     #     -roi_y * d_real / xi_cs,
        #     #     roi_y * d_real / xi_cs,
        #     # ],
        #     cmap="gray",
        #     interpolation="none",
        # )
        # ax[0].axhline(amp_tot_zoom.shape[-2] // 2 + line, color="white", linestyle="--")
        # ax[0].axvline(x_min, color="white", linestyle="--")
        # fig.colorbar(im0, ax=ax[0], shrink=0.6, label="Amplitude")
        # ax[0].set_title("Amplitude")
        # ax[0].set_xlabel(r"$x/\xi$")
        # ax[0].set_ylabel(r"$y/\xi$")
        # import matplotlib.colors as mcolors

        # norm = mcolors.TwoSlopeNorm(vmin=-0.1, vcenter=0.0, vmax=0.4)
        # im1 = ax[1].imshow(
        #     flow_inc,
        #     cmap="coolwarm",
        #     interpolation="none",
        #     norm=norm,
        # )
        # ax[1].axhline(amp_tot_zoom.shape[-2] // 2 + line, color="white", linestyle="--")
        # fig.colorbar(im1, ax=ax[1], shrink=0.6, label="Phase")
        # ax[1].set_title("Phase")
        # ax[1].set_xlabel(r"$x (px)$")
        # ax[1].set_ylabel(r"$y (px)$")
        # ax[2].plot(
        #     u_inc_zoom_x[amp_tot_zoom.shape[-2] // 2 + line, :],
        #     label="u_tot",
        #     color="tab:blue",
        # )
        # ax[2].set_xlabel(r"$x (px)$")
        # ax[2].set_ylabel(r"$u_x/c_s$")
        # ax[2].set_title(r"$\sqrt{\rho_x} \rm{v_x} / c_s$")
        # ax[2].set_ylim(-0.25, 0.6)
        # ax[2].axhline(0, color="black", linestyle="--")
        # ax[2].axvline(x_min, color="black", linestyle="--")
        # ax[2].grid()
        # plt.show()

        velo_x += [u_inc_zoom_x[amp_tot_zoom.shape[-2] // 2 + line, :]]
        velo_d += [u_inc_zoom_x[amp_tot_zoom.shape[-2] // 2 + line, x_min]]

    velo_x = np.array(velo_x)
    velo_d = np.array(velo_d)
    bs = np.array(bs)
    # np.save(f"{scan}/velo_defect.npy", velo_d)
    # np.save(f"{scan}/velo_x.npy", velo_x)
    # np.save(f"{scan}/betas.npy", bs)

    plt.figure(figsize=(3, 3))
    plt.plot(bs, velo_d, "o-", label="Measure at defect position")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$v_x/c_s$")
    plt.title("Fluid velocity at defect position")
    plt.axhline(0.3, color="red", linestyle="--", label="Critical velocity")
    plt.grid()
    plt.legend()
    plt.show()


def vortex_emission(scan):
    cs = np.load(f"{scan}/cs_quench.npy")
    xi_cs = 1 / (k0 * cs)
    tau_cs = k0 * L * cs**2

    velo_d = np.load(f"{scan}/velo_defect.npy")
    velo_x = np.load(f"{scan}/velo_x.npy")
    bs = np.load(f"{scan}/betas.npy")
    bc = 0.35
    tau = 19 * 2 * np.pi

    velo_d_r = 1 - velo_d

    peaks = sp.signal.find_peaks(velo_d_r, prominence=0.1)[0]
    peaks = peaks[:-2]
    peak_betas = bs[peaks]
    peak_velos = velo_d[peaks]

    # peak_betas = peak_betas[:-1]
    # peak_velos = peak_velos[:-1]

    # plt.figure()
    # plt.plot(bs, velo_d, "o-")
    # plt.plot(peak_betas, peak_velos, "ro", label="Vortex emission")
    # plt.axhline(bc, color="red", linestyle="--", label="Threshold")
    # plt.xlabel("Beta")
    # plt.ylabel("Fluid Velocity")
    # plt.title(f"Vortex Emission. tau_cs={tau_cs:.1f}")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    dtau = np.ones(len(peak_velos)) * tau_cs
    for i in range(len(dtau)):
        dtau[i] = dtau[i] / (i + 1)
    ftau = 1 / dtau

    def filter(x, a, b):
        return a * np.sqrt(2) * (x - bc) + b

    popt, pcov = curve_fit(filter, peak_betas, ftau, p0=[0.2, 0.1])

    plt.figure(figsize=(3, 2))
    plt.plot(
        peak_betas,
        ftau,
        "o-",
        label="Pair emission",
        # color="tab:purple",
        # markerfacecolor="thistle",
        # markeredgecolor="tab:purple",
        # markersize=8,
        # markeredgewidth=1.5,
    )
    plt.plot(
        peak_betas,
        filter(peak_betas, *popt),
        "r--",
        label=r"Fit: $a\xi\sim$" f"{popt[0]:.2f}",
    )
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$1 / \Delta \tau$")
    plt.legend()
    plt.show()


def plot_defect_traj(scan: str, plot=False) -> None:
    y = np.linspace(-Ny_def // 2 * d_def * 1e6, Ny_def // 2 * d_def * 1e6, Ny_def)
    kp = np.load(f"{scan}/kp.npy")
    ky = np.load(f"{scan}/ky.npy")
    cs = np.load(f"{scan}/cs_analytical.npy")
    betas = 2 * np.pi / (k0 * cs) * ky
    betas -= betas[0]
    positions_actual = np.load(f"{scan}/positions_actual_valid.npy")
    positions_actual -= np.min(positions_actual)
    positions_actual = np.flip(positions_actual)
    N_points = kp.shape[0]
    profile = np.load(f"{scan}/profile.npy")
    profile_fitted = np.load(f"{scan}/profile_fitted.npy")
    y_center_fit = np.load(f"{scan}/y_center_fit.npy")
    y_center_err = np.load(f"{scan}/y_center_err.npy")
    y_width_fit = np.load(f"{scan}/y_width_fit.npy")
    # remove bad points
    for i in range(y_center_fit.shape[0]):
        std = rolling_std(y_center_fit[i, :], 3)
        roi = std > 2 * np.median(std)
        y_center_fit[i, roi] = np.nan
    # fit 0 velocity component with a spline to remove 0 velocity trajectory
    # try:
    #     roi = np.logical_not(np.isnan(y_center_fit[0, :]))
    #     poly = np.polyfit(positions_actual[roi], y_center_fit[0, roi], deg=3)
    #     zero_velocity = np.polyval(poly, positions_actual)
    # except SystemError:
    #     print("Could not fit !")
    #     zero_velocity = y_center_fit[0, :]
    zero_velocity = y_center_fit[0, :]
    # plt.figure()
    # plt.plot(positions_actual, y_center_fit[0, :], label="Data", marker="o")
    # plt.plot(positions_actual, zero_velocity, label="Fit")
    # plt.xlabel("Objective position in mm")
    # plt.legend()
    # plt.show()
    fig, ax = plt.subplots(1, 2, figsize=(16, 4), layout="constrained")
    col = plt.cm.viridis(np.linspace(0, 1, N_points))
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=colors.Normalize(vmin=np.min(betas), vmax=np.max(betas)),
    )
    col_face = [[col[i][0], col[i][1], col[i][2], 1] for i in range(N_points)]
    trajectory_raw = np.zeros_like(y_center_fit)
    trajectory = np.zeros_like(y_center_fit)
    trajectory_filt = np.zeros_like(y_center_fit)
    trajectory_filt2 = np.zeros_like(y_center_fit)
    for i in range(N_points):
        # for i in np.array([1, -1]):
        # if abs(y_center_fit[i, 0] - zero_pos_med) > 2 * zero_pos_med:
        #     continue
        trajectory_raw[i, :] = y_center_fit[i, :]
        trajectory[i, :] = y_center_fit[i, :] - zero_velocity
        roi = np.logical_not(np.isnan(trajectory_raw[i, :]))
        interpolant = interpolate.UnivariateSpline(
            positions_actual[roi],
            trajectory_raw[i, roi],
            k=3,
        )
        interpolant2 = interpolate.UnivariateSpline(
            positions_actual[roi],
            trajectory[i, roi],
            k=5,
        )
        trajectory_filt[i, :] = interpolant(positions_actual)
        trajectory_filt2[i, :] = interpolant2(positions_actual)
        # trajectory_filt[i, :] = ndimage.gaussian_filter1d(trajectory_raw[i], sigma=2)
        # trajectory_filt2[i, :] = ndimage.gaussian_filter1d(trajectory[i], sigma=2)
        ax[0].plot(
            1e6 * trajectory_filt2[i, :],
            positions_actual,
            marker="o",
            color=col[i],
            markeredgecolor=col[i],
            markerfacecolor=col_face[i],
            markersize=3,
            lw=1,
        )
        ax[1].plot(
            1e6 * trajectory_filt[i, :],
            positions_actual,
            marker="o",
            color=col[i],
            markeredgecolor=col[i],
            markerfacecolor=col_face[i],
            markersize=3,
            lw=1,
        )
    for a in ax:
        a.set_ylabel("Objective position in mm")
        a.set_xlim(-500, 500)
        a.set_ylim(positions_actual[0], positions_actual[-1])
    for a in ax:
        fig.colorbar(sm, ax=a, shrink=0.6, label=r"Mach number $\beta$")
    ax[1].set_title(r"Raw defect displacement")
    ax[1].set_xlabel(r"Defect displacement in $\mu m$")
    ax[0].set_title(r"Angle removed defect displacement")
    ax[0].set_xlabel(r"Defect displacement in $\mu m$")
    plt.show()


def fit_defect_traj(scan: str, plot=False) -> None:
    y = np.linspace(-Ny_def // 2 * d_def * 1e6, Ny_def // 2 * d_def * 1e6, Ny_def)
    kp = np.load(f"{scan}/kp.npy")
    ky = np.load(f"{scan}/ky.npy")
    cs = np.load(f"{scan}/cs_analytical.npy")
    betas = 2 * np.pi / (k0 * cs) * ky
    betas -= betas[0]
    positions_actual = np.load(f"{scan}/positions_actual_valid.npy")
    positions_actual -= np.min(positions_actual)
    positions_actual = np.flip(positions_actual)
    N_points = kp.shape[0]
    profile = np.load(f"{scan}/profile.npy")
    profile_fitted = np.load(f"{scan}/profile_fitted.npy")
    y_center_fit = np.load(f"{scan}/y_center_fit.npy")
    y_center_err = np.load(f"{scan}/y_center_err.npy")
    y_width_fit = np.load(f"{scan}/y_width_fit.npy")
    # remove bad points
    for i in range(y_center_fit.shape[0]):
        std = rolling_std(y_center_fit[i, :], 3)
        roi = std > 2 * np.median(std)
        y_center_fit[i, roi] = np.nan
    # fit 0 velocity component with a spline to remove 0 velocity trajectory
    # try:
    #     roi = np.logical_not(np.isnan(y_center_fit[0, :]))
    #     poly = np.polyfit(positions_actual[roi], y_center_fit[0, roi], deg=3)
    #     zero_velocity = np.polyval(poly, positions_actual)
    # except SystemError:
    #     print("Could not fit !")
    #     zero_velocity = y_center_fit[0, :]
    zero_velocity = y_center_fit[0, :]
    # plt.figure()
    # plt.plot(positions_actual, y_center_fit[0, :], label="Data", marker="o")
    # plt.plot(positions_actual, zero_velocity, label="Fit")
    # plt.xlabel("Objective position in mm")
    # plt.legend()
    # plt.show()

    # alpha = 287282667
    # alpha0 = 1e8
    # alpha = alpha0

    x00 = 3e-6
    x0 = x00

    def d_traj(z, alpha, v0):
        bet = 0.7
        R = (15 + 4) * 48e-6
        dn = 1e-6
        V = k0 * np.abs(dn)
        ki = 2 * np.pi / (795e-9)
        xisurd = 1 / R
        ro = 2 * np.sqrt(2) * xisurd * (bet - np.sqrt(2) * xisurd)
        x = -alpha * (R * V * ro / ki) * z**2 + v0 * z + x0
        return x

    fig, ax = plt.subplots(figsize=(3, 4), layout="constrained")
    col = plt.cm.viridis(np.linspace(0, 1, N_points))
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=colors.Normalize(vmin=np.min(betas), vmax=np.max(betas)),
    )
    col_face = [[col[i][0], col[i][1], col[i][2], 1] for i in range(N_points)]
    trajectory = np.zeros_like(y_center_fit)
    trajectory_filt = np.zeros_like(y_center_fit)
    # for i in range(N_points):
    for i in np.array([N_points - 1]):
        trajectory[i, :] = y_center_fit[i, :] - zero_velocity
        roi = np.logical_not(np.isnan(trajectory[i, :]))
        interpolant = interpolate.UnivariateSpline(
            positions_actual[roi], trajectory[i, roi]
        )
        trajectory_filt[i, :] = interpolant(positions_actual)
        traj_z = trajectory_filt[i, :]
        z_um = positions_actual * 1e-3
        print(z_um)

        n_v = 4
        step = len(z_um) // n_v
        for j in range(n_v):
            if j == 0:
                alpha = 1
                x0 = x00
            if j == 1:
                # alpha = alpha0 * 1
                x0 = 0.2e-6
            if j == 2:
                # alpha = alpha0 * 3
                x0 = -12e-6
            if j == 3:
                # alpha = alpha0 * 11
                x0 = -36e-6
            popt, _ = opt.curve_fit(
                d_traj,
                z_um[j * step : (j + 1) * step],
                traj_z[j * step : (j + 1) * step],
                p0=[1e8, 0.2],
            )
            traj_fit = d_traj(z_um[j * step : (j + 1) * step], *popt)
            print("alpha=", popt[0], "v0/cs=", popt[1] / cs, "x0=", x0 * 1e6, "um")
            velo0 = popt[1] / cs

            ax.plot(
                traj_z[j * step : (j + 1) * step] * 1e6,
                z_um[j * step : (j + 1) * step] * 1e3,
                marker="o",
                color="tab:blue",
                markeredgecolor="tab:blue",
                markerfacecolor="lightsteelblue",
                # markersize=3,
                # lw=1,
            )
            ax.plot(
                traj_fit[0:step] * 1e6,
                z_um[j * step : (j + 1) * step] * 1e3,
                "-",
                color="tab:red",
                label=f"v0/cs[{j}]={velo0}\n",
            )
    ax.set_ylabel("Objective position in mm")
    ax.set_title("Filtered defect displacement")
    ax.set_xlabel(r"Defect displacement in $\mu m$")
    # ax.legend()
    plt.show()


# %% Manual data processing

scan_beta = f"data/08301817_scan_k_defect"
scan_z = f"data/10181300_scan_k_defect_position"

# Fig1c and Fig2a-b
fish_plot(scan_beta, plot=True)

# Fig2c
# fish_crit_velo(scan_beta)
# vortex_emission(scan_beta)


# Fig2e
# tf_defect(scan_beta, plot=True)

# Fig1b and Fig2d
# plot_defect_traj(scan_z)
# fit_defect_traj(scan_z)


# Fig3
# inc_vs_comp_plot(scan_z, plot=True)
