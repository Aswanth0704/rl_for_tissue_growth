"""Matplotlib figures mirroring Figures 6-8 and 11-14 of the paper."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed categorical assignment (one hue per growth law, never cycled)
COLORS = {"Dynamic programming": "#2a78d6", "Deep Q network": "#eb6834",
          "Taber & Humphrey (2001)": "#1baf7a", "Liu et al. (2019)": "#eda100"}
TEXT = "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": TEXT, "axes.labelcolor": "#0b0b0b",
                     "xtick.color": TEXT, "ytick.color": TEXT, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.6, "legend.frameon": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})


def _style(ax, x, y, title=None):
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title, loc="left", fontsize=9, color="#0b0b0b")


def plot_example1(results, data, s_h, path):
    """results: {name: history dict}; data: bersi2017 module."""
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.5))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        _ = ax
    first = next(iter(results.values()))
    # (a) pressure
    ax = axes[0]
    ax.plot(first["t"], first["p_mmHg"], color=TEXT, lw=1.2)
    ax.plot(data.days, data.systolic_pressure_mmHg, "o", mfc="white", mec="#0b0b0b", ms=5, label="Experiment")
    _style(ax, "Time (days)", "Systolic pressure (mmHg)", "(a) pressure")
    panels = [("H_unloaded", "Unloaded wall thickness (mm)", data.unloaded_thickness_mm, "(b) unloaded thickness"),
              ("r_i", "Systolic inner radius (mm)", data.systolic_inner_radius_mm, "(c) inner radius"),
              ("lam_ez", "In vivo axial stretch", data.in_vivo_axial_stretch, "(d) in vivo axial stretch"),
              ("lam_gr", r"$\lambda_{g,r}$", None, "(e) radial growth"),
              ("lam_gt", r"$\lambda_{g,\theta}$", None, "(f) circumferential growth"),
              ("lam_gz", r"$\lambda_{g,z}$", None, "(g) axial growth"),
              ("sig_tt", r"$\sigma_{\theta\theta}$ (kPa)", data.sigma_tt_kPa, "(h) circumferential stress"),
              ("sig_zz", r"$\sigma_{zz}$ (kPa)", data.sigma_zz_kPa, "(i) axial stress"),
              ("tau_w", r"$\tau_w$ (dyn/cm$^2$)", None, "(j) wall shear stress")]
    for k, (key, ylab, exp, title) in enumerate(panels):
        ax = axes[k + 1]
        for name, h in results.items():
            ax.plot(h["t"], h[key], color=COLORS.get(name, "#555"), label=name)
        if exp is not None:
            ax.plot(data.days, exp, "o", mfc="white", mec="#0b0b0b", ms=5, label="Experiment", zorder=5)
        if key in ("sig_tt", "sig_zz", "tau_w"):
            ax.axhline({"sig_tt": s_h[0], "sig_zz": s_h[1], "tau_w": s_h[2]}[key], color="#0b0b0b", ls="--", lw=0.9, label="Homeostatic target")
        _style(ax, "Time (days)", ylab, title)
    axes[1].legend(fontsize=7, loc="lower right")
    axes[10].axis("off")
    axes[11].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_example2_stress(loaded, path, title_prefix="Loaded configuration"):
    """loaded: {name: dict(r (2,5), sig_tt (2,5), sig_zz (2,5))}; first entry is 'Before growth'."""
    n = len(loaded)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, d) in zip(axes, loaded.items()):
        col = COLORS.get(name, "#0b0b0b")
        for k, lab in enumerate(("media", "adventitia")):
            ax.plot(d["r"][k], d["sig_tt"][k], "-", color=col, label=r"$\sigma_{\theta\theta}$" if k == 0 else None)
            ax.plot(d["r"][k], d["sig_zz"][k], "--", color=col, label=r"$\sigma_{zz}$" if k == 0 else None)
        _style(ax, "r (mm)", r"$\sigma$ (kPa)", name)
    axes[0].legend(fontsize=7)
    fig.suptitle(title_prefix, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_example2_growth(histories, loc, path):
    """histories: {name: dict(t, lam_g (T, 2, 5, 3))}."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    comp = [r"\lambda_{g,r}", r"\lambda_{g,\theta}", r"\lambda_{g,z}"]
    for k, lay in enumerate(("M", "A")):
        for j in range(3):
            ax = axes[k, j]
            for name, h in histories.items():
                ax.plot(h["t"], h["lam_g"][:, k, loc, j], color=COLORS.get(name, "#555"), label=name)
            _style(ax, "Time (days)", rf"${comp[j]}^{lay}$", f"Loc {loc + 1}")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_training(hist, path, title="DQN training"):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    ep = hist["episode"]
    w = max(1, len(ep) // 100)
    def smooth(x):
        return np.convolve(x, np.ones(w) / w, mode="valid")
    axes[0].plot(ep[w - 1:], smooth(hist["return"]), color=COLORS["Deep Q network"])
    _style(axes[0], "Episode", "Return (moving avg.)", title)
    axes[1].plot(ep[w - 1:], smooth(hist["final_distance"]), color=COLORS["Deep Q network"])
    _style(axes[1], "Episode", "Final distance to homeostasis")
    axes[2].plot(ep[w - 1:], smooth(hist["length"]), color=COLORS["Deep Q network"])
    _style(axes[2], "Episode", "Episode length")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
