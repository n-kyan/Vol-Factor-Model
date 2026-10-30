"""Shared chart style and chart helpers for the notebooks (main.py and the *_nb.py companions)."""

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"     # low-VIX regime / main series
ORANGE = "#eb6834"   # high-VIX regime
GRAY = "#8a8985"     # reference lines, secondary series
INK = "#52514e"      # axis text


def new_figure(n_panels=1, width=6.0, height=3.6):
    """A figure with n side-by-side panels and quiet axes. Returns (fig, axes)."""
    fig, axes = plt.subplots(1, n_panels, figsize=(width, height), squeeze=False)
    axes = axes[0]
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRAY)
        ax.spines["bottom"].set_color(GRAY)
        ax.tick_params(colors=INK)
        ax.grid(alpha=0.2)
    return fig, axes


def normal_pdf(x, mean, sd):
    """Height of the normal (bell) curve with this mean and SD at each x."""
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def compare_to_normal(x, label, color):
    """Histogram of x against the normal curve with the same mean and SD.

    Left panel: ordinary scale, shows the middle. Right panel: log scale, shows the tails.
    """
    fig, axes = new_figure(n_panels=2, width=11, height=3.8)
    # Why plot in SD units: both charts then share one x-axis meaning ("how many SDs out")
    z = (x - x.mean()) / x.std()
    bins = np.linspace(z.min(), z.max(), 120)
    grid = np.linspace(z.min(), z.max(), 600)
    for ax in axes:
        ax.hist(z, bins=bins, density=True, color=color, alpha=0.55, label="actual days")
        ax.plot(grid, normal_pdf(grid, 0, 1), color="black", linewidth=1.5, label="normal curve, same mean and SD")
        ax.set_xlabel(f"{label}, in SDs from the mean")
    axes[0].set_xlim(-5, 5)
    axes[0].set_title("The middle (ordinary scale, clipped at ±5 SD)", fontsize=10)
    axes[0].set_ylabel("share of days (density)")
    axes[0].legend(frameon=False, fontsize=8)
    # Why a log scale: tail days are too rare to see on an ordinary scale; the log scale
    # shows how far the actual tails sit above the normal curve.
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-6, 1)
    axes[1].set_title("The tails (log scale, full range)", fontsize=10)
    fig.suptitle(label, fontsize=11)
    fig.tight_layout()
    return fig


def vix_timeline(vix, threshold):
    """VIX close over time, shaded orange where it sits at or above the regime threshold."""
    fig, axes = new_figure(width=11, height=3.6)
    ax = axes[0]
    ax.plot(vix.index, vix.values, color=GRAY, linewidth=0.7)
    ax.fill_between(vix.index, threshold, vix.values, where=vix.values >= threshold, color=ORANGE, alpha=0.35, linewidth=0)
    ax.axhline(threshold, color=INK, linewidth=1, linestyle="--")
    ax.text(vix.index[30], threshold - 5, f"median {threshold:.2f}", color=INK, fontsize=8)
    for year, label in [("2008", "2008 financial crisis"), ("2020", "2020 COVID crash")]:
        ax.annotate(label, (vix.loc[year].idxmax(), vix.loc[year].max()),
                    xytext=(-110, -5), textcoords="offset points", fontsize=8, color=INK)
    ax.set_ylabel("VIX close (points)")
    ax.set_title("VIX since 1990: orange = above the median (high-VIX regime)", fontsize=10)
    fig.tight_layout()
    return fig


def gamma_intervals(table, names):
    """Two panels per industry: gamma by regime with 95% intervals, and the high-minus-low difference.

    table: output of estimation.regime_interval_table. names: display label per row of table.
    """
    t = table.assign(label=names).sort_values("gamma high")
    y = np.arange(len(t))
    fig, axes = new_figure(n_panels=2, width=12, height=11)

    ax = axes[0]
    for stat, color, shift, label in [("gamma low", BLUE, -0.15, "low-VIX days"), ("gamma high", ORANGE, 0.15, "high-VIX days")]:
        ax.errorbar(t[stat], y + shift, xerr=[t[stat] - t[f"{stat} lo"], t[f"{stat} hi"] - t[stat]],
                    fmt="o", color=color, markersize=4, linewidth=1, label=label)
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(t["label"], fontsize=7)
    ax.set_xlabel("γ: % return per 1-point vol shock, with 95% interval")
    ax.set_title("Vol-shock sensitivity by regime", fontsize=10)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    ax.errorbar(t["high - low"], y, xerr=[t["high - low"] - t["high - low lo"], t["high - low hi"] - t["high - low"]],
                fmt="o", color=GRAY, markersize=4, linewidth=1)
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("γ high − γ low, with 95% interval")
    ax.set_title("Regime difference", fontsize=10)
    fig.tight_layout()
    return fig


def h2_scatter(table):
    """|gamma low| vs |gamma high| per industry; orange = |high| - |low| interval excludes 0."""
    abs_low = table["gamma low"].abs()
    abs_high = table["gamma high"].abs()
    sig = table["|high| - |low| sig"]
    fig, axes = new_figure(width=6.5, height=6)
    ax = axes[0]
    lim = np.array([0.0, max(abs_low.max(), abs_high.max()) * 1.08])
    ax.plot(lim, lim, color=GRAY, linewidth=1, label="equal in both regimes")
    ax.scatter(abs_low[~sig], abs_high[~sig], color=BLUE, s=22, zorder=3, label="difference not significant")
    ax.scatter(abs_low[sig], abs_high[sig], color=ORANGE, s=30, zorder=3, label="|high| − |low| interval excludes 0")
    for name in table.index[sig]:
        ax.annotate(name, (abs_low[name], abs_high[name]), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("|γ| on low-VIX days, % per point")
    ax.set_ylabel("|γ| on high-VIX days, % per point")
    ax.set_title("Dots above the line = bigger sensitivity at high VIX", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def altair_scatter(df, x, y, x_title, y_title, color_field, color_domain, color_range, tooltip, diagonal=False, label_field=None):
    """Interactive scatter (Altair): one point per row of df, hover tooltip, drag a box to select.

    Wrap it with mo.ui.altair_chart in a notebook to make the selection usable.
    color_field: column whose values (color_domain) map to color_range.
    diagonal: add a y = x reference line from 0 (for "bigger in one regime than the other").
    label_field: column holding a text label for points to name on the chart (empty string = no label).
    Why direct labels: the chart still reads without hovering, e.g. in an export or a screenshot.
    """
    import altair as alt
    import pandas as pd

    points = alt.Chart(df).mark_circle(size=70, opacity=0.9).encode(
        x=alt.X(x, title=x_title),
        y=alt.Y(y, title=y_title),
        color=alt.Color(color_field, scale=alt.Scale(domain=color_domain, range=color_range), legend=alt.Legend(title=None, orient="bottom")),
        tooltip=tooltip,
    )
    layers = [points]
    if label_field is not None:
        labels = alt.Chart(df).mark_text(align="left", dx=7, fontSize=11, color=INK).encode(x=x, y=y, text=label_field)
        layers.append(labels)
    if diagonal:
        top = max(df[x].max(), df[y].max()) * 1.08
        line = alt.Chart(pd.DataFrame({x: [0, top], y: [0, top]})).mark_line(color=GRAY, strokeDash=[4, 4]).encode(x=x, y=y)
        layers.insert(0, line)
    if len(layers) == 1:
        return points.properties(width=520, height=420)
    return alt.layer(*layers).properties(width=520, height=420)
