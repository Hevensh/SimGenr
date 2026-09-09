"""Small scientific overview of two validated seasonal realizations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("winter", type=Path)
    parser.add_argument("summer", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/physics_overview.png"))
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs/.mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worlds = [args.winter, args.summer]
    weather = read_npz(args.winter / "data/stage_05_weather/daily_weather.npz")
    channels = {str(name): i for i, name in enumerate(weather["channel_names"])}
    daily_mean = weather["dynamic"].mean(axis=(2, 3))
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle("SimGenr physics_v3 | synthetic scenarios, uncalibrated regional parameters", fontsize=15)
    axes[0, 0].plot(weather["timestamps"], daily_mean[:, channels["temperature"]], color="#d96d39", lw=1.2)
    axes[0, 0].set(title="Annual weather: temperature", xlabel="Day (zero based)", ylabel="Area mean temperature (°C)")
    axes[0, 1].bar(weather["timestamps"], daily_mean[:, channels["precipitation"]], color="#4389b4", width=1)
    axes[0, 1].set(title="Annual weather: precipitation", xlabel="Day (zero based)", ylabel="Area mean precipitation (mm/day)")
    for col, world in enumerate(worlds):
        source = read_npz(world / "data/stage_11_operation/source_load_forecast.npz")
        kinds = source["bus_kinds"]
        x = np.arange(source["timestamps"].size) / 24
        ax = axes[1, col]
        ax.plot(x, source["p_load_mw"].sum(axis=1), color="#292c37", lw=1.5, label="Gross demand")
        for kind, color, label in [("wind_bus", "#29926b", "Available wind"), ("pv_bus", "#e4af24", "Available PV")]:
            ax.plot(x, source["p_gen_available_mw"][:, kinds == kind].sum(axis=1), color=color, lw=1.2, label=label)
        ax.set(title=f"{'Winter' if col == 0 else 'Summer'} week: exogenous source/load", xlabel="Elapsed solar days", ylabel="Power (MW)")
        ax.legend(fontsize=8, loc="upper right")
    storage = read_npz(args.winter / "data/stage_14_storage_dispatch/storage_dispatch.npz")
    capacities = storage["site_energy_capacity_mwh"]
    if capacities.size:
        fractions = storage["soc_mwh"] / np.maximum(capacities, 1e-9)
        for index in range(capacities.size):
            axes[2, 0].plot(np.arange(fractions.shape[0]) / 24, fractions[:, index], lw=1, label=f"Site {index}")
        axes[2, 0].legend(fontsize=8)
    else:
        axes[2, 0].text(.5, .5, "No storage selected in this scenario", transform=axes[2, 0].transAxes, ha="center")
    axes[2, 0].set(title="Winter storage state of charge", xlabel="Elapsed solar days", ylabel="SOC / nameplate energy", ylim=(0, 1))
    report = json.loads((args.winter / "physics_validation.json").read_text(encoding="utf-8"))
    names = ["daily_hourly_precipitation", "daily_hourly_irradiance", "kcl_incidence", "soc_energy_conservation"]
    by_name = {row["name"]: row for row in report["checks"]}
    ratios = [max(by_name[name]["max_error"] / by_name[name]["tolerance"], 1e-8) for name in names]
    axes[2, 1].barh(["Rain sum", "GHI mean", "Nodal balance", "SOC energy"], ratios, color="#5087a3")
    axes[2, 1].axvline(1, color="#c84e3d", ls="--", label="Tolerance")
    axes[2, 1].set(xscale="log", xlim=(1e-8, 2), title="Winter numerical residuals (lower is better)", xlabel="Maximum error / allowed numerical tolerance")
    axes[2, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
