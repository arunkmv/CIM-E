##############################################################################
# Copyright (C) 2026 Rebecca Pelke, Arunkumar Vaidyanathan                   #
# All Rights Reserved                                                        #
#                                                                            #
# This is work is licensed under the terms described in the LICENSE file     #
# found in the root directory of this source tree.                           #
##############################################################################
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import rc, ticker
import pandas as pd
import numpy as np
import math
import os
import ast
import pickle
import argparse
import json
from functools import reduce

from model_parser import *
from run import *
from RWTHColors import ColorManager

# Colors
cm = ColorManager()

colors = [
    cm.RWTHBlau(),  # blue
    cm.RWTHOrange(),  # orange
    cm.RWTHMaiGruen(),  # light green
    cm.RWTHGruen(),  # green
    cm.RWTHBordeaux(),  # bordeaux
    cm.RWTHTuerkis(),  # turquoise
    cm.RWTHMagenta(),  # magenta
    cm.RWTHPetrol(),  # petrol
    cm.RWTHViolett(),  # violet
]

grid_color = cm.RWTHSchwarz(25)

color_mode = {
    'BNN_I': colors[0],
    'BNN_II': colors[1],
    'BNN_III': colors[2],
    'BNN_IV': colors[3],
    'BNN_V': colors[4],
    'BNN_VI': colors[5],
    'TNN_I': colors[0],
    'TNN_II': colors[1],
    'TNN_III': colors[2],
    'TNN_IV': colors[3],
    'TNN_V': colors[4]
}

marker_mode = {
    'BNN_I':   'o',
    'BNN_II':  's',
    'BNN_III': '^',
    'BNN_IV':  'D',
    'BNN_V':   'v',
    'BNN_VI':  'x',
    'TNN_I':   'o',
    'TNN_II':  's',
    'TNN_III': '^',
    'TNN_IV':  'D',
    'TNN_V':   'v',
}

color_bits = {
    3: colors[0],
    4: colors[1],
    5: colors[2],
    6: colors[3],
    8: colors[4]
}

# Font
rc('text', usetex=True)
rc('text.latex', preamble="\\usepackage{libertine}")
title_fontsize = 12
tick_fontsize = 8
label_fontsize = 10
legend_fontsize = 8

# title_fontsize = 16
# tick_fontsize = 12
# label_fontsize = 14
# legend_fontsize = 10

# Figure sizes
fig_width = 12
fig_height = 3

# Other helpers
times_str = r"$\times$"
nn_labels = {"VGG7": "VGG-7", "LeNet": "LeNet-5"}
bnn_mode_labels = [
    "BNN_I", "BNN_II", "BNN_III", "BNN_IV", "BNN_V", "BNN_VI"
]
tnn_mode_labels = ["TNN_I", "TNN_II", "TNN_III", "TNN_IV", "TNN_V"]


def energy_efficiency_plot(df: pd.DataFrame,
                           store_path: str,
                           s_cat: list,
                           d_cat: list,
                           energy_estimates: dict,
                           plt_legend: bool = True,
                           energy_lims: tuple | None = (0, 5e14)):

    for nn_name in list(df['nn_name'].unique()):
        print(f"Generate energy efficiency plots for {nn_name}.")
        df_nn = df[(df['nn_name'] == nn_name)]

        if 'num_runs' in d_cat:
            max_num_runs = max(df_nn['num_runs'].unique())
            df_nn = df_nn[(df_nn['num_runs'] == max_num_runs)]

        xbar_sizes = df_nn['xbar_size'].unique()

        # Count modes in experiment
        m_modes = list(df_nn['m_mode'].unique())
        bnn_modes = [bm for bm in bnn_mode_labels if bm in m_modes]
        tnn_modes = [tm for tm in tnn_mode_labels if tm in m_modes]
        mm_sets = {}
        if bnn_modes:
            mm_sets["BNN"] = bnn_modes
        if tnn_modes:
            mm_sets["TNN"] = tnn_modes

        layers = energy_estimates[str(
            df_nn.loc[:, "config_idx"].iloc[0])].keys()

        fig, axs = plt.subplots(1,
                                len(mm_sets),
                                figsize=(fig_width * len(mm_sets) / 4,
                                         3),
                                layout='tight',
                                sharey=True)
        axs = np.atleast_1d(axs)
        sec_axs = [ax.twinx() for ax in axs]
        for sax in sec_axs[1:]:
            sax.sharey(sec_axs[0])
        for sax in sec_axs[:-1]:
            sax.tick_params(labelright=False)

        for n, (mm_set_name, mm_set) in enumerate(mm_sets.items()):
            for mm in mm_set:
                xs_strs = []
                mpjs = []
                xus = []
                for xs in xbar_sizes:
                    xs_strs.append(xs[1:-1].replace(', ', times_str))
                    df_xs_mm = df_nn[
                        (df_nn['xbar_size'] == xs) &
                        (df_nn['m_mode'] == mm)]
                    c_idx = str(df_xs_mm.loc[:, "config_idx"].iloc[0])
                    mm_energy_estimates = energy_estimates[c_idx]
                    tot_energy = 0
                    tot_macs = 0
                    xbar_util = 0
                    tot_mvms = 0
                    for l_name, ee_stats in mm_energy_estimates.items():
                        tot_energy += ee_stats["tot_energy"]
                        tot_macs += ee_stats["num_macs"]
                        tot_mvms += ee_stats["num_mvms"]
                        xbar_util += ee_stats["num_mvms"] * \
                            ee_stats["col_util"] * ee_stats["row_util"]
                    mpjs.append(tot_macs / tot_energy)
                    xus.append(xbar_util / tot_mvms)

                zo = mm_set.index(mm) + 1
                axs[n].plot(xs_strs,
                            mpjs,
                            marker=marker_mode[mm],
                            color=color_mode[mm],
                            zorder=zo)
                sec_axs[n].plot(xs_strs,
                                [xu + zo*0.006 for xu in xus],
                                color=color_mode[mm],
                                alpha=0.7,
                                linestyle=':',
                                zorder=zo * 10)

            axs[n].set_title(f"{nn_name} - {mm_set_name}")
            axs[0].set_ylabel("Energy Efficiency (MACs/J)",
                              fontsize=label_fontsize)
            axs[n].set_xlabel("Crossbar Sizes",
                              fontsize=label_fontsize)

            axs[n].grid(axis='y', linestyle=':', color=grid_color, zorder=0)

            if energy_lims:
                axs[n].set_ylim(*energy_lims)
            else:
                y_min = min(ax.get_ylim()[0] for ax in axs)
                y_max = max(ax.get_ylim()[1] for ax in axs)
                padding = 0.05 * (y_max - y_min)
                y_max += padding
                for ax in axs:
                    ax.set_ylim(y_min, y_max)

            axs[n].tick_params(axis='both', labelsize=tick_fontsize)
            sec_axs[-1].set_ylabel("Crossbar Utilization",
                                   fontsize=label_fontsize)
            sec_axs[-1].set_ylim(0.0, 1.05)
            sec_axs[-1].tick_params(axis='y', labelsize=tick_fontsize)

            if plt_legend:
                # Create structured legend
                # Legend for XBar utilization
                dotted_line_legend = [mlines.Line2D([], [],
                                                    color='black',
                                                    linestyle=':',
                                                    label="Crossbar Util.")]
                # Legend for colors (Mapping modes)
                color_legend = [
                    mlines.Line2D([], [],
                                  color=c,
                                  marker=marker_mode[mm],
                                  linestyle='-',
                                  label=mm.replace('NN_', ' '))
                    for mm, c in color_mode.items() if mm in mm_set
                ]
                leg1 = sec_axs[n].legend(handles=dotted_line_legend,
                                         loc='lower right',
                                         fontsize=legend_fontsize,
                                         ncol=1)
                leg1.set_zorder(100)
                sec_axs[n].add_artist(leg1)
                leg2 = sec_axs[n].legend(handles=color_legend,
                                         loc='upper left',
                                         fontsize=legend_fontsize,
                                         ncol=3)
                leg2.set_zorder(100)

    fig.savefig(
        f"{store_path}/energy_efficiency_{nn_name}.pdf",
        dpi=300)
    fig.savefig(
        f"{store_path}/energy_efficiency_{nn_name}.png",
        dpi=300)


def per_layer_energy_plot(df: pd.DataFrame,
                          store_path: str,
                          s_cat: list,
                          d_cat: list,
                          energy_estimates: dict,
                          nn_name: str | None = None,
                          plt_legend: bool = True):
    bar_width = 0.5
    size_gap = 0.4
    layer_gap = 1
    components = {"DAC": ["crossbar.row_driver"],
                  "Memristor Array": ["crossbar.memristors"],
                  "ADC + Accumulate": ["crossbar.adcs", "crossbar.adders"]}
    color_component = {"DAC": colors[6],
                       "Memristor Array": colors[7],
                       "ADC + Accumulate": colors[8]}

    for nn_name in (list(df['nn_name'].unique()) if not nn_name else [nn_name]):
        print(f"Generate per-layer energy plots for {nn_name}.")
        df_nn = df[(df['nn_name'] == nn_name)]

        if 'num_runs' in d_cat:
            max_num_runs = max(df_nn['num_runs'].unique())
            df_nn = df_nn[(df_nn['num_runs'] == max_num_runs)]

        xbar_sizes = df_nn['xbar_size'].unique()
        xs_strs = {xs: xs[1:-1].replace(', ', times_str) for xs in xbar_sizes}

        # Count modes in experiment
        m_modes = list(df_nn['m_mode'].unique())
        bnn_modes = [bm for bm in bnn_mode_labels if bm in m_modes]
        tnn_modes = [tm for tm in tnn_mode_labels if tm in m_modes]
        mm_sets = {'BNN': bnn_modes, 'TNN': tnn_modes}

        layers = energy_estimates[str(
            df_nn.loc[:, "config_idx"].iloc[0])].keys()
        print(f"Layers: {layers}.")

        for mm_set_name, mm_set in mm_sets.items():
            if len(mm_set) > 0:
                fig, (ax, sec_ax) = plt.subplots(
                    2, 1, figsize=(fig_width, fig_height * 0.9), sharex=True,
                    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.0},
                    layout='tight'
                )
                pos_offset = 0.0
                layer_ticks: list[tuple] = []
                for x, l_name in enumerate(layers):
                    l_start = pos_offset
                    for y, xs in enumerate(xbar_sizes):
                        size_start = pos_offset
                        for z, mm in enumerate(mm_set):
                            df_xs_mm = df_nn[
                                (df_nn['xbar_size'] == xs) &
                                (df_nn['m_mode'] == mm)]
                            c_idx = str(df_xs_mm.loc[:, "config_idx"].iloc[0])
                            energy = energy_estimates[c_idx][l_name]["tot_energy"]
                            ax.bar(
                                pos_offset,
                                energy,
                                width=bar_width,
                                color=color_mode[mm],
                                edgecolor="white",
                                linewidth=0.3,
                                zorder=10
                            )

                            bottom = 0.0
                            for c_name, comps in components.items():
                                norm_comp_energy = sum(
                                    [energy_estimates[c_idx][l_name][c] for c in comps]) / energy
                                sec_ax.bar(
                                    pos_offset,
                                    norm_comp_energy,
                                    bottom=bottom,
                                    width=bar_width,
                                    color=color_component[c_name],
                                    edgecolor="white",
                                    linewidth=0.3)
                                bottom += norm_comp_energy

                            pos_offset += bar_width
                        sec_ax.text(
                            (size_start + pos_offset - bar_width) / 2,
                            -0.1,
                            xs_strs[xs],
                            transform=sec_ax.get_xaxis_transform(),
                            ha="center",
                            va="top",
                            fontsize=legend_fontsize,
                            rotation=0,
                            color="dimgray")
                        pos_offset += size_gap
                    pos_offset -= size_gap
                    layer_ticks.append(
                        ((l_start + pos_offset - bar_width) / 2, l_name))
                    if x != len(layers) - 1:
                        ax.axvline((pos_offset + (layer_gap - bar_width) / 2),
                                   color="grey", linewidth=0.6, linestyle=":")
                        sec_ax.axvline((pos_offset + (layer_gap - bar_width) / 2),
                                       color="grey", linewidth=0.6, linestyle=":")
                    pos_offset += layer_gap

                ax.grid(axis='y', linestyle=':', color=grid_color)
                ax.set_ylabel("Energy\nConsumption (J)",
                              fontsize=label_fontsize, wrap=True)
                ax.set_yscale("log")
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
                ax.spines["bottom"].set_linewidth(1.1)

                sec_ax.set_ylabel("Norm.\nComponent\nEnergy",
                                  fontsize=label_fontsize, wrap=True)
                sec_ax.set_xticks([c for c, _ in layer_ticks])
                sec_ax.set_xticklabels(
                    [lbl for _, lbl in layer_ticks], fontsize=label_fontsize)
                sec_ax.tick_params(axis='x', pad=12, top=False)
                sec_ax.set_ylim(1.05, 0)
                sec_ax.spines["top"].set_linewidth(1.1)

                ax.set_title(f"Per Layer Energy - {nn_name} - {mm_set_name}")

            if plt_legend:
                # Create structured legend
                # Legend for components
                comp_color_legend = [mpatches.Patch(
                    facecolor=color_component[c],
                    label=c)
                    for c in components.keys()
                ]
                # Legend for colors (Mapping modes)
                mode_color_legend = [
                    mpatches.Patch(
                        facecolor=c,
                        label=mm.replace('NN_', ' '))
                    for mm, c in color_mode.items() if mm in mm_set
                ]
                leg1 = ax.legend(handles=mode_color_legend,
                                 loc='upper right',
                                 fontsize=legend_fontsize,
                                 ncol=1,)
                leg1.set_zorder(10)
                ax.add_artist(leg1)
                leg2 = ax.legend(handles=comp_color_legend,
                                 loc='upper right',
                                 fontsize=legend_fontsize,
                                 ncol=1,
                                 bbox_to_anchor=(0.93, 1.0)
                                 )
                leg2.set_zorder(10)

            fig.savefig(
                f"{store_path}/per_layer_energy_{nn_name}_{mm_set_name}.pdf",
                dpi=300)
            fig.savefig(
                f"{store_path}/per_layer_energy_{nn_name}_{mm_set_name}.png",
                dpi=300)


def raella_energy_comparison_plot(df: pd.DataFrame,
                                  store_path: str,
                                  energy_estimates: dict):

    print(f"Generate RAELLA comparison plot.")
    xbar_size = "[512, 512]"
    raella_512_resnet_values = {
        "S":  {"ADC": 61.36, "CimUnit": 17.21, "RowDrivers": 1.94, "ShiftAdd": 3.93},
        "WS": {"ADC": 96.76, "CimUnit": 6.88, "RowDrivers": 1.94, "ShiftAdd": 9.35},
    }
    hatch_mode = {'BNN': '////', 'TNN': '\\\\\\\\'}

    def get_raella_efficiency(per_comp_values: dict) -> float:
        """Convert per-component fJ/MAC to total MACs/J."""
        return 1 / sum(per_comp_values.values()) * 1e15

    def get_mapping_efficiency(per_layer_values: dict) -> float:
        """Convert per-layer energy to total MACs/J."""
        tot_macs = 0.0
        tot_energy = 0.0
        for _, l_est in per_layer_values.items():
            tot_macs += l_est["num_macs"]
            tot_energy += l_est["tot_energy"]
        return tot_macs / tot_energy

    # Count modes in experiment
    df_nn_xs = df[(df['nn_name'] == "ResNetE18") &
                  (df['xbar_size'] == xbar_size)]
    m_modes = list(df_nn_xs['m_mode'].unique())
    bnn_modes = [bm for bm in bnn_mode_labels if bm in m_modes]
    tnn_modes = [tm for tm in tnn_mode_labels if tm in m_modes]
    mm_sets = {'BNN': bnn_modes, 'TNN': tnn_modes}

    bar_params: list[tuple] = []
    baseline = get_raella_efficiency(raella_512_resnet_values["WS"])
    pos = iter(np.arange(len(bnn_modes) + len(tnn_modes) + 2))

    bar_params.append({"x_pos": next(pos),
                       "value": 1,
                       "color": colors[7],
                       "label": "RAELLA\nWithout\nSpec.",
                       "hatch": ''
                       })

    bar_params.append({"x_pos": next(pos),
                       "value": get_raella_efficiency(raella_512_resnet_values["S"]) / baseline,
                       "color": colors[8],
                       "label": "RAELLA\nWith\nSpec.",
                       "hatch": ''
                       })

    for mm_set_name, mm_set in mm_sets.items():
        for mm in mm_set:
            df_mm = df_nn_xs[(df_nn_xs['m_mode'] == mm)]
            c_idx = str(df_mm.loc[:, "config_idx"].iloc[0])
            bar_params.append({"x_pos": next(pos),
                               "value": get_mapping_efficiency(energy_estimates[c_idx]) / baseline,
                               "color": color_mode[mm],
                               "label": mm.replace('NN_', ' '),
                               "hatch": hatch_mode[mm_set_name]
                               })

    fig, ax = plt.subplots(figsize=(fig_width * 0.7, fig_height * 0.6),
                           layout='tight')
    bars = ax.bar(
        [d['label'] for d in bar_params],
        [d['value'] for d in bar_params],
        width = 0.5,
        color=[d['color'] for d in bar_params],
        edgecolor="white",
        linewidth=0.3,
        hatch=[d['hatch'] for d in bar_params],
    )

    for bp in bar_params:
        ax.text(bp['x_pos'], bp['value'] + 0.5, f"{bp['value']:.1f}x", ha="center", va="bottom", fontsize=legend_fontsize)

    ax.set_title("Energy Efficiency Comparison (norm. to RAELLA Without Speculation)")
    ax.set_ylabel("Norm. Energy\nEfficiency", fontsize=label_fontsize)
    ax.tick_params(axis='both', which='both', size=label_fontsize)
    ax.grid(axis='y', linestyle=':', color=grid_color)
    ax.set_ylim(0, 55)
    ax.set_yticks(range(0, 50, 20))

    fig.savefig(
        f"{store_path}/raella_energy_comparison.pdf",
        dpi=300)
    fig.savefig(
        f"{store_path}/raella_energy_comparison.pdf",
        dpi=300)


def get_exp_products(config: str):
    exp_name = config.split('/')[-1].split('.json')[0]
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    exp_result_path = repo_path + '/results/' + exp_name
    df = pd.read_csv(f"{exp_result_path}/{exp_name}.csv")
    return exp_name, repo_path, exp_result_path, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        help='Path to experiment config',
                        required=True)

    args = parser.parse_args()

    with open(args.config, 'r') as json_file:
        cfg = json.load(json_file)

    exp_name, repo_path, exp_result_path, df = get_exp_products(args.config)

    categories = df.columns
    cat_static = []  # Categories (columns) that all experiments have in common
    cat_dynamic = []  # Categories that change for at least one experiment

    for c in categories:
        if len(set(df[c])) > 1:
            if type(df[c].iloc[0]) in [float, np.float64, np.float32]:
                if all(math.isnan(x) for x in df[c]):
                    cat_static.append(c)
                    continue
            cat_dynamic.append(c)
        else:
            cat_static.append(c)

    print(
        f"The benchmark has the following (static) properties:\n{cat_static}")
    print(f"The benchmarks varies the following properties:\n{cat_dynamic}")

    store_path = f"{exp_result_path}"

    if exp_name.startswith('energy_estimation'):
        energy_estimates = json.load(
            open(f"{exp_result_path}/energy_estimates.json", 'r'))
        # energy_efficiency_plot(df=df,
        #                        store_path=store_path,
        #                        s_cat=cat_static,
        #                        d_cat=cat_dynamic,
        #                        energy_estimates=energy_estimates,
        #                        plt_legend=(cfg['nn_names'][0] == 'LeNet5'))
        # per_layer_energy_plot(df=df,
        #                       store_path=store_path,
        #                       s_cat=cat_static,
        #                       d_cat=cat_dynamic,
        #                       energy_estimates=energy_estimates)
        if cfg['nn_names'][0] == 'ResNetE18':
            raella_energy_comparison_plot(df=df,
                                          store_path=store_path,
                                          energy_estimates=energy_estimates)
    else:
        raise Exception(f"Plot for experiment {exp_name} not implemented.")
