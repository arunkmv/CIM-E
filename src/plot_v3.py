##############################################################################
# Copyright (C) 2026 Rebecca Pelke, Arunkumar Vaidyanathan                   #
# All Rights Reserved                                                        #
#                                                                            #
# This is work is licensed under the terms described in the LICENSE file     #
# found in the root directory of this source tree.                           #
##############################################################################
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
label_fontsize = 11
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


def energy_efficiency_plot(df: pd.DataFrame,
                           store_path: str,
                           s_cat: list,
                           d_cat: list,
                           energy_estimates: dict,
                           plt_legend: bool = True):

    bnn_mode_labels = [
        "BNN_I", "BNN_II", "BNN_III", "BNN_IV", "BNN_V", "BNN_VI"
    ]
    tnn_mode_labels = ["TNN_I", "TNN_II", "TNN_III", "TNN_IV", "TNN_V"]

    for nn_name in list(df['nn_name'].unique()):
        print(f"Generate plots for {nn_name}.")
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
        print(f"Layers: {layers}.")

        fig, axs = plt.subplots(1,
                                len(mm_sets),
                                figsize=(3.7 * len(mm_sets),
                                         3),
                                layout='tight')
        axs = np.atleast_1d(axs)
        sec_axs = [ax.twinx() for ax in axs]

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

                zo = mm_set.index(mm)
                axs[n].plot(xs_strs,
                            mpjs,
                            marker=marker_mode[mm],
                            label=f"{mm.replace('NN_', ' ')}",
                            color=color_mode[mm],
                            zorder=zo)
                sec_axs[n].plot(xs_strs,
                                [xu + zo*0.005 for xu in xus],
                                color=color_mode[mm],
                                alpha=0.7,
                                linestyle=':',
                                zorder=zo)

            axs[n].set_title(f"{nn_name} - {mm_set_name}")
            axs[n].set_ylabel("Energy Efficiency (MACs/J)",
                              fontsize=label_fontsize)
            axs[n].set_xlabel("Crossbar Sizes",
                              fontsize=label_fontsize)

            axs[n].grid(axis='y', linestyle=':', color=grid_color)

            y_min = min(ax.get_ylim()[0] for ax in axs)
            y_max = max(ax.get_ylim()[1] for ax in axs)
            padding = 0.05 * (y_max - y_min)
            y_max += padding
            for ax in axs:
                ax.set_ylim(y_min, y_max)

            axs[n].tick_params(axis='both', labelsize=tick_fontsize)
            sec_axs[n].set_ylabel("Crossbar Utilization",
                                  fontsize=label_fontsize)
            sec_axs[n].set_ylim(0.0, 1.0)
            sec_axs[n].tick_params(axis='y', labelsize=tick_fontsize)

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
                leg1.set_zorder(10)
                sec_axs[n].add_artist(leg1)
                leg2 = sec_axs[n].legend(handles=color_legend,
                                         loc='upper left',
                                         fontsize=legend_fontsize,
                                         ncol=3)
                leg2.set_zorder(10)

    fig.savefig(
        f"{store_path}/energy_efficiency_{nn_name}.pdf",
        dpi=300)
    fig.savefig(
        f"{store_path}/energy_efficiency_{nn_name}.png",
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

    if exp_name.startswith('mvm_profiling'):
        energy_estimates = json.load(
            open(f"{exp_result_path}/energy_estimates.json", 'r'))
        energy_efficiency_plot(df=df,
                               store_path=store_path,
                               s_cat=cat_static,
                               d_cat=cat_dynamic,
                               energy_estimates=energy_estimates)
    else:
        raise Exception(f"Plot for experiment {exp_name} not implemented.")
