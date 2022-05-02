import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.style as sty
import numpy as np


def get_res_by_key(res):
    """results originally in list (img_n) of list (interaction_n) of dicts
    with keys 'metric_name'
    that are transfomed to {'metric_name':array of shape
    (img_n, interaction_n)}
    """
    res_by_key = {}
    for key in res[0][0]:
        res_by_key[key] = np.empty_like(res)
        for img_n, _ in enumerate(res):
            for n_click, _ in enumerate(res[img_n]):
                res_by_key[key][img_n, n_click] = res[img_n][n_click][key]
    return res_by_key


def plot_model_res(model_res, model_names, dest_dir=".", prefix=""):
    for metric_name in model_res[model_names[0]]:
        plt.figure()
        plt.ylabel(metric_name)
        plt.xlabel("clicks")
        for ind, model_name in enumerate(model_names):
            plt.plot(
                model_res[model_name][metric_name].mean(0), label=model_name
            )
        plt.legend()
        plt.savefig(os.path.join(dest_dir, prefix + metric_name) + ".png")


def plot_grid_res(
    model_res,
    model_names,
    robot_prefixes,
    dest_dir=".",
    prefix="",
    font_size=15,
):
    sty.use("seaborn")
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"] + [(0, 0, 0)]

    plt.rc("font", size=font_size)  # controls default text sizes
    plt.rc("axes", titlesize=font_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=font_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=font_size * 2 // 3)  # legend fontsize
    plt.rc(
        "figure", titlesize=font_size * 3 // 2
    )  # fontsize of the figure title

    for metric_name in model_res[robot_prefixes[0] + model_names[0]]:
        plt.figure()
        plt.ylabel(metric_name)
        plt.xlabel("clicks")
        assert (
            len(model_names) <= 4
        ), "Not enough line styles for this many models"
        lines = ["-", "--", "-.", ":"][: len(model_names)]
        for ind, model_name in enumerate(model_names):
            for rind, robot_prefix in enumerate(robot_prefixes):
                y = model_res[robot_prefix + model_name][metric_name].mean(0)
                x = range(1, len(y) + 1)
                plt.plot(
                    x,
                    y,
                    lines[ind],
                    color=colors[rind],
                    label=model_name + " " + robot_prefix[:-1],
                    alpha=0.7,
                    linewidth=2,
                )
                plt.scatter(
                    x,
                    y,
                    marker="_",
                    color=colors[rind],
                    alpha=0.5,
                    linewidth=1,
                )
                plt.xticks(x)
        plt.legend(ncol=len(model_names))
        plt.savefig(
            os.path.join(dest_dir, prefix + metric_name) + ".png", dpi=600
        )


if __name__ == "__main__":
    res_dir = "/home/franchesoni/iis/iislib/results/tmp/"
    prefix = exp_prefix = "first_pseudo_battle_"
    # load config file and get variables
    with open(os.path.join(res_dir, exp_prefix + "config.pkl"), "rb") as f:
        config = pickle.load(f)
    # get variables from config dict
    model_names = config["model_names"]
    robot_prefixes = config["robot_prefixes"]

    model_res = {}
    for model_name in model_names:
        for robot_prefix in robot_prefixes:
            res = np.load(
                os.path.join(
                    res_dir, f"scores_{prefix}{robot_prefix}{model_name}.npy"
                ),
                allow_pickle=True,
            )
            model_res[robot_prefix + model_name] = get_res_by_key(res)

    plot_grid_res(
        model_res,
        model_names,
        robot_prefixes,
        dest_dir=res_dir,
        prefix=prefix,
    )
    # model_res, [r + n for r in robot_prefixes for n in model_names], dest_dir=res_dir, prefix=prefix
