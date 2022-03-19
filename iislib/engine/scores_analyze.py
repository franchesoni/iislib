import matplotlib.pyplot as plt
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


def plot_model_res(model_res, model_names):
    for metric_name in model_res[model_names[0]]:
        plt.figure()
        plt.ylabel(metric_name)
        plt.xlabel("clicks")
        for model_name in model_names:
            plt.plot(
                model_res[model_name][metric_name].mean(0), label=model_name
            )
        plt.legend()
        plt.savefig(metric_name + ".png")


if __name__ == "__main__":
    model_res = {}
    model_names = ["ours", "ritm", "gto99"]
    for model_name in model_names:
        res = np.load(f"scores_{model_name}.npy", allow_pickle=True)
        model_res[model_name] = get_res_by_key(res)

    plot_model_res(model_res, model_names)
