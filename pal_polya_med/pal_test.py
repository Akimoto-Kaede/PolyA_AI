import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

tf.debugging.enable_check_numerics()
print(tf.config.list_physical_devices("GPU"))

output_dir = "/scem/work/zhangyt/project/ai/pal_polya_med"


def custom_metric(y_true, y_pred):
    y_true_mean = keras.backend.mean(y_true, axis=0)
    y_pred_mean = keras.backend.mean(y_pred, axis=0)
    r_numerator = keras.backend.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
    x_square_sum = keras.backend.sum((y_true - y_true_mean) * (y_true - y_true_mean))
    y_square_sum = keras.backend.sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))
    r_denominator = keras.backend.sqrt(x_square_sum * y_square_sum)
    r = r_numerator / r_denominator
    metric_out = -(keras.backend.mean(r) ** 2)
    return metric_out


def predict(path_to_model, x_ary):
    model = keras.models.load_model(
        path_to_model, custom_objects={"custom_metric": custom_metric}
    )
    y_pred = model.predict(x_ary).ravel()
    return y_pred


def scatter_plot(x, y, fn):
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)(values)
    spearman_corr, _ = stats.spearmanr(x, y)
    pearson_corr, _ = stats.pearsonr(x, y)
    n = len(x)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=kernel, cmap="RdYlBu_r", s=20)
    plt.xlabel("Measured data")
    plt.ylabel("Predicted data")
    plt.colorbar(scatter, label="Density")
    plt.axis("equal")
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="grey", linewidth=1)
    plt.text(
        0.95,
        0.05,
        f"n={n}\nPearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.savefig(fn, dpi=300, bbox_inches="tight")


x_ary = np.load("/scem/work/zhangyt/project/ai/polya_med/data/x_input.npy")[20000:]
y_df = (
    pd.read_csv("/scem/work/zhangyt/project/ai/polya_med/data/y_input.csv")
    .iloc[20000:]
    .reset_index(drop=True)
)
temp = pd.DataFrame(None)
for i in range(10):
    y_pred = predict(output_dir + f"/models/median_best_model_CV_group_{i+1}.h5", x_ary)
    temp[f"group_{i+1}"] = y_pred
y_df["y_pred"] = temp.mean(axis=1).to_numpy()
y_df.to_csv(output_dir + "/prediction/pal_test.csv", index=False)
test = pd.read_csv("/scem/work/zhangyt/project/ai/polya_med/prediction/pal_test.csv")
scatter_plot(test["y"], test["y_pred"], output_dir + "/scatter_plot/pal_test.png")
