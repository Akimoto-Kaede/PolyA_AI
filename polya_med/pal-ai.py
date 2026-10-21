import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
import pandas as pd
from keras import layers
from sklearn.model_selection import KFold
from scipy import stats
import matplotlib.pyplot as plt

tf.debugging.enable_check_numerics()
print(tf.config.list_physical_devices("GPU"))

output_dir = "/scem/work/zhangyt/project/AI/polya_med"


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


def pal_model(
    filters: int = 64,
    kernel_size: int = 5,
    dropout_rate: float = 0.1,
    learning_rate: float = 0.001,
    l1: float = 0.01,
    l2: float = 0.000001,
) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(1056, 4)),
            layers.Conv1D(filters=filters, kernel_size=kernel_size),
            layers.LayerNormalization(),
            layers.Activation(keras.activations.selu),
            layers.Conv1D(filters=filters, kernel_size=kernel_size),
            layers.Dropout(rate=dropout_rate),
            layers.MaxPooling1D(pool_size=2),
            layers.LayerNormalization(),
            layers.Activation(keras.activations.selu),
            layers.GRU(
                units=96,
                go_backwards=False,
                kernel_regularizer=keras.regularizers.l1_l2(l1, l2),
            ),
            layers.BatchNormalization(),
            layers.Activation(keras.activations.selu),
            layers.Dense(
                units=96,
                kernel_regularizer=keras.regularizers.l1_l2(l1, l2),
            ),
            layers.Dropout(rate=dropout_rate),
            layers.BatchNormalization(),
            layers.Activation(keras.activations.selu),
            layers.Dense(units=1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae", custom_metric],
    )
    return model


def train_model(x_train, y_train, x_test, y_test, session_id: str = "", **kwargs):
    model = pal_model(**kwargs)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=output_dir + "/models/" + session_id + "_callback_best_model.h5",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10),
    ]
    out = model.fit(
        x=x_train,
        y=y_train,
        batch_size=50,
        epochs=30,
        verbose=1,
        shuffle=True,
        callbacks=callbacks,
        validation_data=(x_test, y_test),
    )
    model = keras.models.load_model(
        output_dir + "/models/" + session_id + "_callback_best_model.h5",
        custom_objects={"custom_metric": custom_metric},
    )
    return out, model


def loss_plot(train, test, fn):
    plt.clf()
    plt.plot(train, label="train")
    plt.plot(test, label="test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss or metric")
    plt.savefig(fn)


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


x_ary = np.load("/scem/work/zhangyt/project/AI/polya_med/data/x_input.npy")[:20000]
y_df = pd.read_csv("/scem/work/zhangyt/project/AI/polya_med/data/y_input.csv").head(
    20000
)
stat = pd.DataFrame(
    None, columns=["group", "rep", "r2", "loss", "mae", "custom_metric"]
)
res = pd.DataFrame(None, columns=["group", "id", "y", "y_pred"])
kf = KFold(n_splits=10, shuffle=True, random_state=57)
for i, (train_idx, test_idx) in enumerate(kf.split(x_ary, y_df["y"])):
    best_r2 = 0
    for j in range(5):
        x_train = x_ary[train_idx]
        x_test = x_ary[test_idx]
        y_train = y_df.loc[train_idx, "y"].to_numpy()
        y_test = y_df.loc[test_idx, "y"].to_numpy()
        history, model = train_model(
            x_train, y_train, x_test, y_test, session_id="median"
        )
        model.save(
            output_dir
            + "/models/"
            + "median"
            + "_best_model_CV_group_"
            + str(i + 1)
            + "_rep_"
            + str(j + 1)
            + ".h5"
        )
        scores = model.evaluate(x_test, y_test, batch_size=50, verbose=1)
        y_pred = model.predict(x_test, verbose=1).ravel()
        r_value = stats.linregress(y_test, y_pred).rvalue
        if r_value**2 > best_r2:
            best_r2 = r_value**2
            best_model = model
        del model
        keras.backend.clear_session()
        stat.loc[len(stat)] = [i + 1, j + 1, round(r_value**2, 3)] + list(scores)
        loss_plot(
            train=history.history["loss"],
            test=history.history["val_loss"],
            fn=output_dir
            + "/loss_plot/"
            + "median"
            + "_loss_over_epoch_line_plot_CV_group_"
            + str(i + 1)
            + "_rep_"
            + str(j + 1)
            + ".png",
        )
        loss_plot(
            train=history.history["custom_metric"],
            test=history.history["val_custom_metric"],
            fn=output_dir
            + "/loss_plot/"
            + "median"
            + "_metric_over_epoch_line_plot_CV_group_"
            + str(i + 1)
            + "_rep_"
            + str(j + 1)
            + ".png",
        )
    best_model.save(
        output_dir
        + "/models/"
        + "median"
        + "_best_model_CV_group_"
        + str(i + 1)
        + ".h5"
    )
    y_pred = best_model.predict(x_test, verbose=1).ravel()
    for k in range(len(y_test)):
        res.loc[len(res)] = list(
            map(
                str,
                [i + 1, y_df.loc[test_idx]["id"].to_numpy()[k], y_test[k], y_pred[k]],
            )
        )
stat.to_csv(
    output_dir + "/prediction/" + "median" + "_train_test_CV_stats.csv", index=False
)
res.to_csv(
    output_dir + "/prediction/" + "median" + "_train_test_CV_prediction_results.csv",
    index=False,
)
scatter_plot(
    res["y"],
    res["y_pred"],
    fn=output_dir + "/scatter_plot/" + "median" + "_final_scatter_plot.png",
)
