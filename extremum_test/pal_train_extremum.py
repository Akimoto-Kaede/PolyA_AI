import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
import pandas as pd
from keras import layers
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import matplotlib.pyplot as plt

tf.debugging.enable_check_numerics()
print(tf.config.list_physical_devices("GPU"))
tf.keras.utils.set_random_seed(57)
output_dir = "/scem/work/zhangyt/project/ai/extremum_test"


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
    l1: float = 0.005,
    l2: float = 0.000001,
) -> keras.Model:
    inp = layers.Input(shape=(1056, 4), name="input")
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size)(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Activation(keras.activations.selu)(x)
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size)(x)
    x = layers.Dropout(rate=dropout_rate)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation(keras.activations.selu)(x)
    x = layers.GRU(
        units=96,
        go_backwards=False,
        kernel_regularizer=keras.regularizers.l1_l2(l1, l2),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.selu)(x)
    reg_out = layers.Dense(
        units=96,
        kernel_regularizer=keras.regularizers.l1_l2(l1, l2),
    )(x)
    reg_out = layers.Dropout(rate=dropout_rate)(reg_out)
    reg_out = layers.BatchNormalization()(reg_out)
    reg_out = layers.Activation(keras.activations.selu)(reg_out)
    reg_out = layers.Dense(units=1, name="regression")(reg_out)
    cls_out = layers.Dense(32, activation=keras.activations.selu)(x)
    cls_out = layers.Dense(units=1, activation="sigmoid", name="classification")(
        cls_out
    )
    model = keras.Model(inputs=inp, outputs=[reg_out, cls_out])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"regression": "mse", "classification": "binary_crossentropy"},
        metrics={
            "regression": ["mae", custom_metric],
            "classification": ["accuracy", keras.metrics.AUC()],
        },
        loss_weights={"regression": 1.0, "classification": 1.0},
    )
    return model


def train_model(
    x_train,
    y_reg_train,
    y_cls_train,
    x_val,
    y_reg_val,
    y_cls_val,
    session_id: str = "",
    **kwargs,
):
    model = pal_model(**kwargs)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=output_dir + "/models/" + session_id + "_callback_best_model.h5",
            monitor="val_regression_loss",
            mode="min",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_regression_loss", mode="min", patience=10
        ),
    ]
    out = model.fit(
        x=x_train,
        y={"regression": y_reg_train, "classification": y_cls_train},
        batch_size=50,
        epochs=30,
        verbose=1,
        shuffle=True,
        callbacks=callbacks,
        validation_data=(x_val, {"regression": y_reg_val, "classification": y_cls_val}),
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


x_ary = np.load("/scem/work/zhangyt/project/ai/extremum_test/data/x_input.npy")
y_df = pd.read_csv("/scem/work/zhangyt/project/ai/extremum_test/data/y_input.csv")
stat = pd.DataFrame(
    None,
    columns=["group", "rep", "r2", "loss", "mae", "custom_metric", "accuracy", "AUC"],
)
res = pd.DataFrame(None, columns=["group", "id", "y", "y_pred", "label"])
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=57)
for i, (tr_idx, val_idx) in enumerate(kf.split(x_ary, y_df["label"])):
    best_r2 = 0
    for j in range(5):
        x_tr, x_val = x_ary[tr_idx], x_ary[val_idx]
        y_tr_reg, y_val_reg = (
            y_df.loc[tr_idx, "y"].to_numpy(),
            y_df.loc[val_idx, "y"].to_numpy(),
        )
        y_tr_cls, y_val_cls = (
            y_df.loc[tr_idx, "label"].to_numpy(),
            y_df.loc[val_idx, "label"].to_numpy(),
        )
        history, model = train_model(
            x_tr, y_tr_reg, y_tr_cls, x_val, y_val_reg, y_val_cls, session_id="extremum"
        )
        model.save(
            output_dir
            + "/models/"
            + "extremum"
            + "_best_model_CV_group_"
            + str(i + 1)
            + "_rep_"
            + str(j + 1)
            + ".h5"
        )
        y_pred_reg_arr, y_pred_cls_prob = model.predict(x_val, batch_size=50, verbose=0)
        y_pred_reg = y_pred_reg_arr.ravel()
        r_value = stats.linregress(y_val_reg, y_pred_reg).rvalue
        scores = model.evaluate(
            x_val,
            {"regression": y_val_reg, "classification": y_val_cls},
            batch_size=50,
            verbose=1,
            return_dict=True,
        )
        if r_value**2 > best_r2:
            best_r2 = r_value**2
            best_model = model
        del model
        keras.backend.clear_session()
        stat.loc[len(stat)] = [
            i + 1,
            j + 1,
            round(r_value**2, 3),
            scores["regression_loss"],
            scores["regression_mae"],
            scores["regression_custom_metric"],
            scores["classification_accuracy"],
            scores["classification_auc"],
        ]
        loss_plot(
            train=history.history["regression_loss"],
            test=history.history["val_regression_loss"],
            fn=output_dir
            + "/loss_plot/"
            + "extremum"
            + "_loss_over_epoch_line_plot_CV_group_"
            + str(i + 1)
            + "_rep_"
            + str(j + 1)
            + ".png",
        )
        loss_plot(
            train=history.history["regression_custom_metric"],
            test=history.history["val_regression_custom_metric"],
            fn=output_dir
            + "/loss_plot/"
            + "extremum"
            + "_metric_over_epoch_line_plot_CV_group_"
            + str(i + 1)
            + "_rep_"
            + str(j + 1)
            + ".png",
        )
    best_model.save(
        output_dir
        + "/models/"
        + "extremum"
        + "_best_model_CV_group_"
        + str(i + 1)
        + ".h5"
    )
    y_pred_reg_arr, y_pred_cls_prob = best_model.predict(
        x_val, batch_size=50, verbose=1
    )
    y_pred_reg = y_pred_reg_arr.ravel()
    y_pred_cls = y_pred_cls_prob.ravel()
    for k in range(len(y_val_reg)):
        res.loc[len(res)] = [
            i + 1,
            y_df.loc[val_idx]["id"].to_numpy()[k],
            y_val_reg[k],
            y_pred_reg[k],
            y_pred_cls[k],
        ]
stat.to_csv(
    output_dir + "/prediction/" + "extremum" + "_train_test_CV_stats.csv", index=False
)
res.to_csv(
    output_dir + "/prediction/" + "extremum" + "_train_test_CV_prediction_results.csv",
    index=False,
)
res = pd.read_csv(
    output_dir + "/prediction/" + "extremum" + "_train_test_CV_prediction_results.csv"
)
scatter_plot(
    res["y"],
    res["y_pred"],
    fn=output_dir + "/scatter_plot/" + "extremum" + "_final_scatter_plot.png",
)
