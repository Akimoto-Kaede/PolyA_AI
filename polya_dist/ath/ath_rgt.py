import tensorflow as tf
import keras
import numpy as np
import math
import pandas as pd
from sklearn.model_selection import KFold
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import json
import random
import ot

L_NT = 3000  # 原始序列长度 (nt)
BIN_WIDTH = 20.0
D_RES = 128  # 卷积通道 / GRU 隐单元
D_MODEL = 256  # Transformer 维度
N_HEAD = 8  # Multi-Head 数
N_LATENT = 9  # 输出 z 维度
DILATIONS = [1, 2, 4]
λ = 0.0

centers = np.arange(10, 210, 20, dtype="float32")[:-1] + 10


def w1_loss(y_true, y_pred):
    cdf_t = tf.cumsum(y_true, axis=-1)
    cdf_pr = tf.cumsum(y_pred, axis=-1)
    return tf.reduce_mean(tf.abs(cdf_t - cdf_pr), axis=-1) * BIN_WIDTH


def kl(y_true, y_pred):
    kl = tf.keras.losses.KLDivergence()(y_true, y_pred)
    # w1 = w1_loss(y_true, y_pred)
    return kl  # + λ * w1


def w2_batch_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)  # 确保都是 np.array
    y_pred = np.asarray(y_pred, dtype=np.float32)
    d = [
        np.sqrt(ot.wasserstein_1d(centers, centers, t, p, p=2))
        for t, p in zip(y_true, y_pred)
    ]
    return np.float32(np.mean(d))


def w2_metric(y_true, y_pred):
    return tf.py_function(func=w2_batch_np, inp=[y_true, y_pred], Tout=tf.float32)


def sum_wasserstein_1d(y_true, y_pred):
    cdf_t = tf.cumsum(y_true, axis=-1)
    cdf_pr = tf.cumsum(y_pred, axis=-1)
    return tf.reduce_sum(tf.abs(cdf_t - cdf_pr), axis=-1)


# ---------- Residual Dilated Conv block ----------
def res_block(x, k_size, dilation, name):
    h = keras.layers.Conv1D(
        D_RES,
        k_size,
        padding="same",
        dilation_rate=dilation,
        activation="gelu",
        name=f"{name}_conv1",
    )(x)
    h = keras.layers.BatchNormalization(name=f"{name}_bn1")(h)
    h = keras.layers.Conv1D(
        D_RES,
        k_size,
        padding="same",
        dilation_rate=dilation,
        activation=None,
        name=f"{name}_conv2",
    )(h)
    h = keras.layers.BatchNormalization(name=f"{name}_bn2")(h)
    return keras.layers.Activation("gelu", name=f"{name}_out")(x + h)


# ---------- Transformer block ----------
def transformer_block(x, name):
    # LayerNorm + MH-Attention
    h1 = keras.layers.LayerNormalization(epsilon=1e-5, name=f"{name}_ln1")(x)
    h1 = keras.layers.MultiHeadAttention(
        num_heads=N_HEAD, key_dim=D_MODEL // N_HEAD, dropout=0.1, name=f"{name}_mha"
    )(h1, h1)
    x = x + h1  # residual

    # FFN
    h2 = keras.layers.LayerNormalization(epsilon=1e-5, name=f"{name}_ln2")(x)
    h2 = keras.layers.Dense(4 * D_MODEL, activation="gelu", name=f"{name}_ffn1")(h2)
    h2 = keras.layers.Dense(D_MODEL, name=f"{name}_ffn2")(h2)
    return x + h2  # residual


def rgt(L_NT=L_NT, D_RES=D_RES, DILATIONS=DILATIONS, N_LATENT=N_LATENT):
    inp = keras.Input(shape=(L_NT, 4), name="input")

    # 0) ConvStem -- stride 2 ×2   (3000→750)
    x = keras.layers.Conv1D(
        D_RES, 11, strides=2, padding="same", activation="gelu", name="stem_conv1"
    )(inp)
    x = keras.layers.Conv1D(
        D_RES, 11, strides=2, padding="same", activation="gelu", name="stem_conv2"
    )(x)

    # 1) Residual Dilated CNN      (750→375 通过后续池化，下采样=2)
    for i, d in enumerate(DILATIONS):
        x = res_block(x, k_size=11 if d < 4 else 15, dilation=d, name=f"res{i+1}")

    # 2) Bi-GRU ×2
    x = keras.layers.Bidirectional(
        keras.layers.GRU(D_RES, return_sequences=True), name="bigru1"
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.GRU(D_RES, return_sequences=True), name="bigru2"
    )(x)

    # 3) Avg-Pool /3  (375→125)
    x = keras.layers.AveragePooling1D(pool_size=3, name="avgpool")(x)

    # 4) 2 × Transformer
    x = transformer_block(x, name="trans1")
    x = transformer_block(x, name="trans2")

    # 5) Global pool + head
    g = keras.layers.GlobalAveragePooling1D(name="gap")(x)
    g = keras.layers.Dense(256, activation="gelu", name="dense256")(g)
    g = keras.layers.Dense(128, activation="gelu", name="dense128")(g)
    z_pred = keras.layers.Dense(N_LATENT, activation="softmax", name="embed")(g)

    model = keras.Model(inp, z_pred, name="ResGRUTransformer")

    model.compile(
        optimizer=keras.optimizers.AdamW(3e-4),
        loss="categorical_crossentropy",
        metrics=[w1_loss, kl, w2_metric],
    )
    return model


def train_model(model, x_train, y_train, x_test, y_test, session_id: str = ""):
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="/scem/work/zhangyt/project/ai/polya_dist/ath/model/"
            + session_id
            + "_callback_best_model.h5",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=15),
    ]
    out = model.fit(
        x=x_train,
        y=y_train,
        batch_size=50,
        epochs=80,
        verbose=2,
        shuffle=True,
        callbacks=callbacks,
        validation_data=(x_test, y_test),
    )
    model = keras.models.load_model(
        "/scem/work/zhangyt/project/ai/polya_dist/ath/model/"
        + session_id
        + "_callback_best_model.h5",
        custom_objects={
            "w2_metric": w2_metric,
            "kl": kl,
            "w1_loss": w1_loss,
        },
    )
    return out, model


def loss_plot(train, test, fn):
    plt.clf()
    plt.plot(train, label="train")
    plt.plot(test, label="test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss or W₂")
    plt.savefig(fn)


x_ary = np.load("/scem/work/zhangyt/project/ai/polya_dist/ath/data/x_input.npy").astype(
    "float32"
)
y_df = pd.read_csv("/scem/work/zhangyt/project/ai/polya_dist/ath/data/y_input.csv")


hist_true = np.vstack(y_df["leaf_polya_ratio"].apply(json.loads)).astype("float32")
hist_true += np.float32(1e-7)
hist_true /= hist_true.sum(1, keepdims=True)

kf = KFold(n_splits=5, shuffle=True, random_state=57)
res = []

for i, (tr, val) in enumerate(kf.split(x_ary)):
    best_w2 = 1000
    x_tr, x_val = x_ary[tr], x_ary[val]
    y_tr = hist_true[tr]
    y_val = hist_true[val]
    model = rgt()
    history, model = train_model(
        model, x_tr, y_tr, x_val, y_val, session_id="resgrutransformer_ath"
    )
    model.save(
        "/scem/work/zhangyt/project/ai/polya_dist/ath"
        + "/model/"
        + "resgrutransformer_ath"
        + "_best_model_CV_group_"
        + str(i + 1)
        + ".h5"
    )
    scores = model.evaluate(x_val, y_val, batch_size=50, verbose=2)
    pred = model.predict(x_val, batch_size=50, verbose=2)
    """w2 = wasserstein_2d(y_val, pred)
    if w2 < best_w2:
        best_w2 = w2
        best_model = model"""
    del model
    keras.backend.clear_session()
    loss_plot(
        train=history.history["loss"],
        test=history.history["val_loss"],
        fn="/scem/work/zhangyt/project/ai/polya_dist/ath/loss_plot/"
        + "resgrutransformer_ath"
        + "_loss_over_epoch_line_plot_CV_group_"
        + str(i + 1)
        + ".png",
    )
    loss_plot(
        train=history.history["w2_metric"],
        test=history.history["val_w2_metric"],
        fn="/scem/work/zhangyt/project/ai/polya_dist/ath/loss_plot/"
        + "resgrutransformer_ath"
        + "_w2_over_epoch_line_plot_CV_group_"
        + str(i + 1)
        + ".png",
    )
    """best_model.save(
        "/scem/work/zhangyt/project/ai/polya_dist/ath"
        + "/model/"
        + "resgrutransformer_ath"
        + "_best_model_CV_group_"
        + str(i + 1)
        + ".h5")"""
    rows = pd.DataFrame(
        {
            "group": str(i + 1),
            "mRNA": y_df.iloc[val]["mRNA"].values,
            "polya_ratio": list(map(json.dumps, y_val.tolist())),
            "hist_pred": list(map(json.dumps, pred.tolist())),
        }
    )
    res.append(rows)
pd.concat(res, ignore_index=True).to_csv(
    "/scem/work/zhangyt/project/ai/polya_dist/ath/result/crossentrophy_train_val_CV_result.tsv",
    sep="\t",
    index=False,
)
