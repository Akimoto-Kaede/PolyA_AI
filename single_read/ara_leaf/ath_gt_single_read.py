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

L_NT = 1800  # 原始序列长度 (nt)
D_RES = 64  # 卷积通道 / GRU 隐单元
D_MODEL = 128  # Transformer 维度
N_HEAD = 8  # Multi-Head 数
N_LATENT = 1  # 输出维度
DILATIONS = [1, 2, 4]


def pearson_r2(y_true, y_pred):
    x = y_true - tf.reduce_mean(y_true)
    y = y_pred - tf.reduce_mean(y_pred)
    r = tf.reduce_sum(x * y) / tf.sqrt(
        tf.reduce_sum(x * x) * tf.reduce_sum(y * y) + tf.keras.backend.epsilon()
    )
    return r**2


@keras.saving.register_keras_serializable()
class PositionalEmbedding(keras.layers.Layer):

    def __init__(self, d_model, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.d_model = d_model
        self.embeddings = None

    def build(self, input_shape):
        # input_shape: (batch, seq_len, d_model_in)
        seq_len = input_shape[1]  # 自动获取序列长度
        self.embeddings = self.add_weight(
            "pos_emb",
            shape=(seq_len, self.d_model),
            initializer="uniform",
            trainable=True,
        )
        super(PositionalEmbedding, self).build(input_shape)

    def call(self, x):
        return x + self.embeddings[None, :, :]  # 广播到 batch 维度

    def compute_output_shape(self, input_shape):
        return input_shape


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

    # 0) ConvStem -- stride 2 ×2
    x = keras.layers.Conv1D(
        D_RES, 11, strides=2, padding="same", activation="gelu", name="stem_conv1"
    )(inp)
    x = keras.layers.Conv1D(
        D_RES, 11, strides=2, padding="same", activation="gelu", name="stem_conv2"
    )(x)

    ## 1) Residual Dilated CNN
    # for i, d in enumerate(DILATIONS):
    #    x = res_block(x, k_size=11 if d < 4 else 15, dilation=d, name=f"res{i+1}")
    # x = keras.layers.SpatialDropout1D(0.1)(x)

    # 2) Bi-GRU ×2
    x = keras.layers.Bidirectional(
        keras.layers.GRU(
            D_RES,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.000001),
        ),
        name="bigru",
    )(x)

    # 3) Avg-Pool /3
    x = keras.layers.AveragePooling1D(pool_size=2, name="avgpool")(x)

    # 4) 2 × Transformer
    x = keras.layers.Dense(D_MODEL, use_bias=False, name="proj_to_model")(x)
    x = PositionalEmbedding(d_model=D_MODEL, name="pos_enc")(x)
    x = transformer_block(x, name="trans1")
    x = transformer_block(x, name="trans2")

    # 5) Global pool + head
    g = keras.layers.GlobalAveragePooling1D(name="gap")(x)
    g = keras.layers.Dense(
        96,
        name="dense96",
        kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.000001),
    )(g)
    g = keras.layers.Dropout(0.1)(g)
    g = keras.layers.BatchNormalization()(g)
    g = keras.layers.Activation(keras.activations.gelu)(g)
    out = keras.layers.Dense(N_LATENT, name="output")(g)

    model = keras.Model(inp, out, name="GRUTransformer")
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4, decay_steps=25600, alpha=0.2
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(lr_schedule, weight_decay=1e-4),
        loss=tf.keras.losses.Huber(delta=1.5),
        metrics=[
            pearson_r2,
            tf.keras.metrics.MeanAbsoluteError(name="MAE"),
            tf.keras.metrics.MeanSquaredError(name="MSE"),
        ],
    )
    return model


def train_model(model, x_train, y_train, x_test, y_test, session_id: str = ""):
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="/scem/work/zhangyt/project/ai/single_read/ara_leaf/model/"
            + session_id
            + "_callback_best_model.h5",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10),
    ]
    out = model.fit(
        x=x_train,
        y=y_train,
        batch_size=32,
        epochs=80,
        verbose=2,
        shuffle=True,
        callbacks=callbacks,
        validation_data=(x_test, y_test),
    )
    model = keras.models.load_model(
        "/scem/work/zhangyt/project/ai/single_read/ara_leaf/model/"
        + session_id
        + "_callback_best_model.h5",
        custom_objects={
            "pearson_r2": pearson_r2,
        },
    )
    return out, model


def loss_plot(train, test, fn):
    plt.clf()
    plt.plot(train, label="train")
    plt.plot(test, label="test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE or r²")
    plt.savefig(fn)


train_ary = np.load(
    "/scem/work/zhangyt/project/ai/dnabert2/polya_single_read/data/gene_200/train_data_leaf.250820.npy"
)
train_df = pd.read_csv(
    "/scem/work/zhangyt/project/ai/dnabert2/polya_single_read/data/gene_200/train_data_leaf_shuffled.250820.csv"
)
eval_ary = np.load(
    "/scem/work/zhangyt/project/ai/dnabert2/polya_single_read/data/gene_200/eval_data_leaf.250820.npy"
)
eval_df = pd.read_csv(
    "/scem/work/zhangyt/project/ai/dnabert2/polya_single_read/data/gene_200/eval_data_leaf_shuffled.250820.csv"
)
x_tr, x_val = train_ary, eval_ary
y_tr = train_df.loc[:, "log1p_scaled"].to_numpy("float32")[:, None]
y_val = eval_df.loc[:, "log1p_scaled"].to_numpy("float32")[:, None]
model = rgt()
history, model = train_model(
    model, x_tr, y_tr, x_val, y_val, session_id="ath_leaf_single_read_gene200+"
)
model.save(
    "/scem/work/zhangyt/project/ai/single_read/ara_leaf"
    + "/model/"
    + "ath_leaf_single_read_gene200+"
    + "_best_model_CV_group_1"
    + ".h5"
)
scores = model.evaluate(x_val, y_val, batch_size=32, verbose=2)
pred = model.predict(x_val, batch_size=32, verbose=2)
df_out = pd.DataFrame(
    {
        "sequence": eval_df["sequence"],
        "y_true": eval_df["log1p_scaled"],  # 如果有真实值
        "y_pred": pred.flatten(),  # 注意 shape，可能需要 flatten
    }
)

df_out.to_csv(
    "/scem/work/zhangyt/project/ai/single_read/ara_leaf/result/ath_gt_gene200+_predictions.csv",
    index=False,
)
"""w2 = wasserstein_2d(y_val, pred)
if w2 < best_w2:
    best_w2 = w2
    best_model = model"""
del model
keras.backend.clear_session()
loss_plot(
    train=history.history["loss"],
    test=history.history["val_loss"],
    fn="/scem/work/zhangyt/project/ai/single_read/ara_leaf/loss_plot/"
    + "ath_leaf_single_read_gene200+"
    + "_loss_over_epoch_line_plot_CV_group_"
    + str(1)
    + ".png",
)
loss_plot(
    train=history.history["pearson_r2"],
    test=history.history["val_pearson_r2"],
    fn="/scem/work/zhangyt/project/ai/single_read/ara_leaf/loss_plot/"
    + "ath_leaf_single_read_gene200+"
    + "_r2_over_epoch_line_plot_CV_group_"
    + str(1)
    + ".png",
)
"""best_model.save(
    "/scem/work/zhangyt/project/ai/single_read/ara_leaf"
    + "/model/"
    + "ath_leaf_single_read_gene200+"
    + "_best_model_CV_group_"
    + str(i + 1)
    + ".h5")"""
rows = pd.DataFrame(
    {
        "group": str(1),
        "seq": eval_df.loc[:, "sequence"].values,
        "y": y_val.squeeze().tolist(),
        "y_pred": pred.squeeze().tolist(),
    }
)
'''x_ary = np.load(
    "/scem/work/zhangyt/project/ai/single_read/ara_leaf/data/x_input.npy"
).astype("float32")
y_df = pd.read_csv(
    "/scem/work/zhangyt/project/ai/single_read/ara_leaf/data/y_input.csv"
)


kf = KFold(n_splits=5, shuffle=True, random_state=57)
res = []

for i, (tr, val) in enumerate(kf.split(x_ary)):
    best_r2 = 0
    x_tr, x_val = x_ary[tr], x_ary[val]
    y_tr = y_df.loc[tr, "standardized"].to_numpy("float32")[:, None]
    y_val = y_df.loc[val, "standardized"].to_numpy("float32")[:, None]
    model = rgt()
    history, model = train_model(
        model, x_tr, y_tr, x_val, y_val, session_id="ath_leaf_single_read_gene200+"
    )
    model.save(
        "/scem/work/zhangyt/project/ai/single_read/ara_leaf"
        + "/model/"
        + "ath_leaf_single_read_gene200+"
        + "_best_model_CV_group_"
        + str(i + 1)
        + ".h5"
    )
    scores = model.evaluate(x_val, y_val, batch_size=100, verbose=2)
    pred = model.predict(x_val, batch_size=100, verbose=2)
    """w2 = wasserstein_2d(y_val, pred)
    if w2 < best_w2:
        best_w2 = w2
        best_model = model"""
    del model
    keras.backend.clear_session()
    loss_plot(
        train=history.history["loss"],
        test=history.history["val_loss"],
        fn="/scem/work/zhangyt/project/ai/single_read/ara_leaf/loss_plot/"
        + "ath_leaf_single_read_gene200+"
        + "_loss_over_epoch_line_plot_CV_group_"
        + str(i + 1)
        + ".png",
    )
    loss_plot(
        train=history.history["pearson_r2"],
        test=history.history["val_pearson_r2"],
        fn="/scem/work/zhangyt/project/ai/single_read/ara_leaf/loss_plot/"
        + "ath_leaf_single_read_gene200+"
        + "_r2_over_epoch_line_plot_CV_group_"
        + str(i + 1)
        + ".png",
    )
    """best_model.save(
        "/scem/work/zhangyt/project/ai/single_read/ara_leaf"
        + "/model/"
        + "ath_leaf_single_read_gene200+"
        + "_best_model_CV_group_"
        + str(i + 1)
        + ".h5")"""
    rows = pd.DataFrame(
        {
            "group": str(i + 1),
            "read_id": y_df.iloc[val]["read_id"].values,
            "y": y_val.squeeze().tolist(),
            "y_pred": pred.squeeze().tolist(),
        }
    )
    res.append(rows)
pd.concat(res, ignore_index=True).to_csv(
    "/scem/work/zhangyt/project/ai/single_read/ara_leaf/result/chr135_train_val_CV_result.tsv",
    sep="\t",
    index=False,
)'''
