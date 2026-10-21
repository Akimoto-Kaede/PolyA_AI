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

L = 64
LATENT = 2
BATCH = 50
BETA = 0.8


def wasserstein_1d(p, q):
    cdf_p = tf.cumsum(p, axis=-1)
    cdf_q = tf.cumsum(q, axis=-1)
    w1 = tf.reduce_sum(tf.abs(cdf_p - cdf_q), axis=-1)
    return tf.reduce_mean(w1) * 5


def mmd_rbf(z, sigma=None):
    if sigma is None:
        sigma = tf.maximum(tf.math.reduce_mean(tf.square(z)), 1e-3)

    def k(a, b):
        return tf.exp(-tf.square(a - b) / (2.0 * sigma))

    prior = tf.random.normal(tf.shape(z))
    xx = k(z[:, None, :], z[None, :, :])
    pp = k(prior[:, None, :], prior[None, :, :])
    xp = k(z[:, None, :], prior[None, :, :])
    return tf.reduce_mean(xx) + tf.reduce_mean(pp) - 2.0 * tf.reduce_mean(xp)


@keras.saving.register_keras_serializable()
class Encoder(keras.Model):

    def __init__(self, LATENT=2):
        super().__init__()
        self.backbone = keras.Sequential(
            [
                keras.layers.Input(shape=(L,)),
                keras.layers.Dense(256, activation="gelu"),
                keras.layers.Dense(128, activation="gelu"),
                keras.layers.Dense(64, activation="gelu"),
            ]
        )
        self.mu = keras.layers.Dense(LATENT)
        self.logv = keras.layers.Dense(LATENT)

    def call(self, x):
        h = self.backbone(x)
        return self.mu(h), self.logv(h)


@keras.saving.register_keras_serializable()
class Decoder(keras.Model):

    def __init__(self, tau=0.6):
        super().__init__()
        self.tau = tau
        self.net = keras.Sequential(
            [
                keras.layers.Input(shape=(LATENT,)),
                keras.layers.Dense(256, activation="gelu"),
                keras.layers.Dense(256, activation="gelu"),
                keras.layers.Dense(128, activation="gelu"),
                keras.layers.Dense(L),
            ]
        )

    def call(self, z):
        logits = self.net(z) / self.tau
        return tf.nn.softmax(logits, axis=-1)


@keras.saving.register_keras_serializable()
class WAE2D(keras.Model):

    def __init__(self, LATENT=2, tau=0.6, **kwargs):
        super().__init__(**kwargs)
        self.LATENT = LATENT
        self.tau = tau
        self.enc = Encoder(LATENT)
        self.dec = Decoder(tau)
        self.kl_metric = keras.metrics.Mean(name="mean_KL")

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, x):
        with tf.GradientTape() as tape:
            mu, logv = self.enc(x)
            eps = tf.random.normal(tf.shape(mu))
            z = mu + tf.exp(0.5 * logv) * eps
            x_hat = self.dec(z)

            loss_rec = wasserstein_1d(x, x_hat)
            loss_mmd = mmd_rbf(z)
            loss = loss_rec + BETA * loss_mmd

            kl = tf.reduce_mean(
                tf.reduce_sum(
                    x * (tf.math.log(x + 1e-8) - tf.math.log(x_hat + 1e-8)), axis=-1
                )
            )

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.kl_metric.update_state(kl)
        return {
            "loss": loss,
            "rec": loss_rec,
            "mmd": loss_mmd,
            "kl": self.kl_metric.result(),
        }

    def encode_mean(self, x):
        mu, _ = self.enc(x)
        return mu

    def reconstruct(self, z):
        z = tf.convert_to_tensor(z, dtype=tf.float32)
        if z.ndim == 1:
            z = z[None, :]
        return self.dec(z).numpy()

    def call(self, inputs, training=False):
        mu, _ = self.enc(inputs, training=training)
        return self.dec(mu, training=training)

    def get_config(self):
        return {
            "LATENT": self.LATENT,
            "tau": self.tau,
            **super().get_config(),
        }

    def get_build_config(self):
        return {"input_shape": (None, L)}

    @classmethod
    def build_from_config(cls, config):
        model = cls(**config)
        dummy = tf.zeros((1, L), dtype="float32")
        model(dummy)
        return model


wae = keras.models.load_model(
    "/scem/work/zhangyt/project/ai/polya_dist/1_resgrutransformer/data/wae2d",
    compile=False,
)
wae.dec.trainable = False

L_NT = 3000  # 原始序列长度 (nt)
D_RES = 128  # 卷积通道 / GRU 隐单元
D_MODEL = 256  # Transformer 维度
N_HEAD = 8  # Multi-Head 数
N_LATENT = 2  # 输出 z 维度
DILATIONS = [1, 2, 4]


def sum_wasserstein_1d(y_true, y_pred):
    cdf_t = tf.cumsum(y_true, axis=-1)
    cdf_pr = tf.cumsum(y_pred, axis=-1)
    return tf.reduce_sum(tf.abs(cdf_t - cdf_pr), axis=-1) * 5


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
    g = keras.layers.Dense(128, activation="gelu", name="dense128")(g)
    z_pred = keras.layers.Dense(N_LATENT, name="embed")(g)

    h = wae.dec(z_pred)
    h = keras.layers.Activation("linear", name="hist")(h)
    model = keras.Model(inp, {"embed": z_pred, "hist": h}, name="ResGRUTransformer")

    model.compile(
        optimizer=keras.optimizers.AdamW(3e-4),
        loss={"embed": keras.losses.MeanSquaredError(), "hist": sum_wasserstein_1d},
        loss_weights={"embed": 1.0, "hist": 0.8},
        metrics={
            "embed": keras.metrics.RootMeanSquaredError(name="rmse_z"),
            "hist": wasserstein_1d,
        },
    )
    return model


def train_model(model, x_train, y_train, x_test, y_test, session_id: str = ""):
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="/scem/work/zhangyt/project/ai/polya_dist/1_resgrutransformer/model/"
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
        "/scem/work/zhangyt/project/ai/polya_dist/1_resgrutransformer/model/"
        + session_id
        + "_callback_best_model.h5",
        custom_objects={
            "sum_wasserstein_1d": sum_wasserstein_1d,
            "wasserstein_1d": wasserstein_1d,
        },
    )
    return out, model


def loss_plot(train, test, fn):
    plt.clf()
    plt.plot(train, label="train")
    plt.plot(test, label="test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(fn)


x_ary = np.load(
    "/scem/work/zhangyt/project/ai/polya_dist/1_resgrutransformer/data/x_input.npy"
).astype("float32")
y_df = pd.read_csv(
    "/scem/work/zhangyt/project/ai/polya_dist/1_resgrutransformer/data/y_input.csv"
)
z_true = y_df[["embed_z1", "embed_z2"]].to_numpy("float32")


hist_true = np.vstack(y_df["polya_ratio"].apply(json.loads)).astype("float32")
hist_true += np.float32(1e-8)
hist_true /= hist_true.sum(1, keepdims=True)

kf = KFold(n_splits=5, shuffle=True, random_state=57)
res = pd.DataFrame(
    None,
    columns=[
        "group",
        "mRNA",
        "z",
        "polya_ratio",
        "z_pred",
        "reconstruct",
    ],
)
for i, (tr, val) in enumerate(kf.split(x_ary)):
    x_tr, x_val = x_ary[tr], x_ary[val]
    y_tr = {"embed": z_true[tr], "hist": hist_true[tr]}
    y_val = {"embed": z_true[val], "hist": hist_true[val]}
    model = rgt()
    history, model = train_model(
        model, x_tr, y_tr, x_val, y_val, session_id="resgrutranformer"
    )
    model.save(
        "/scem/work/zhangyt/project/ai/polya_dist/1_resgrutransformer"
        + "/model/"
        + "resgrutranformer"
        + "_best_model_CV_group_"
        + str(i + 1)
        + ".h5"
    )
    scores = model.evaluate(x_val, y_val, batch_size=50, verbose=2)
    pred = model.predict(x_val, batch_size=50, verbose=2)
    z_hat = pred["embed"]
    h_hat = pred["hist"]
    del model
    keras.backend.clear_session()
    loss_plot(
        train=history.history["loss"],
        test=history.history["val_loss"],
        fn="/scem/work/zhangyt/project/ai/polya_dist/1_resgrutransformer/loss_plot/"
        + "resgrutranformer"
        + "_loss_over_epoch_line_plot_CV_group_"
        + str(i + 1)
        + ".png",
    )
    n_val = x_val.shape[0]
    for k in range(n_val):
        res.loc[len(res)] = list(
            map(
                str,
                [
                    i + 1,
                    y_df.loc[val]["mRNA"].to_numpy()[k],
                    y_val["embed"][k].tolist(),
                    y_val["hist"][k].tolist(),
                    z_hat[k].tolist(),
                    h_hat[k].tolist(),
                ],
            )
        )
res.to_csv(
    "/scem/work/zhangyt/project/ai/polya_dist/1_resgrutransformer/result/train_val_CV_result.tsv",
    sep="\t",
    index=False,
)
