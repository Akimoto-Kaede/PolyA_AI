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
EPOCHS = 200
BETA = 0.8

dist_df = pd.read_csv(
    "/scem/work/zhangyt/project/ai/polya_dist/data/polya_distribution.tsv", sep="\t"
)
count = dist_df["polya_lengths"].apply(lambda x: len(x.split(",")))
dist_df["count"] = count
filtered_df = dist_df[dist_df["count"] >= 200].reset_index(drop=True)
X = np.vstack(filtered_df["polya_ratio"].apply(json.loads)).astype("float32")
X += np.float32(1e-8)
X /= X.sum(axis=1, keepdims=True)


def wasserstein_1d(p, q):
    cdf_p = tf.cumsum(p, axis=-1)
    cdf_q = tf.cumsum(q, axis=-1)
    w1 = tf.reduce_sum(tf.abs(cdf_p - cdf_q), axis=-1)
    return tf.reduce_mean(w1)


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
        self.input_layer = keras.layers.InputLayer(input_shape=(L,))
        self.enc = Encoder(LATENT)
        self.dec = Decoder(tau)
        self.kl_metric = keras.metrics.Mean(name="mean_KL")
        self.build((None, L))

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

    def test_step(self, x):
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

        self.kl_metric.update_state(kl)

        return {
            "loss": loss,
            "rec": loss_rec,
            "mmd": loss_mmd,
            "kl": self.kl_metric.result(),
        }

    def encode(self, x):
        return self.enc(x)[0]

    def encode_mean(self, x):
        mu, _ = self.enc(x)
        return mu

    def reconstruct(self, z):
        z = tf.convert_to_tensor(z, dtype=tf.float32)
        if z.ndim == 1:
            z = z[None, :]
        return self.dec(z).numpy()

    def call(self, inputs, training=False):
        inputs = self.input_layer(inputs)
        mu, _ = self.enc(inputs, training=training)
        return self.dec(mu, training=training)

    def get_config(self):
        return {
            "LATENT": self.LATENT,
            "tau": self.tau,
            **super().get_config(),
        }

    def get_build_config(self):
        return {"input_shape": (None, 64)}

    @classmethod
    def build_from_config(cls, config):
        model = cls(**config)
        dummy = tf.zeros((1, 64), dtype="float32")
        model(dummy)
        return model


def loss_plot(train, test, fn):
    plt.clf()
    plt.plot(train, label="train")
    plt.plot(test, label="test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(fn)


pack = lambda arr: json.dumps(arr.tolist(), separators=(",", ":"))
records = []

kf = KFold(n_splits=5, shuffle=True, random_state=57)
for fold, (tr, val) in enumerate(kf.split(X), 1):
    ds_tr = (
        tf.data.Dataset.from_tensor_slices(X[tr])
        .shuffle(len(tr))
        .batch(BATCH)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_val = (
        tf.data.Dataset.from_tensor_slices(X[val])
        .batch(BATCH)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = WAE2D()
    model(tf.zeros((1, L)))
    model.compile(optimizer=keras.optimizers.Adam(1e-3))
    es = keras.callbacks.EarlyStopping(
        "val_loss", patience=15, restore_best_weights=True
    )
    cp = keras.callbacks.ModelCheckpoint(
        f"/scem/work/zhangyt/project/ai/polya_dist/embedding/wae/model/callback_best_wae2d_CV_group_{fold}",
        save_format="tf",
        save_best_only=True,
        monitor="val_loss",
    )
    history = model.fit(
        ds_tr,
        validation_data=ds_val,
        epochs=EPOCHS,
        verbose=2,
        callbacks=[es, cp],
    )
    loss_plot(
        train=history.history["loss"],
        test=history.history["val_loss"],
        fn="/scem/work/zhangyt/project/ai/polya_dist/embedding/wae/loss_plot/"
        + "wae2d"
        + "_loss_over_epoch_line_plot_CV_group_"
        + str(fold)
        + ".png",
    )
    # 验证集嵌入 + 重建
    z_val = model.encode(X[val]).numpy()
    h_pred = model.reconstruct(z_val)

    for idx, z_vec, h_vec in zip(val, z_val, h_pred):
        records.append(
            {
                "fold": fold,
                "mRNA": filtered_df.at[idx, "mRNA"],
                "hist_true": pack(X[idx]),
                "z_pred": pack(z_vec),
                "hist_pred": pack(h_vec),
            }
        )
    del model
    keras.backend.clear_session()
pd.DataFrame(records).to_csv(
    "/scem/work/zhangyt/project/ai/polya_dist/embedding/wae/result/wae_kfold_embeddings.tsv",
    sep="\t",
    index=False,
)
