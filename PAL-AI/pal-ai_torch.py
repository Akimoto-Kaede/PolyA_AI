import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from scipy import stats
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

output_dir = "/public/home/zhangyt/project/ai/PAL-AI"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(output_dir + "/models", exist_ok=True)
os.makedirs(output_dir + "/loss_plot", exist_ok=True)
os.makedirs(output_dir + "/prediction", exist_ok=True)
os.makedirs(output_dir + "/scatter_plot", exist_ok=True)


class PALModel(nn.Module):
    def __init__(
        self, input_dim=4, filters=64, kernel_size=5, gru_units=96, dropout_rate=0.1
    ):
        super(PALModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, filters, kernel_size)
        self.norm1 = nn.LayerNorm([filters, 1056 - kernel_size + 1])
        self.conv2 = nn.Conv1d(filters, filters, kernel_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.pool = nn.MaxPool1d(kernel_size=2)

        conv_out_len = (1056 - kernel_size + 1 - kernel_size + 1) // 2
        self.gru = nn.GRU(filters, gru_units, batch_first=True)
        self.fc1 = nn.Linear(gru_units, 96)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(96, 1)
        self.activation = nn.SELU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # B x C x L
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.pool(self.dropout1(self.activation(self.conv2(x))))
        x = x.permute(0, 2, 1)  # B x T x F
        _, h = self.gru(x)
        x = self.activation(h[-1])
        x = self.dropout2(self.activation(self.fc1(x)))
        return self.fc2(x)


def pearson_loss(y_true, y_pred):
    x = y_true - y_true.mean()
    y = y_pred - y_pred.mean()
    numerator = (x * y).sum()
    denominator = torch.sqrt((x**2).sum() * (y**2).sum())
    r = numerator / (denominator + 1e-8)
    return -(r**2)


def train_one_fold(model, train_loader, val_loader, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    history = {"loss": [], "val_loss": [], "metric": [], "val_metric": []}
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_metric = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch).squeeze()
            loss = criterion(pred, y_batch)
            metric = pearson_loss(y_batch, pred)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            epoch_metric += metric.item()

        val_loss = 0
        val_metric = 0
        model.eval()
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                pred = model(x_val).squeeze()
                val_loss += criterion(pred, y_val).item()
                val_metric += pearson_loss(y_val, pred).item()

        history["loss"].append(epoch_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["metric"].append(epoch_metric / len(train_loader))
        history["val_metric"].append(val_metric / len(val_loader))

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model)
    return model, history


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


x_ary = np.load(output_dir + "/data/x_input.npy").astype(np.float32)
y_df = pd.read_csv(output_dir + "/data/y_input.csv")
y_ary = y_df["y"].to_numpy().astype(np.float32)
stat = []
res = []

kf = KFold(n_splits=10, shuffle=True, random_state=57)
for i, (train_idx, test_idx) in enumerate(kf.split(x_ary, y_ary)):
    best_r2 = 0
    for j in range(5):
        model = PALModel().to(device)
        x_train = torch.tensor(x_ary[train_idx])
        y_train = torch.tensor(y_ary[train_idx])
        x_test = torch.tensor(x_ary[test_idx])
        y_test = torch.tensor(y_ary[test_idx])
        train_loader = DataLoader(
            TensorDataset(x_train, y_train), batch_size=50, shuffle=True
        )
        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=50)

        model, history = train_one_fold(
            model, train_loader, test_loader, epochs=30, lr=0.001
        )

        y_pred = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                yp = model(xb).squeeze().cpu().numpy()
                y_pred.extend(yp)
        y_pred = np.array(y_pred)
        r = stats.linregress(y_ary[test_idx], y_pred).rvalue
        r2 = r**2

        stat.append(
            [
                i + 1,
                j + 1,
                round(r2, 3),
                history["val_loss"][-1],
                history["val_metric"][-1],
            ]
        )
        loss_plot(
            history["loss"],
            history["val_loss"],
            f"{output_dir}/loss_plot/group_{i+1}_rep_{j+1}_loss.png",
        )
        loss_plot(
            history["metric"],
            history["val_metric"],
            f"{output_dir}/loss_plot/group_{i+1}_rep_{j+1}_metric.png",
        )

        if r2 > best_r2:
            best_r2 = r2
            best_pred = y_pred
            best_model = model

    torch.save(
        best_model.state_dict(), f"{output_dir}/models/best_model_group_{i+1}.pt"
    )

    for idx, yp in zip(test_idx, best_pred):
        res.append([i + 1, y_df.loc[idx, "id"], y_ary[idx], yp])

pd.DataFrame(stat, columns=["group", "rep", "r2", "val_loss", "val_metric"]).to_csv(
    output_dir + "/prediction/torch_train_test_CV_stats.csv", index=False
)
res_df = pd.DataFrame(res, columns=["group", "id", "y", "y_pred"])
res_df.to_csv(
    output_dir + "/prediction/torch_train_test_CV_prediction_results.csv", index=False
)

scatter_plot(
    res_df["y"],
    res_df["y_pred"],
    fn=output_dir + "/scatter_plot/final_scatter_plot_torch.png",
)
