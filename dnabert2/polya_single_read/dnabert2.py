import os
import sys, logging
from scipy import stats
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from transformers.models.bert.configuration_bert import BertConfig
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

pretrained = (
    "/scem/work/zhangyt/project/ai/dnabert2/polya_single_read/data/dnabert2_117m"
)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained, local_files_only=True, trust_remote_code=True
)
config = BertConfig.from_pretrained(pretrained)
config.return_dict = True
backbone = AutoModel.from_pretrained(
    pretrained, local_files_only=True, trust_remote_code=True, config=config
)


class PolyARegression(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.reg_head = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden//2, 1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        seq_out = out[0]  # (B, L, H)
        cls = seq_out.mean(dim=1) # [CLS]
        pred = self.reg_head(cls).squeeze(-1)  # (B,)
        loss = None
        if labels is not None:
            loss = self.loss_fn(pred, labels.float())
        return pred, loss


class PolyADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_len=512):
        self.seqs = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        lab = self.labels[idx]
        tok = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in tok.items()}
        item["labels"] = torch.tensor(lab, dtype=torch.float32)
        return item


input_df = pd.read_csv(
    "/scem/work/zhangyt/project/ai/dnabert2/polya_single_read/data/ara_leaf_3000_chr1235.csv"
)
seq_list = input_df["read_seq"].tolist()
y_list = input_df["log1p_scaled"].tolist()
X_tr, X_val, y_tr, y_val = train_test_split(
    seq_list, y_list, test_size=0.1, random_state=42
)

train_ds = PolyADataset(X_tr, y_tr, tokenizer)
val_ds = PolyADataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PolyARegression(backbone).to(device)

optimizer = torch.optim.AdamW(
    [
        {"params": model.backbone.parameters(), "lr": 2e-5},
        {"params": model.reg_head.parameters(), "lr": 1e-4},
    ],
    weight_decay=1e-2,
)

num_epochs = 20
patience = 5
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)


def pearson_r(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    return np.corrcoef(y_true, y_pred)[0, 1]


log_dir = "/scem/work/zhangyt/project/ai/dnabert2/polya_single_read/result"
best_r = -1.0
best_epoch = -1
bad_count = 0
tr_losses, val_losses, tr_rs, val_rs = [], [], [], []

for epoch in range(num_epochs):
    # -------- train --------
    model.train()
    tr_loss = 0.0
    tr_pred_all, tr_gt_all = [], []
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred, loss = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        tr_loss += loss.item()
        tr_pred_all.append(pred.detach())
        tr_gt_all.append(batch["labels"].detach())
    tr_loss /= len(train_loader)
    tr_r = pearson_r(torch.cat(tr_gt_all), torch.cat(tr_pred_all))
    # -------- eval --------
    model.eval()
    val_loss = 0.0
    preds, gts = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred, loss = model(
                batch["input_ids"], batch["attention_mask"], batch["labels"]
            )
            val_loss += loss.item()
            preds.append(pred)
            gts.append(batch["labels"])
    val_loss /= len(val_loader)
    preds = torch.cat(preds)
    gts = torch.cat(gts)
    val_r = pearson_r(gts, preds)

    log.info(
        f"Epoch {epoch+1}: "
        f"tr_loss={tr_loss:.4f} tr_r={tr_r:.3f} | "
        f"val_loss={val_loss:.4f} val_r={val_r:.3f}"
    )

    # ---- 累积到内存曲线 ----
    tr_losses.append(tr_loss)
    val_losses.append(val_loss)
    tr_rs.append(tr_r)
    val_rs.append(val_r)

    # ---- 保存 r 最大的模型 ----
    if val_r > best_r:
        best_r = val_r
        best_epoch = epoch + 1
        bad_count = 0
        best_val_df = pd.DataFrame(
            {
                "epoch": best_epoch,
                "y_true": gts.cpu().numpy().ravel(),
                "y_pred": preds.cpu().numpy().ravel(),
            }
        )
        torch.save(
            {
                "epoch": best_epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": backbone.config.to_dict(),
                "best_r": float(best_r),
            },
            "/scem/work/zhangyt/project/ai/dnabert2/polya_single_read/model/best_by_r.pt",
        )
        log.info(
            f"new best r={best_r:.3f} @ epoch {best_epoch}, saved to checkpoints/best_by_r.pt"
        )
        # 保存当轮验证集预测
        df_pred = pd.DataFrame(
            {
                "y_true": gts.cpu().numpy().squeeze(),
                "y_pred": preds.cpu().numpy().squeeze(),
            }
        )
        df_pred.to_csv(os.path.join(log_dir, "best_val_pred.csv"), index=False)
    else:
        bad_count += 1
        if bad_count >= patience:
            log.info("Early stop at epoch %d", epoch + 1)
            break
# ============= 训练结束后画图 =============
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

epochs = range(1, len(tr_losses) + 1)

plt.figure()
plt.plot(epochs, tr_losses, label="train_loss")
plt.plot(epochs, val_losses, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(
    "/scem/work/zhangyt/project/ai/dnabert2/polya_single_read/loss_plot/loss_curve.png"
)

plt.figure()
plt.plot(epochs, tr_rs, label="train_r")
plt.plot(epochs, val_rs, label="val_pearson_r")
plt.xlabel("Epoch")
plt.ylabel("Pearson r")
plt.legend()
plt.tight_layout()
plt.savefig(
    "/scem/work/zhangyt/project/ai/dnabert2/polya_single_read/loss_plot/r_curve.png"
)
scatter_plot(best_val_df['y_true'],best_val_df['y_pred'],'/scem/work/zhangyt/project/ai/dnabert2/polya_single_read/loss_plot/best_pred_scatter.png')
log.info(f"Best r={best_r:.3f} at epoch {best_epoch}")
best_val_df.to_csv("/scem/work/zhangyt/project/ai/dnabert2/polya_single_read/result/val_predictions_best_epoch.tsv", sep="\t", index=False)
