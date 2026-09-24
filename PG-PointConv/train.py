import torch
from torch.utils.data import DataLoader, Dataset
from models.pg_pointconv import PGPointConv
from models.losses import PGPointConvLoss
from risk.risk_target import compute_risk_target
import torch.optim as optim


class DummyCMSPD(Dataset):
    def __init__(self, num_samples=10):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 20k points as requested
        pts = torch.rand(20000, 3) * 2.0
        # Dummy labels for 1280 downsampled points in deep layer
        labels = torch.randint(0, 5, (1280,))
        expert = torch.rand(1280)

        # Mock physics parameters
        delta_h = torch.rand(1280) * 0.03
        w = torch.rand(1280) * 1.5
        obs_mask = torch.ones(1280)

        risk_phys = compute_risk_target(delta_h, w, obs_mask)
        return pts, labels, expert, risk_phys


def main():
    # 1. Reproducibility
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # 2. Dataset & Model
    dataset = DummyCMSPD()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch 1 for simple demo

    model = PGPointConv(num_classes=5).to(device)
    criterion = PGPointConvLoss().to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

    # 4. Training Loop (Demo for 2 epochs)
    model.train()
    for epoch in range(200):
        total_loss = 0
        for pts, labels, expert, risk_phys in dataloader:
            pts, labels, expert, risk_phys = pts.to(device), labels.to(device), expert.to(device), risk_phys.to(device)

            optimizer.zero_grad()
            seg_logits, risk_pred, _ = model(pts)

            loss, l_cls, l_exp, l_risk = criterion(seg_logits, labels, risk_pred, expert, risk_phys)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}/200 | Total Loss: {total_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    main()