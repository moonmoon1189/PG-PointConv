import torch
from models.pg_pointconv import PGPointConv
from risk.dbscan_regions import RiskRegionBuilder
from risk.geometric_measure import GeometricMeasure
from risk.risk_target import compute_risk_target
from risk.priority_mapping import PriorityMapper


def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize complete pipeline
    model = PGPointConv(num_classes=5).to(device)
    model.eval()

    region_builder = RiskRegionBuilder(eps=0.1, min_samples=5)
    geo_measure = GeometricMeasure()
    mapper = PriorityMapper()

    # Mock single room point cloud
    mock_pts = torch.rand(1, 20000, 3).to(device) * 2.0

    with torch.no_grad():
        # Forward pass
        seg_logits, risk_pred, deep_xyz = model(mock_pts)
        seg_preds = torch.argmax(seg_logits, dim=-1).squeeze(0)  # [1280]

        # 1. DBSCAN regions based on predicted obstacle semantics
        regions, masks = region_builder.build(deep_xyz.squeeze(0), seg_preds)

        print(f"Detected {len(regions)} obstacle regions.")

        # 2. Extract Geometric Metrics & Compute Physics Risk Target
        if len(regions) > 0:
            delta_h, w = geo_measure.extract(regions, deep_xyz.squeeze(0))
            obs_mask = torch.ones(len(regions))
            r_phys = compute_risk_target(delta_h, w, obs_mask)

            # Aggregate risk predictions using masks
            risk_pred_flat = risk_pred.squeeze()  # [1280]
            region_risks = []

            for mask in masks:
                # Average risk prediction within the region cluster
                mean_risk = risk_pred_flat[mask].mean()
                region_risks.append(mean_risk)

            region_risks = torch.tensor(region_risks)

            # 3. Priority Mapping
            scaled_scores, priorities = mapper.map_priority(region_risks)

            for i in range(len(regions)):
                print(f"Region {i + 1}:")
                print(f"  Delta H: {delta_h[i]:.3f}m | Effective Width: {w[i]:.3f}m")
                print(f"  Physics Target: {r_phys[i]:.3f}")
                print(f"  Predicted Model Score (0-10): {scaled_scores[i]:.2f}")
                print(f"  Action Priority: {priorities[i]}")


if __name__ == "__main__":
    main()