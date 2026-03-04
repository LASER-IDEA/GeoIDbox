import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import pinn_field_reconstruction as base


def train_no_era5_anchor(model, dataset, normalizer, device, epochs=200, batch_size=256):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=40, factor=0.5)

    print("Starting training (NO ERA5 anchor loss)...")
    start_time = time.time()
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            optimizer.zero_grad()

            s_x = batch['s_x'].to(device)
            s_p_true = batch['s_p'].to(device)
            s_t_true = batch['s_t'].to(device)

            s_preds = model(s_x)
            s_p_res_raw, s_t_res_raw = s_preds[:, 0], s_preds[:, 1]
            s_p_res, s_t_res = normalizer.scale_outputs(s_p_res_raw, s_t_res_raw)

            s_h_phys = normalizer.unnormalize_coords_h(s_x[:, 2])
            s_p_base, s_t_base = base.standard_atmosphere(s_h_phys)

            loss_obs_p = torch.mean(((s_p_base + s_p_res) - s_p_true) ** 2)
            loss_obs_t = torch.mean(((s_t_base + s_t_res) - s_t_true) ** 2)

            noise = (torch.rand_like(s_x) - 0.5) * 0.1
            x_colloc = (s_x + noise).clamp(-1, 1)
            x_colloc[:, 2:] = x_colloc[:, 2:].clamp(0, 1)
            loss_pde = base.physics_loss(model, x_colloc, normalizer)

            total_loss = (loss_obs_p) * 1e-5 + (loss_obs_t) * 1e-1 + loss_pde * 1.0
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        scheduler.step(avg_loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.5f} | OBS_P={loss_obs_p.item():.1f} | PDE={loss_pde.item():.3f}")

    print(f"Training finished in {time.time() - start_time:.1f}s")
    return model, losses


def evaluate_per_sensor(model, normalizer, sensor_df, device, samples_per_sensor=50):
    rows = []
    for uid in sorted(sensor_df['uid'].unique()):
        d = sensor_df[sensor_df['uid'] == uid]
        if len(d) == 0:
            continue
        sample = d.sample(min(samples_per_sensor, len(d)), random_state=42)
        errs = []
        for _, r in sample.iterrows():
            try:
                h = base.solve_height(model, normalizer, r['lat'], r['lon'], r['timestamp'], r['pressure'], device)
                errs.append(h - r['alt'])
            except Exception:
                continue
        if errs:
            errs = np.array(errs)
            rows.append({
                'uid': uid,
                'n': len(errs),
                'mae': float(np.mean(np.abs(errs))),
                'rmse': float(np.sqrt(np.mean(errs ** 2)))
            })
    return pd.DataFrame(rows)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    sensor_df, era5_df = base.load_data()
    if sensor_df is None:
        return

    s_temp = sensor_df[['lat', 'lon', 'alt', 'timestamp', 'pressure', 'temperature']]
    e_temp = era5_df[['lat', 'lon', 'static_height', 'timestamp', 'sp', 't2m']].copy()
    e_temp.columns = ['lat', 'lon', 'alt', 'timestamp', 'pressure', 'temperature']
    combined_temp = pd.concat([s_temp, e_temp], ignore_index=True)

    normalizer = base.DataNormalizer()
    normalizer.fit(combined_temp)

    dataset = base.CombinedDataset(sensor_df, era5_df, normalizer)

    model = base.WeatherField(num_freqs=10).to(device)
    model, _ = train_no_era5_anchor(model, dataset, normalizer, device, epochs=200, batch_size=256)

    out_model = 'pinn_model_no_era5_anchor.pth'
    torch.save(model.state_dict(), out_model)
    print(f"Saved {out_model}")

    print("\nEvaluating per-sensor height MAE...")
    res = evaluate_per_sensor(model, normalizer, sensor_df, device, samples_per_sensor=50)
    if len(res) == 0:
        print("No evaluation rows.")
        return

    print(res.sort_values('mae').to_string(index=False))
    print("\nSummary:")
    print(f"Mean MAE: {res['mae'].mean():.2f} m")
    print(f"Median MAE: {res['mae'].median():.2f} m")

    out_csv = 'ablation_no_era5_anchor_results.csv'
    res.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == '__main__':
    main()
