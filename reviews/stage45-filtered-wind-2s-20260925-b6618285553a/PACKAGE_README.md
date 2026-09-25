# Stage 4/5: two-second full-organism wind screen

The complete 2-second trajectory is negative for source orientation: absolute bearing error 18.377° prepared, 20.689° before wind, 19.414° just after the 20-ms physical wind pulse, and 27.558° at 2 seconds. The source distance falls because tonic forward drive remains on. See `campaign/README.md`, `campaign/merged_full_01/ANALYSIS.json`, the raw `traces.npz`, plot, and MuJoCo MP4.

The two segments were joined only after exact cold-state restoration and filter-state recovery. The first segment ended at 1000 ms solely due to an unregistered snapshot name; the second completed steps 1001–2000. This capsule provides local receipts and hashes for that verification but omits the ~0.5-GB full checkpoints and transitive runtime dependencies, so it cannot independently reproduce the full simulation. The motor filter is an engineered prosthetic intervention, not a biological mechanism. Stage 4 and 5 remain open.

For independent review, inspect the code, `MANIFEST.json`, both RESULT files, RESTORATION, MOTOR_WIND_AUDIT, merged trace, and analyzer. The compressed event and Gaussian interval logs retain their original SHA-256 in the manifest. Please report concrete discrepancies without tuning this exposed life.
