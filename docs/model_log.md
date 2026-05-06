# Model Experiment Log

## 2026-05-05 - Log Initialized
- Note: No prior experiment log was found in the workspace.
- Action: Initialized this log to track CNN experiments.

## 2026-05-05 - Baseline CNN (Notebook Original)
- Experiment date: 2026-05-05
- Model version: simple_cnn_v0
- Architecture summary: Conv(16)-ReLU-MaxPool -> Conv(32)-ReLU-MaxPool -> Conv(64)-ReLU-MaxPool -> Flatten -> Dense(128)-Dropout(0.3) -> Dense(num_classes)
- Class weight strategy: Inverse frequency on train split, normalized to mean 1.0
- Training setup: Adam lr=1e-3, epochs=20, best model by val macro F1 (via callback)
- Patient-level accuracy: 0.6111111111111112 Patient-level precision (macro): 0.34027777777777773 Patient-level recall (macro): 0.5462962962962963 Patient-level F1 (macro): 0.38725490196078427
- Main errors observed: Not recorded
- What improved compared to the previous model: N/A (baseline)
- What should change in the next model: Increase minority class emphasis (alpha > 1), consider BatchNorm, tune dropout and regularization

## 2026-05-05 - Improved CNN (Pending Run)
- Experiment date: 2026-05-05
- Model version: cnn_bn_v1
- Architecture summary: Conv(16)-BN-ReLU-MaxPool -> Conv(32)-BN-ReLU-MaxPool -> Conv(64)-BN-ReLU-MaxPool -> Flatten -> Dense(128)-Dropout(0.4) -> Dense(num_classes)
- Class weight strategy: Inverse frequency on train split with exponent alpha=1.5, normalized to mean 1.0
- Training setup: AdamW lr=1e-3, weight_decay=1e-4, epochs=25, best model by val macro F1
- Validation metrics: TBD after run
- Test metrics: TBD after run
- Main errors observed: TBD after run
- What improved compared to the previous model: TBD after run
- What should change in the next model: TBD after run
