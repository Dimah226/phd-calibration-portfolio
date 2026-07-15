# Results

Data source: `OpenML mirror (data_id=1597)`

| Method | AUC-ROC | AUC-PR | Brier | ECE | Top-decile CE |
|---|---|---|---|---|---|
| Baseline (uncalibrated) | 0.9704 | 0.8089 | 0.00051 | 0.00030 | 0.00084 |
| Platt scaling | 0.9716 | 0.8065 | 0.00053 | 0.00016 | 0.00246 |
| Isotonic regression | 0.9696 | 0.8051 | 0.00050 | 0.00015 | 0.00012 |
| Undersampled (raw) | 0.9701 | 0.6425 | 0.00461 | 0.03050 | 0.15865 |
| Undersampled (Elkan-corrected) | 0.9701 | 0.6425 | 0.00095 | 0.00075 | 0.00408 |
| MLP, cross-entropy only | 0.9705 | 0.6866 | 0.00213 | 0.01152 | 0.07320 |
| MLP, cross-entropy + soft-ECE | 0.9688 | 0.6901 | 0.00202 | 0.01119 | 0.07090 |
