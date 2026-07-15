# Results

Data source: `OpenML mirror (data_id=1597)`

| Method | AUC-ROC | AUC-PR | Log loss | Brier | ECE | ACE | Top-1% CE |
|---|---|---|---|---|---|---|---|
| Baseline (uncalibrated) | 0.9704 | 0.8089 | 0.00333 | 0.00051 | 0.00030 | 0.00025 | 0.00360 |
| Platt scaling | 0.9716 | 0.8065 | 0.00356 | 0.00053 | 0.00016 | 0.00048 | 0.01609 |
| Isotonic regression | 0.9696 | 0.8051 | 0.00321 | 0.00050 | 0.00015 | 0.00019 | 0.00411 |
| Undersampled (raw) | 0.9701 | 0.6425 | 0.03898 | 0.00461 | 0.03050 | 0.03050 | 0.38708 |
| Undersampled (Elkan-corrected) | 0.9701 | 0.6425 | 0.00780 | 0.00095 | 0.00075 | 0.00050 | 0.04041 |
| MLP, cross-entropy only | 0.9705 | 0.6866 | 0.00486 | 0.00070 | 0.00043 | 0.00010 | 0.00520 |
| MLP, cross-entropy + soft-ECE | 0.9688 | 0.6901 | 0.00485 | 0.00069 | 0.00050 | 0.00015 | 0.00139 |
