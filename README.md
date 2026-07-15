# Probability Calibration in Imbalanced Fraud Detection

A small, self-contained project on **probability calibration in imbalanced
binary classification**, using credit-card fraud detection as the running
example: a model can separate fraud from genuine transactions very well
(high AUC-ROC / AUC-PR) while its output scores are still poor estimates of
the true probability of fraud. That distinction — discrimination vs.
calibration — is the practical motivation for this repository, and for my
interest in **differentiable calibration objectives that can be trained
directly into a classifier**, rather than bolted on afterwards.

The project builds up in four steps:

1. **Diagnose** the problem: train a standard classifier and show it is
   well-discriminating but poorly calibrated (reliability diagram, Expected
   Calibration Error, and a top-decile calibration error focused on the
   high-score region where investigation decisions actually happen).
2. **Fix it after training**, with the two classic post-hoc methods: Platt
   scaling and isotonic regression.
3. **Fix a specific, well-known failure mode**: training on an
   undersampled, artificially rebalanced dataset shifts predicted
   probabilities away from the true class prior. This is corrected with the
   closed-form prior-correction formula from Dal Pozzolo, Caelen, Johnson &
   Bontempi (2015).
4. **Fix it during training instead**: a small PyTorch classifier is
   trained with an added *differentiable* soft-ECE loss, jointly with the
   usual cross-entropy loss, and compared to a plain cross-entropy baseline
   under the same class-imbalance shift as in step 3.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(Worldline / ULB Machine Learning Group): 284,807 European card transactions
over two days, 492 of which are fraudulent (0.172%). Features `V1`-`V28` are
the result of a PCA transformation applied to the original (undisclosed)
attributes for confidentiality; `Time` and `Amount` are kept as-is.

The raw CSV is not included in this repository (file size and licensing).
`src/data.py` will, in order:

1. look for a local copy at `data/creditcard.csv` (download it from Kaggle
   and place it there for the exact canonical file), then
2. fall back to fetching the same dataset from
   [OpenML](https://www.openml.org/search?type=data&id=1597) (`data_id=1597`),
   which requires no account or API key, then
3. fall back to a synthetic dataset with a matching imbalance ratio, so the
   pipeline always runs end to end even without internet access.

Results in `results/` were produced with the real dataset (see
`results/metrics_summary.json`, field `data_source`).

## Method notes

- **Baseline model**: a `RandomForestClassifier`, trained without any
  class-balancing, on purpose — it is a good example of a model that
  discriminates well but is not a probability estimator out of the box.
- **Calibration error metrics**: alongside the standard Expected Calibration
  Error, the pipeline reports a *top-decile* calibration error, restricted
  to the highest 10% of predicted scores. In an imbalanced setting the
  standard ECE is dominated by the very large number of low-score genuine
  transactions and can look small even when the high scores that actually
  drive investigation are unreliable — the practical concern behind
  Guilbert, Caelen, Chirita & Saerens (2024).
- **Undersampling correction**: `correct_undersampled_probabilities` in
  `src/calibrate.py` implements the closed-form correction for training on
  an undersampled majority class (Elkan, 2001; used for fraud detection in
  Dal Pozzolo, Caelen, Johnson & Bontempi, 2015).
- **Differentiable calibration loss**: `soft_ece_loss` in
  `src/differentiable_calibration.py` is a soft-binned relaxation of ECE —
  each example gets a smooth, Gaussian-kernel membership across probability
  bins instead of being hard-assigned to one, so the objective is
  differentiable and can be minimised jointly with cross-entropy. This is a
  simplified, from-scratch version of ideas from the differentiable /
  trainable calibration literature (e.g. Kumar et al., 2018; Karandikar et
  al., 2021), applied here to a binary setting as a first step; extending
  this kind of objective to the multi-class setting, where per-class
  behaviour can be hidden by aggregate calibration measures, is the
  question I want to pursue further.

## Results

All numbers below are from a real run on the full 284,807-transaction
dataset (see `results/metrics_summary.json`, field `data_source`).
Discrimination (AUC-ROC / AUC-PR) stays roughly stable across methods, while
calibration error (ECE, top-decile CE) responds very differently depending
on what caused the miscalibration in the first place.

| Method | AUC-ROC | AUC-PR | Brier | ECE | Top-decile CE |
|---|---|---|---|---|---|
| Baseline (uncalibrated) | 0.9704 | 0.8089 | 0.00051 | 0.00030 | 0.00084 |
| Platt scaling | 0.9716 | 0.8065 | 0.00053 | 0.00016 | 0.00246 |
| Isotonic regression | 0.9696 | 0.8051 | 0.00050 | 0.00015 | 0.00012 |
| Undersampled (raw) | 0.9701 | 0.6425 | 0.00461 | 0.03050 | 0.15865 |
| Undersampled (Elkan-corrected) | 0.9701 | 0.6425 | 0.00095 | 0.00075 | 0.00408 |
| MLP, cross-entropy only | 0.9705 | 0.6866 | 0.00213 | 0.01152 | 0.07320 |
| MLP, cross-entropy + soft-ECE | 0.9688 | 0.6901 | 0.00202 | 0.01119 | 0.07090 |

A few things worth noting rather than glossing over:

- The untouched baseline is already fairly well calibrated on average here
  (a Random Forest's predicted probabilities are themselves bagged empirical
  frequencies) -- the textbook "accurate but overconfident" failure is not
  automatic, it depends on the model and how it was trained. **Naive
  undersampling is where the real damage happens**: ECE jumps roughly
  40x and top-decile CE almost 200x, and the closed-form correction removes
  nearly all of it.
- Platt scaling actually *increases* top-decile calibration error here even
  though it improves overall ECE -- its parametric sigmoid shape does not
  fit this particular tail well, while the non-parametric isotonic
  regression improves both. A reminder that "apply a calibration method"
  is not a single well-defined fix.
- The differentiable soft-ECE loss gives a small, consistent improvement in
  calibration at essentially no cost to discrimination, but the effect is
  far more modest than the undersampling correction, and it was easy to make
  much worse: pushing its weight higher in early experiments collapsed
  discrimination entirely (a shortcut where the network learns to predict a
  near-constant, "trivially calibrated" score). Getting a differentiable
  calibration objective to behave well, and extending it to the multi-class
  setting, is precisely the open problem this repository stops short of.

### Baseline: good discrimination, already-tight calibration

![Baseline reliability diagram](results/01_baseline_reliability.png)

### Post-hoc calibration: Platt scaling vs. isotonic regression

![Calibration methods comparison](results/02_calibration_methods_comparison.png)

### Undersampling breaks calibration; the Elkan/Dal Pozzolo-Caelen correction fixes it

![Undersampling correction](results/03_undersampling_correction.png)

### A differentiable calibration loss, trained jointly with cross-entropy

![Differentiable calibration loss](results/04_differentiable_calibration_loss.png)

## Running it

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Optional: download `creditcard.csv` from Kaggle and place it in `data/` to
run on a local copy instead of fetching from OpenML.

## References

- Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015).
  *Calibrating Probability with Undersampling for Unbalanced
  Classification.* IEEE Symposium Series on Computational Intelligence.
- Guilbert, T., Caelen, O., Chirita, A., & Saerens, M. (2024). *Calibration
  methods in imbalanced binary classification.* Annals of Mathematics and
  Artificial Intelligence, 92(5), 1319-1352.
- Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning.* IJCAI.
- Kumar, A., Sarawagi, S., & Jain, U. (2018). *Trainable Calibration
  Measures for Neural Networks from Kernel Mean Embeddings.* ICML.
- Karandikar, A., et al. (2021). *Soft Calibration Objectives for Neural
  Networks.* NeurIPS.

## License

MIT (see [LICENSE](LICENSE)).
