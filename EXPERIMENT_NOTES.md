# Experiment notes

This file records how the project evolved. I keep it because the final table
alone hides most of the decisions and mistakes made along the way.

## Starting point

I initially expected the unmodified random forest to provide an obvious
example of an accurate but badly calibrated model. That did not happen. Its
global ECE was already small on this dataset. I kept the result instead of
changing models only to obtain the story I expected.

The clearest calibration failure appeared after majority-class
undersampling. Ranking performance remained useful, but the scores were on
the wrong probability scale because the training prior had changed.

## Metric choice

My first version reported only equal-width ECE and a top-decile gap. The top
decile was too broad for a fraud-review queue: it represented thousands of
mostly genuine transactions. I replaced it with a top-1% gap and added:

- log loss;
- Brier score;
- equal-width ECE;
- equal-mass adaptive ECE (ACE).

The metrics do not always agree. I now treat that disagreement as a result,
not as something to hide.

## Mistake in the first MLP comparison

Both MLPs were trained on undersampled data, but I originally evaluated their
raw probabilities on the original test distribution. Most of their apparent
miscalibration therefore came from the changed class prior. I corrected this
by applying the same analytical prior correction to both models before
comparing their losses.

## Soft-ECE attempts

The current soft-ECE term uses Gaussian membership around probability-bin
centres. A moderate weight gives a better top-1% calibration gap in this run,
but slightly worse global ECE/ACE and ROC-AUC. Larger weights tried during
development pushed the network towards nearly constant predictions and hurt
discrimination. I did not retain those runs in the headline table because I
had not yet built a systematic hyperparameter protocol.

## What I would do next

1. Repeat the experiment over several seeds and report uncertainty intervals.
2. Use a separate validation set for choosing the calibration-loss weight.
3. Add a multiclass public dataset and compare confidence, classwise, and
   marginal calibration.
4. Compare the soft-binning loss with a kernel-based calibration objective.
5. Study what happens under an explicit train/test prior shift rather than
   only random undersampling.
