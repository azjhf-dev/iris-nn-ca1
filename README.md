# CA1 - Neural network classifier on the iris dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/azjhf-dev/iris-nn-ca1/blob/main/iris_ca1.ipynb)

A small feedforward neural network that classifies iris flowers into three
species (setosa, versicolor, virginica) from four measurements
(sepal length, sepal width, petal length, petal width). 150 rows, balanced
classes - the classic beginner dataset.

Everything below is the actual workflow, in order, with the real results this
code produces. The notebook (`iris_ca1.ipynb`) and the script (`train.py`)
run the same pipeline.

## Workflow - the thought process

1. **Understand the data.** 150 rows, 4 numeric features, 3 classes with 50
   rows each (perfectly balanced, so plain accuracy is a fair metric).
   Small dataset - this drives every later decision.
2. **Explore first (EDA).** The pairplot (`figures/pairplot.png`) shows
   setosa is cleanly separable from the other two, while versicolor and
   virginica overlap slightly on every feature pair. Petal measurements
   separate the classes much better than sepal measurements. Conclusion:
   the problem is *near-linearly separable*, so a simple model should work.
3. **Preprocess.** Stratified 80/20 split (120 train+val / 30 test, seed 42)
   so each class keeps its 1/3 share. Standardise features (zero mean, unit
   variance) using statistics computed on the training data only.
4. **Architecture: one hidden layer.** Two reasons. (a) The universal
   approximation theorem says a single hidden layer with enough neurons can
   approximate any continuous function, so depth is not *required*.
   (b) With only 120 training rows, a deep network would just memorise the
   data. Near-linear separability (step 2) says we don't need depth anyway.
   Final shape: 4 inputs -> N hidden neurons (ReLU) -> 3 outputs (softmax),
   trained with Adam and categorical cross-entropy.
5. **How wide? Sweep 2, 4, 8 neurons with 5-fold cross-validation.**
   A single train/val split on 120 rows is too noisy to trust, so each
   candidate width is trained and validated 5 times on different folds and
   we compare mean validation accuracy.
6. **Pick the winner, retrain, evaluate once.** The winning width is
   retrained on all 120 train+val rows, then evaluated **once** on the
   held-out test set that no training or selection step ever touched.

## Results (real outputs of this code)

5-fold cross-validation, mean accuracy over folds:

| hidden neurons | fold accuracies | mean +/- std |
|---|---|---|
| 2 | 1.000, 0.958, 0.958, 0.958, 0.917 | 0.958 +/- 0.026 |
| 4 | 1.000, 0.958, 0.958, 0.958, 0.917 | 0.958 +/- 0.026 |
| **8 (winner)** | 1.000, 1.000, 0.958, 0.958, 0.917 | **0.967 +/- 0.031** |

**Final test accuracy: 0.967 (29 of 30 correct).**

What the results say:

- Even 2 hidden neurons reach ~96% CV accuracy - confirming the EDA
  conclusion that iris is near-linearly separable. 8 neurons wins, but by
  less than one percentage point; the returns from extra width are small.
- The training curves (`figures/training_curves.png`) show train and
  validation loss falling together and settling - no overfitting gap, so
  the model size is appropriate for the data size.
- The confusion matrix (`figures/confusion_matrix.png`) shows the single
  test error is one versicolor predicted as virginica - exactly the overlap
  region the pairplot showed in step 2.
- The decision boundary (`figures/decision_boundary.png`, plotted in
  petal length vs petal width using a 2-feature model of the same
  architecture) makes the learned regions visible: a clean cut around
  setosa and a diagonal boundary through the versicolor/virginica overlap.

## Files

- `iris_ca1.ipynb` - Colab-ready notebook, same pipeline with the reasoning
  written out between cells. Badge above opens it directly.
- `train.py` - the same pipeline as a script; writes `figures/*.png` and
  `results.json`.
- `results.json` - CV scores per width, winner, test metrics.
- `figures/` - pairplot (EDA), training curves, confusion matrix,
  decision boundary.
- `requirements.txt` - dependencies.

## Run it

```bash
pip install -r requirements.txt
python train.py
```

Deterministic by default (fixed seed, `TF_DETERMINISTIC_OPS=1`): rerunning
reproduces the numbers above.
