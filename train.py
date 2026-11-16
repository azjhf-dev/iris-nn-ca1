"""
CA1 - Neural network classifier on the iris dataset.

Workflow (the thought process, made runnable):
  1. Understand the data: 150 rows, 4 features, 3 balanced classes.
  2. Explore: pairplot - setosa is cleanly separable, versicolor/virginica
     overlap a little, so the problem is near-linearly separable.
  3. Preprocess: stratified train/test split, standardise features.
  4. Architecture: ONE hidden layer. The universal approximation theorem says
     one hidden layer is enough in principle, and with only 150 rows a deep
     network would overfit. Near-linear separability says we don't need depth.
  5. Width: sweep the hidden layer size - 2, 4, 8 neurons - with 5-fold
     cross-validation (a single split is unreliable on 150 rows).
  6. Pick the winner, retrain on all training data, evaluate ONCE on the
     held-out test set.

Run:  pip install -r requirements.txt && python train.py
Outputs: figures/*.png and results.json
"""

import json
import os

SEED = 42
# deterministic training: must be set before tensorflow is imported
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

HIDDEN_NEURONS = [2, 4, 8]
EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 0.01
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)  # seeds python, numpy and tensorflow

# ---------------------------------------------------------------- 1. data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = [n.replace(" (cm)", "") for n in iris.feature_names]
target_names = list(iris.target_names)

df = pd.DataFrame(X, columns=feature_names)
df["species"] = [target_names[t] for t in y]
print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features, "
      f"classes = {target_names} (balanced: {np.bincount(y)})")

# ---------------------------------------------------------------- 2. explore
g = sns.pairplot(df, hue="species", diag_kind="hist", height=2.0)
g.figure.suptitle("Iris - pairwise feature scatter (EDA)", y=1.02)
g.figure.savefig(f"{FIG_DIR}/pairplot.png", dpi=150, bbox_inches="tight")
plt.close(g.figure)
print("saved figures/pairplot.png")

# ---------------------------------------------------------------- 3. preprocess
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
scaler = StandardScaler().fit(X_tv)
X_tv_s = scaler.transform(X_tv)
X_test_s = scaler.transform(X_test)
print(f"Split: {len(X_tv)} train+val / {len(X_test)} test (stratified, seed={SEED})")


# ---------------------------------------------------------------- 4. architecture
def build_model(n_hidden: int, n_features: int = 4) -> tf.keras.Model:
    """n_features -> n_hidden (relu) -> 3 (softmax)."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(n_hidden, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------- 5. sweep 2/4/8 with 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = {}
for n in HIDDEN_NEURONS:
    fold_acc = []
    for fold, (tr, va) in enumerate(skf.split(X_tv_s, y_tv)):
        model = build_model(n)
        model.fit(X_tv_s[tr], y_tv[tr], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        acc = model.evaluate(X_tv_s[va], y_tv[va], verbose=0)[1]
        fold_acc.append(acc)
    cv_results[n] = {
        "fold_accuracies": [round(a, 4) for a in fold_acc],
        "mean": round(float(np.mean(fold_acc)), 4),
        "std": round(float(np.std(fold_acc)), 4),
    }
    print(f"hidden={n}: folds={[round(a, 4) for a in fold_acc]} "
          f"mean={np.mean(fold_acc):.4f} +/- {np.std(fold_acc):.4f}")

# winner: highest mean CV accuracy, smallest network on a tie
best_n = min(HIDDEN_NEURONS, key=lambda n: (-cv_results[n]["mean"], n))
print(f"Winner: {best_n} hidden neurons")

# ------------------------------------------------- 6. final model, one test eval
final = build_model(best_n)
history = final.fit(
    X_tv_s, y_tv, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=0
)
test_pred = np.argmax(final.predict(X_test_s, verbose=0), axis=1)
test_acc = accuracy_score(y_test, test_pred)
report = classification_report(y_test, test_pred, target_names=target_names, output_dict=True)
print(f"Test accuracy (evaluated once): {test_acc:.4f}")

# loss / accuracy curves
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["loss"], label="train")
ax[0].plot(history.history["val_loss"], label="validation")
ax[0].set_title(f"Loss ({best_n} hidden neurons)")
ax[0].set_xlabel("epoch")
ax[0].legend()
ax[1].plot(history.history["accuracy"], label="train")
ax[1].plot(history.history["val_accuracy"], label="validation")
ax[1].set_title(f"Accuracy ({best_n} hidden neurons)")
ax[1].set_xlabel("epoch")
ax[1].legend()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/training_curves.png", dpi=150)
plt.close(fig)
print("saved figures/training_curves.png")

# confusion matrix on the test set
cm = confusion_matrix(y_test, test_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, ax=ax)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title(f"Confusion matrix - test set ({best_n} hidden neurons)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=150)
plt.close(fig)
print("saved figures/confusion_matrix.png")

# ------------------------------------------- decision boundary (petal features)
# A 2D boundary plot needs a 2-feature model: petal length vs petal width,
# the two most discriminative features. Same architecture and winner width.
pet = [2, 3]
scaler2 = StandardScaler().fit(X_tv[:, pet])
bmodel = build_model(best_n, n_features=2)
bmodel.fit(scaler2.transform(X_tv[:, pet]), y_tv, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

x0 = np.linspace(X[:, pet[0]].min() - 0.5, X[:, pet[0]].max() + 0.5, 300)
x1 = np.linspace(X[:, pet[1]].min() - 0.5, X[:, pet[1]].max() + 0.5, 300)
xx, yy = np.meshgrid(x0, x1)
grid = scaler2.transform(np.c_[xx.ravel(), yy.ravel()])
zz = np.argmax(bmodel.predict(grid, verbose=0), axis=1).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.contourf(xx, yy, zz, alpha=0.3, levels=2, cmap="Set2")
for t, name in enumerate(target_names):
    m = y == t
    ax.scatter(X[m, pet[0]], X[m, pet[1]], label=name, edgecolors="k", s=35)
ax.set_xlabel(feature_names[pet[0]])
ax.set_ylabel(feature_names[pet[1]])
ax.set_title(f"Decision boundary - {feature_names[pet[0]]} vs {feature_names[pet[1]]} "
             f"({best_n} hidden neurons)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/decision_boundary.png", dpi=150)
plt.close(fig)
print("saved figures/decision_boundary.png")

# ---------------------------------------------------------------- results
results = {
    "seed": SEED,
    "epochs": EPOCHS,
    "hidden_neurons_swept": HIDDEN_NEURONS,
    "cv": {str(k): v for k, v in cv_results.items()},
    "winner_hidden_neurons": best_n,
    "test_accuracy": round(float(test_acc), 4),
    "test_report": report,
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved results.json")
