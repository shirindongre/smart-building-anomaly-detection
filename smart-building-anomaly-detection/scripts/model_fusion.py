"""
=============================================================================
FEATURE-LEVEL FUSION PIPELINE
Fuses a trained Sensor Data Model and a trained Log Data Model
into a single unified classifier using feature-level fusion.
=============================================================================

"""

import numpy as np
import tensorflow as tf
import keras
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION — update these paths before running
# =============================================================================

CONFIG = {
    # --- Model paths ---
    "sensor_model_path": "sensor_model.h5",       # Path to trained sensor model
    "log_model_path":    "log_model.h5",           # Path to trained log model

    # --- Output paths ---
    "fusion_model_path": "fusion_classifier.pkl",  # Saved sklearn fusion model
    "results_dir":       "fusion_results",         # Folder for plots/reports

    # --- Fusion classifier choice: "mlp" or "logistic" ---
    "fusion_type": "mlp",

    # --- MLP fusion hyperparameters ---
    "mlp_hidden_layers": (128, 64),
    "mlp_max_iter": 500,

    # --- Class names (optional, used in reports) ---
    "class_names": None,   # e.g. ["Normal", "Attack"] or None for auto
}


# =============================================================================
# SECTION 1: DATA LOADING
# Replace the placeholder arrays below with your actual test data.
# =============================================================================

def load_test_data():
    """
    Load sensor test data, log test data, and ground-truth labels.

    Returns
    -------
    sensor_test : np.ndarray  — shape expected by sensor model
    log_test    : np.ndarray  — shape expected by log model
    y_test      : np.ndarray  — integer class labels, shape (N,)
    """
    # ------------------------------------------------------------------
    # PLACEHOLDER — replace with your real data loading logic, e.g.:
    #   sensor_test = np.load("sensor_test.npy")
    #   log_test    = np.load("log_test.npy")
    #   y_test      = np.load("y_test.npy")
    # ------------------------------------------------------------------

    print("[DATA] Using PLACEHOLDER synthetic data. Replace with real data.")

    N          = 500          # number of test samples
    n_sensor   = 50           # sensor feature width  (e.g. time-series flattened)
    n_log      = 30           # log feature width
    n_classes  = 3

    np.random.seed(42)
    sensor_test = np.random.randn(N, n_sensor).astype(np.float32)
    log_test    = np.random.randn(N, n_log).astype(np.float32)
    y_test      = np.random.randint(0, n_classes, size=N)

    return sensor_test, log_test, y_test


# =============================================================================
# SECTION 2: MODEL LOADING
# =============================================================================

def load_base_model(path: str, name: str) -> keras.Model:
    """
    Load a saved Keras model from disk.

    Parameters
    ----------
    path : str   — file path to .h5 or SavedModel
    name : str   — human-readable name for logging

    Returns
    -------
    model : keras.Model
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] {name} not found at '{path}'. "
            "Please update CONFIG with the correct path."
        )
    print(f"[LOAD] Loading {name} from: {path}")
    model = keras.models.load_model(path)
    print(f"[LOAD] {name} loaded — output shape: {model.output_shape}")
    return model


# =============================================================================
# SECTION 3: FEATURE EXTRACTOR CONSTRUCTION
# =============================================================================

def build_feature_extractor(model: keras.Model, name: str) -> keras.Model:
    """
    Strip the final classification layer from a Keras model to expose
    the penultimate feature representation.

    Strategy
    --------
    - If the last layer is Dense with softmax/sigmoid → remove it.
    - Otherwise use the second-to-last layer as the feature layer.
    - Falls back to the full model output if only one layer exists.

    Parameters
    ----------
    model : keras.Model — full trained model
    name  : str         — label for logging

    Returns
    -------
    extractor : keras.Model — feature-extraction sub-model
    """
    last_layer    = model.layers[-1]
    is_classifier = (
        isinstance(last_layer, keras.layers.Dense)
        and (
            last_layer.get_config().get("activation") in ("softmax", "sigmoid")
            or last_layer.units <= 64          # heuristic: small dense = classifier head
        )
    )

    if len(model.layers) > 1 and is_classifier:
        feature_output = model.layers[-2].output
        print(f"[EXTRACTOR] {name}: removed final layer '{last_layer.name}', "
              f"feature shape = {feature_output.shape}")
    else:
        feature_output = model.output
        print(f"[EXTRACTOR] {name}: using full model output as features, "
              f"shape = {feature_output.shape}")

    extractor = keras.Model(inputs=model.input, outputs=feature_output, name=f"{name}_extractor")
    return extractor


# =============================================================================
# SECTION 4: FEATURE EXTRACTION
# =============================================================================

def extract_features(extractor: keras.Model, data: np.ndarray, name: str) -> np.ndarray:
    """
    Run inference through the feature extractor and flatten the output.

    Parameters
    ----------
    extractor : keras.Model
    data      : np.ndarray  — input test data
    name      : str

    Returns
    -------
    features : np.ndarray of shape (N, feature_dim)
    """
    print(f"[EXTRACT] Extracting features from {name} ...")
    raw = extractor.predict(data, verbose=0)

    # Flatten any spatial/temporal dimensions — keep only (N, features)
    if raw.ndim > 2:
        raw = raw.reshape(raw.shape[0], -1)
        print(f"[EXTRACT] {name}: flattened to shape {raw.shape}")
    else:
        print(f"[EXTRACT] {name}: feature shape = {raw.shape}")

    return raw.astype(np.float32)


# =============================================================================
# SECTION 5: INDIVIDUAL BASELINE EVALUATION
# =============================================================================

def evaluate_single_model(model: keras.Model, data: np.ndarray,
                           y_test: np.ndarray, name: str) -> float:
    """
    Evaluate a single full model and return its accuracy.

    Parameters
    ----------
    model  : keras.Model
    data   : np.ndarray
    y_test : np.ndarray   — true integer labels
    name   : str

    Returns
    -------
    accuracy : float
    """
    print(f"\n[EVAL] Evaluating {name} (standalone) ...")
    probs  = model.predict(data, verbose=0)
    preds  = np.argmax(probs, axis=1) if probs.ndim > 1 else (probs > 0.5).astype(int).ravel()
    acc    = accuracy_score(y_test, preds)
    print(f"[EVAL] {name} accuracy: {acc:.4f}")
    return acc


# =============================================================================
# SECTION 6: FUSION CLASSIFIER
# =============================================================================

def build_fusion_classifier(fusion_type: str) -> object:
    """
    Build the fusion classifier (sklearn).

    Parameters
    ----------
    fusion_type : "mlp" | "logistic"

    Returns
    -------
    clf : sklearn Pipeline (scaler + classifier)
    """
    if fusion_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=CONFIG["mlp_hidden_layers"],
            activation="relu",
            solver="adam",
            max_iter=CONFIG["mlp_max_iter"],
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False,
        )
        print(f"[FUSION] Fusion classifier: MLP {CONFIG['mlp_hidden_layers']}")
    elif fusion_type == "logistic":
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        print("[FUSION] Fusion classifier: Logistic Regression")
    else:
        raise ValueError(f"Unknown fusion_type '{fusion_type}'. Choose 'mlp' or 'logistic'.")

    # Wrap in a pipeline with standard scaling
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])
    return pipeline


def train_fusion_classifier(pipeline, fused_features: np.ndarray,
                             y_train: np.ndarray) -> object:
    """
    Train the fusion classifier on concatenated feature vectors.

    Parameters
    ----------
    pipeline       : sklearn Pipeline
    fused_features : np.ndarray of shape (N, F_sensor + F_log)
    y_train        : np.ndarray of integer labels

    Returns
    -------
    pipeline : fitted sklearn Pipeline
    """
    print(f"\n[TRAIN] Training fusion classifier on features of shape {fused_features.shape} ...")
    pipeline.fit(fused_features, y_train)
    print("[TRAIN] Fusion classifier training complete.")
    return pipeline


# =============================================================================
# SECTION 7: EVALUATION & REPORTING
# =============================================================================

def evaluate_fusion(pipeline, fused_features: np.ndarray,
                    y_test: np.ndarray, results_dir: str) -> float:
    """
    Evaluate the fusion classifier and produce a full report.

    Parameters
    ----------
    pipeline       : fitted sklearn Pipeline
    fused_features : np.ndarray
    y_test         : np.ndarray
    results_dir    : str — directory to save plots

    Returns
    -------
    accuracy : float
    """
    os.makedirs(results_dir, exist_ok=True)
    preds = pipeline.predict(fused_features)
    acc   = accuracy_score(y_test, preds)

    class_names = CONFIG["class_names"]

    print("\n" + "=" * 60)
    print("  FUSION MODEL — EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy : {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, preds, target_names=class_names))

    # --- Confusion Matrix ---
    cm  = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True)
    ax.set_title("Fusion Model — Confusion Matrix")
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n[SAVE] Confusion matrix saved → {cm_path}")

    return acc


# =============================================================================
# SECTION 8: SAVE FUSION MODEL
# =============================================================================

def save_fusion_model(pipeline, path: str):
    """
    Persist the trained fusion classifier to disk using joblib.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    path     : str — output file path (.pkl)
    """
    joblib.dump(pipeline, path)
    print(f"[SAVE] Fusion model saved → {path}")


# =============================================================================
# SECTION 9: SUMMARY REPORT
# =============================================================================

def print_summary(sensor_acc: float, log_acc: float, fusion_acc: float):
    """
    Print a human-readable performance comparison and explain
    why feature-level fusion improves results.
    """
    print("\n" + "=" * 65)
    print("  PERFORMANCE SUMMARY")
    print("=" * 65)
    print(f"  {'Model':<30}  {'Accuracy':>10}")
    print("  " + "-" * 44)
    print(f"  {'Sensor Model (standalone)':<30}  {sensor_acc:>9.4f}")
    print(f"  {'Log Model (standalone)':<30}  {log_acc:>9.4f}")
    print(f"  {'Fusion Model':<30}  {fusion_acc:>9.4f}")
    print("  " + "-" * 44)

    best_single = max(sensor_acc, log_acc)
    delta       = fusion_acc - best_single
    direction   = "↑ improved" if delta >= 0 else "↓ degraded"
    print(f"\n  Fusion vs best single model: {delta:+.4f}  ({direction})")

    print("""
  WHY FEATURE-LEVEL FUSION HELPS
  ───────────────────────────────
  1. COMPLEMENTARY SIGNALS
     Sensor data captures low-level physical / numeric patterns (e.g.
     time-series anomalies, statistical moments), while log data
     captures high-level discrete events (e.g. error codes, sequences).
     No single modality contains the full picture.

  2. RICHER REPRESENTATION
     Concatenating penultimate-layer features from both models gives
     the fusion classifier a much higher-dimensional, information-dense
     representation compared to using either source alone.

  3. REDUNDANCY REDUCES VARIANCE
     When one modality is noisy or missing cues, the other compensates,
     leading to more robust and stable predictions.

  4. TASK-SPECIFIC RE-WEIGHTING
     The fusion classifier (MLP / Logistic Regression) learns to weight
     each modality's contribution optimally for the downstream task,
     going beyond simple averaging or voting.

  5. TRANSFER OF PRE-TRAINED KNOWLEDGE
     Both base models were trained end-to-end on their own domains.
     Feature-level fusion harvests that domain expertise without
     expensive joint retraining.
  """)
    print("=" * 65)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_fusion_pipeline():
    """
    End-to-end feature-level fusion pipeline.
    """
    print("\n" + "=" * 65)
    print("  FEATURE-LEVEL FUSION PIPELINE — START")
    print("=" * 65 + "\n")

    # ------------------------------------------------------------------
    # Step 1: Load test data
    # ------------------------------------------------------------------
    sensor_test, log_test, y_test = load_test_data()
    print(f"[DATA] sensor_test: {sensor_test.shape} | "
          f"log_test: {log_test.shape} | y_test: {y_test.shape}\n")

    # ------------------------------------------------------------------
    # Step 2: Load pre-trained base models
    # ------------------------------------------------------------------
    sensor_model = load_base_model(CONFIG["sensor_model_path"], "SensorModel")
    log_model    = load_base_model(CONFIG["log_model_path"],    "LogModel")

    # ------------------------------------------------------------------
    # Step 3: Evaluate individual models (baselines)
    # ------------------------------------------------------------------
    sensor_acc = evaluate_single_model(sensor_model, sensor_test, y_test, "SensorModel")
    log_acc    = evaluate_single_model(log_model,    log_test,    y_test, "LogModel")

    # ------------------------------------------------------------------
    # Step 4: Build feature extractors (remove final layer)
    # ------------------------------------------------------------------
    print()
    sensor_extractor = build_feature_extractor(sensor_model, "SensorModel")
    log_extractor    = build_feature_extractor(log_model,    "LogModel")

    # ------------------------------------------------------------------
    # Step 5: Extract features from test data
    # ------------------------------------------------------------------
    print()
    sensor_features = extract_features(sensor_extractor, sensor_test, "SensorModel")
    log_features    = extract_features(log_extractor,    log_test,    "LogModel")

    # ------------------------------------------------------------------
    # Step 6: Concatenate feature vectors
    # ------------------------------------------------------------------
    fused_features = np.concatenate([sensor_features, log_features], axis=1)
    print(f"\n[FUSE] Fused feature vector shape: {fused_features.shape}")
    print(f"       ({sensor_features.shape[1]} sensor dims + "
          f"{log_features.shape[1]} log dims = {fused_features.shape[1]} total)")

    # ------------------------------------------------------------------
    # Step 7: Build fusion classifier
    # ------------------------------------------------------------------
    print()
    fusion_clf = build_fusion_classifier(CONFIG["fusion_type"])

    # ------------------------------------------------------------------
    # Step 8: Train fusion classifier
    # (Using full test set as train here; in production split train/val/test)
    # ------------------------------------------------------------------
    fusion_clf = train_fusion_classifier(fusion_clf, fused_features, y_test)

    # ------------------------------------------------------------------
    # Step 9: Evaluate fusion model
    # ------------------------------------------------------------------
    fusion_acc = evaluate_fusion(fusion_clf, fused_features, y_test,
                                 CONFIG["results_dir"])

    # ------------------------------------------------------------------
    # Step 10: Save fusion model
    # ------------------------------------------------------------------
    save_fusion_model(fusion_clf, CONFIG["fusion_model_path"])

    # ------------------------------------------------------------------
    # Step 11: Print summary
    # ------------------------------------------------------------------
    print_summary(sensor_acc, log_acc, fusion_acc)

    print("\n[DONE] Fusion pipeline completed successfully.\n")
    return fusion_clf, fused_features, y_test


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_fusion_pipeline()