import io
import os
import base64
import traceback
import numpy as np
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB

STATE = {
    "data": None,
    "label_encoders": {},
    "target_col": None,
    "X_columns": None,
    "dt_model": None,
    "nb_model": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None
}

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return "data:image/png;base64," + img_b64

def create_confusion_matrix_plot(cm, title, cmap='Blues', class_names=None):
    """Buat plot confusion matrix dengan TP dan TN di bawah judul, di atas heatmap"""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Buat heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                cbar_kws={'shrink': 0.8}, 
                annot_kws={'size': 12, 'weight': 'bold'},
                linewidths=0.5, linecolor='gray')
    
    # Set judul dengan padding untuk memberi ruang label
    ax.set_title(title, fontsize=14, fontweight='bold', pad=50)
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    
    # Set class names jika provided
    if class_names is not None and len(class_names) == cm.shape[0]:
        ax.set_xticklabels(class_names, rotation=0, fontsize=10)
        ax.set_yticklabels(class_names, rotation=0, fontsize=10)
    
    # Tambahkan label TP dan TN di bawah judul dan di atas heatmap untuk binary classification
    if cm.shape == (2, 2):
        # Konfigurasi untuk bbox
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor='lightgray', 
                         alpha=0.9, edgecolor='black', linewidth=1)
        
        # Posisi y untuk label (di antara judul dan heatmap)
        label_y_position = -0.2
        
        # TP - di kiri atas (sesuai dengan posisi sel TP di heatmap)
        ax.text(0.5, label_y_position, 'True Positive\n(TP)', color='darkgreen', 
                fontsize=10, fontweight='bold', ha='center', va='center',
                bbox=bbox_props)
        
        # TN - di kanan atas (sesuai dengan posisi sel TN di heatmap)
        ax.text(1.5, label_y_position, 'True Negative\n(TN)', color='darkblue', 
                fontsize=10, fontweight='bold', ha='center', va='center',
                bbox=bbox_props)
        
        # FP dan FN tetap di bawah heatmap
        ax.text(0.5, 2.4, 'False Positive\n(FP)', color='darkred', 
                fontsize=10, fontweight='bold', ha='center', va='center',
                bbox=bbox_props)
        
        ax.text(1.5, 2.4, 'False Negative\n(FN)', color='darkorange', 
                fontsize=10, fontweight='bold', ha='center', va='center',
                bbox=bbox_props)
    
    plt.tight_layout()
    return fig

def create_confusion_matrix_plot_multiclass(cm, title, cmap='Blues', class_names=None):
    """Buat plot confusion matrix untuk multi-class classification"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Buat heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                cbar_kws={'shrink': 0.8}, 
                annot_kws={'size': 9},
                linewidths=0.5, linecolor='gray')
    
    # Set judul dan label
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    
    # Set class names
    if class_names is not None and len(class_names) == cm.shape[0]:
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(class_names, rotation=0, fontsize=9)
    
    # Highlight diagonal (correct predictions) dengan border hijau
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                # Correct predictions (diagonal) - beri border hijau
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                         edgecolor='green', linewidth=2))
    
    plt.tight_layout()
    return fig

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "File tidak ditemukan"}), 400

        df = pd.read_csv(file)
        STATE["data"] = df.copy()

        # cari kolom target otomatis
        target_col = None
        for col in df.columns:
            if col.lower() in ['label', 'keterangan', 'status', 'target', 'class', 'hasil']:
                target_col = col
                break
        if target_col is None:
            # Jika tidak ditemukan, gunakan kolom terakhir
            target_col = df.columns[-1]
            print(f"Kolom target tidak ditemukan, menggunakan kolom terakhir: {target_col}")

        STATE["target_col"] = target_col
        print(f"Kolom target: {target_col}")

        # Label encode
        encoders = {}
        df_enc = df.copy()
        for col in df_enc.columns:
            if df_enc[col].dtype == 'object' or df_enc[col].dtype.name == 'category':
                le = LabelEncoder()
                df_enc[col] = le.fit_transform(df_enc[col].fillna("NA").astype(str))
                encoders[col] = le
        STATE["label_encoders"] = encoders

        X = df_enc.drop(columns=[target_col])
        y = df_enc[target_col]
        STATE["X_columns"] = list(X.columns)

        print(f"Features: {STATE['X_columns']}")
        print(f"Target classes: {y.unique()}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        STATE.update({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test})

        # Model Decision Tree
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        STATE["dt_model"] = dt

        # Model Naive Bayes
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        STATE["nb_model"] = nb

        # Prediksi
        y_pred_dt = dt.predict(X_test)
        y_pred_nb = nb.predict(X_test)

        acc_dt = accuracy_score(y_test, y_pred_dt)
        acc_nb = accuracy_score(y_test, y_pred_nb)

        f1_dt = f1_score(y_test, y_pred_dt, average='weighted', zero_division=0)
        f1_nb = f1_score(y_test, y_pred_nb, average='weighted', zero_division=0)

        # Confusion Matrix
        cm_dt = confusion_matrix(y_test, y_pred_dt)
        cm_nb = confusion_matrix(y_test, y_pred_nb)

        # Dapatkan nama kelas untuk label
        class_names = [str(cls) for cls in sorted(y.unique())]
        
        # Confusion Matrix plot dengan label TP, FP, TN, FN
        if cm_dt.shape == (2, 2):  # Binary classification
            fig1 = create_confusion_matrix_plot(cm_dt, "Confusion Matrix - Decision Tree", 'Blues', class_names)
            fig2 = create_confusion_matrix_plot(cm_nb, "Confusion Matrix - Naive Bayes", 'Oranges', class_names)
        else:  # Multi-class classification
            fig1 = create_confusion_matrix_plot_multiclass(cm_dt, "Confusion Matrix - Decision Tree", 'Blues', class_names)
            fig2 = create_confusion_matrix_plot_multiclass(cm_nb, "Confusion Matrix - Naive Bayes", 'Oranges', class_names)
        
        fig_dt_cm = fig_to_base64(fig1)
        fig_nb_cm = fig_to_base64(fig2)

        # Struktur Decision Tree
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        plot_tree(dt, filled=True, rounded=True, feature_names=list(X.columns),
                  class_names=[str(c) for c in sorted(y.unique())], fontsize=8, ax=ax3)
        ax3.set_title("Decision Tree Structure")
        fig_tree = fig_to_base64(fig3)

        # Classification report sebagai dictionary
        report_dt_dict = classification_report(y_test, y_pred_dt, output_dict=True, zero_division=0)
        report_nb_dict = classification_report(y_test, y_pred_nb, output_dict=True, zero_division=0)
        
        print("Decision Tree Report keys:", report_dt_dict.keys())
        print("Naive Bayes Report keys:", report_nb_dict.keys())
        
        # Format detailed metrics untuk frontend
        detailed_metrics = {
            "decision_tree": {
                "accuracy": report_dt_dict.get('accuracy', 0),
                "classes": [],
                "macro_avg": report_dt_dict.get('macro avg', {}),
                "weighted_avg": report_dt_dict.get('weighted avg', {})
            },
            "naive_bayes": {
                "accuracy": report_nb_dict.get('accuracy', 0),
                "classes": [],
                "macro_avg": report_nb_dict.get('macro avg', {}),
                "weighted_avg": report_nb_dict.get('weighted avg', {})
            }
        }
        
        # Process class-wise metrics untuk Decision Tree
        for class_name, metrics in report_dt_dict.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(metrics, dict):
                detailed_metrics["decision_tree"]["classes"].append({
                    "name": str(class_name),
                    "precision": metrics.get('precision', 0),
                    "recall": metrics.get('recall', 0),
                    "f1_score": metrics.get('f1-score', 0),
                    "support": int(metrics.get('support', 0))
                })
        
        # Process class-wise metrics untuk Naive Bayes
        for class_name, metrics in report_nb_dict.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(metrics, dict):
                detailed_metrics["naive_bayes"]["classes"].append({
                    "name": str(class_name),
                    "precision": metrics.get('precision', 0),
                    "recall": metrics.get('recall', 0),
                    "f1_score": metrics.get('f1-score', 0),
                    "support": int(metrics.get('support', 0))
                })

        print(f"Decision Tree classes count: {len(detailed_metrics['decision_tree']['classes'])}")
        print(f"Naive Bayes classes count: {len(detailed_metrics['naive_bayes']['classes'])}")

        sample_html = df.head(10).to_html(classes="table table-striped", index=False)

        response_data = {
            "message": f"Berhasil memproses file! Dataset: {len(df)} sampel, {len(X.columns)} fitur",
            "sample_table": sample_html,
            "metrics": {
                "decision_tree": {
                    "accuracy": round(float(acc_dt), 4), 
                    "f1": round(float(f1_dt), 4)
                },
                "naive_bayes": {
                    "accuracy": round(float(acc_nb), 4), 
                    "f1": round(float(f1_nb), 4)
                }
            },
            "detailed_metrics": detailed_metrics,
            "images": {
                "dt_confusion": fig_dt_cm,
                "nb_confusion": fig_nb_cm,
                "tree_plot": fig_tree
            },
            "columns": STATE["X_columns"],
            "target_column": target_col,
            "is_binary_classification": cm_dt.shape == (2, 2)  # Tambahkan info jenis klasifikasi
        }

        print("Response data keys:", response_data.keys())
        print("Detailed metrics included:", 'detailed_metrics' in response_data)
        print("Is binary classification:", cm_dt.shape == (2, 2))
        
        return jsonify(response_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.json or {}
        if STATE["dt_model"] is None or STATE["nb_model"] is None:
            return jsonify({"error": "Model belum dilatih. Upload dataset terlebih dahulu."}), 400

        cols = STATE["X_columns"]
        encoders = STATE["label_encoders"]
        row = {}

        for c in cols:
            val = payload.get(c, "")
            if c in encoders:
                le = encoders[c]
                try:
                    row[c] = le.transform([str(val)])[0]
                except Exception:
                    row[c] = 0
            else:
                row[c] = float(val) if val != "" else 0

        X_new = pd.DataFrame([row], columns=cols)
        pred_dt = STATE["dt_model"].predict(X_new)[0]
        pred_nb = STATE["nb_model"].predict(X_new)[0]

        target = STATE["target_col"]
        if target in encoders:
            pred_dt_label = encoders[target].inverse_transform([int(pred_dt)])[0]
            pred_nb_label = encoders[target].inverse_transform([int(pred_nb)])[0]
        else:
            pred_dt_label = str(pred_dt)
            pred_nb_label = str(pred_nb)

        return jsonify({
            "prediction": {"decision_tree": pred_dt_label, "naive_bayes": pred_nb_label}
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    app.run(host="0.0.0.0", port=port, debug=False)