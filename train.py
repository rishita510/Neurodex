"""
=============================================================
train_model.py — Train classifiers on features_all.csv
=============================================================
Input:  outputs/features_all.csv
Output: best_model.pkl, scaler.pkl
        outputs/classifier_results.csv
        outputs/confusion_matrices/
        outputs/roc_curves/

Usage:
  python train_model.py
=============================================================
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from sklearn.preprocessing   import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, roc_curve, auc)
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.ensemble        import (ExtraTreesClassifier, RandomForestClassifier,
                                     AdaBoostClassifier)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not installed — skipping")

os.makedirs('outputs/confusion_matrices', exist_ok=True)
os.makedirs('outputs/roc_curves',         exist_ok=True)

# ════════════════════════════════════════════════════════
# ✏️  CONFIGURE HERE
# ════════════════════════════════════════════════════════

TEST_SIZE   = 0.20   # ← change to 0.30 for 70/30 split
RANDOM_SEED = 42

# ════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════
GESTURE_NAMES = {
    0: 'Thumb', 1: 'Index', 2: 'Middle', 3: 'Ring', 4: 'Little'
}

# ════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════
def load_data():
    path = 'outputs/features_all.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ {path} not found! Run build_dataset.py first.")

    df = pd.read_csv(path)
    meta = ['subject_id', 'gesture_id', 'gesture_name', 'rep_num']

    X = df.drop(columns=meta).values.astype(np.float32)
    y = df['gesture_id'].values - 1  # 0-indexed

    print(f"  Total samples : {X.shape[0]}")
    print(f"  Features      : {X.shape[1]}")
    print(f"\n  Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"    {GESTURE_NAMES[cls]:10s} : {cnt} samples")

    return X, y

# ════════════════════════════════════════════════════════
# CLASSIFIERS
# ════════════════════════════════════════════════════════
def get_classifiers():
    clfs = {
        'SVM_Linear':     SVC(kernel='linear', C=1, probability=True, random_state=RANDOM_SEED),
        'SVM_RBF':        SVC(kernel='rbf', gamma='scale', C=1, probability=True, random_state=RANDOM_SEED),
        'ANN_15_tanh':    MLPClassifier(hidden_layer_sizes=(15,), activation='tanh', max_iter=1000, random_state=RANDOM_SEED),
        'ANN_15x8_tanh':  MLPClassifier(hidden_layer_sizes=(15,8), activation='tanh', max_iter=2000, random_state=RANDOM_SEED),
        'KNN_K1':         KNeighborsClassifier(n_neighbors=1, metric='euclidean'),
        'KNN_K10':        KNeighborsClassifier(n_neighbors=10, metric='euclidean'),
        'ExtraTrees_200': ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
        'ExtraTrees_400': ExtraTreesClassifier(n_estimators=400, random_state=RANDOM_SEED, n_jobs=-1),
        'RF_200':         RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
        'AdaBoost_100':   AdaBoostClassifier(n_estimators=100, random_state=RANDOM_SEED),
    }
    if HAS_XGB:
        clfs['XGBoost_100'] = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            objective='multi:softmax', num_class=5,
            random_state=RANDOM_SEED, eval_metric='mlogloss', verbosity=0)
        clfs['XGBoost_300'] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            objective='multi:softmax', num_class=5,
            random_state=RANDOM_SEED, eval_metric='mlogloss', verbosity=0)
    return clfs

# ════════════════════════════════════════════════════════
# CONFUSION MATRIX
# ════════════════════════════════════════════════════════
def plot_confusion_matrix(cm, name):
    classes = [GESTURE_NAMES[i] for i in range(5)]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(5), yticks=range(5),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='True',
           title=f'Confusion Matrix — {name}')
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i,j]}', ha='center', va='center',
                    color='white' if cm[i,j] > thresh else 'black', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'outputs/confusion_matrices/{name}.png', dpi=120)
    plt.close()

# ════════════════════════════════════════════════════════
# ROC CURVES
# ════════════════════════════════════════════════════════
def plot_roc(clf, X_test_scaled, y_test, name):
    try:
        y_bin  = label_binarize(y_test, classes=list(range(5)))
        y_prob = clf.predict_proba(X_test_scaled)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(f'ROC Curves — {name}', fontsize=13)
        for i, ax in enumerate(axes.flat):
            if i >= 5: ax.axis('off'); continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color='red', lw=2, label=f'AUC={roc_auc:.3f}')
            ax.plot([0,1],[0,1],'k--',lw=0.8)
            ax.set(xlim=[0,0.15], ylim=[0.4,1.02],
                   xlabel='1-Specificity', ylabel='Sensitivity',
                   title=f'{GESTURE_NAMES[i]}')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'outputs/roc_curves/{name}_ROC.png', dpi=120)
        plt.close()
    except Exception as e:
        print(f"    ROC plot failed: {e}")

# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  EMG Model Trainer                                   ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    print("📂 Loading features...")
    X, y = load_data()

    # Stratified split — equal class proportions in both train and test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y          # ← guarantees equal class split
    )

    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)
    X_full_s = scaler.transform(X)

    train_pct = int((1 - TEST_SIZE) * 100)
    test_pct  = int(TEST_SIZE * 100)
    print(f"\n  Split : {train_pct}/{test_pct} stratified")
    print(f"  Train : {len(X_tr)} samples")
    print(f"  Test  : {len(X_te)} samples")

    print(f"\n  Test class distribution:")
    unique, counts = np.unique(y_te, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"    {GESTURE_NAMES[cls]:10s} : {cnt} samples")

    print(f"\n{'─'*68}")
    print(f"{'Classifier':<25} {'Train%':>8} {'Test%':>8} {'CV Mean%':>10} {'CV Std':>8}")
    print(f"{'─'*68}")

    classifiers = get_classifiers()
    results     = []

    for name, clf in classifiers.items():
        clf.fit(X_tr_s, y_tr)

        train_acc = accuracy_score(y_tr, clf.predict(X_tr_s))
        test_acc  = accuracy_score(y_te, clf.predict(X_te_s))
        cv_sc     = cross_val_score(clf, X_full_s, y, cv=5, scoring='accuracy')

        print(f"{name:<25} {train_acc*100:>7.1f}% {test_acc*100:>7.1f}% "
              f"{cv_sc.mean()*100:>9.1f}% {cv_sc.std()*100:>7.1f}%")

        cm = confusion_matrix(y_te, clf.predict(X_te_s))
        plot_confusion_matrix(cm, name)
        plot_roc(clf, X_te_s, y_te, name)

        results.append({
            'classifier':     name,
            'train_accuracy': round(train_acc * 100, 2),
            'test_accuracy':  round(test_acc  * 100, 2),
            'cv_mean':        round(cv_sc.mean() * 100, 2),
            'cv_std':         round(cv_sc.std()  * 100, 2),
        })

    print(f"{'─'*68}")

    res_df = pd.DataFrame(results).sort_values('test_accuracy', ascending=False)
    res_df.to_csv('outputs/classifier_results.csv', index=False)

    print(f"\n{'═'*68}")
    print(f"TOP 5 CLASSIFIERS:")
    print(f"{'═'*68}")
    print(res_df.head(5).to_string(index=False))

    # ── Save best model ──
    best_name    = res_df.iloc[0]['classifier']
    best_clf     = get_classifiers()[best_name]
    final_scaler = StandardScaler()
    X_tr_s2      = final_scaler.fit_transform(X_tr)
    X_te_s2      = final_scaler.transform(X_te)
    best_clf.fit(X_tr_s2, y_tr)

    joblib.dump(best_clf,     'best_model.pkl')
    joblib.dump(final_scaler, 'scaler.pkl')

    print(f"\n🏆 Best : {best_name} ({res_df.iloc[0]['test_accuracy']}% test acc)")
    print(f"✅ Saved: best_model.pkl + scaler.pkl")
    print(f"\n📊 Classification report:")
    print(classification_report(y_te, best_clf.predict(X_te_s2),
          target_names=[GESTURE_NAMES[i] for i in range(5)]))
    print(f"\n👉 Next: python predictor.py")

if __name__ == "__main__":
    main()
