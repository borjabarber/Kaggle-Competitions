"""
Spaceship Titanic - CatBoost with Extended Seed Averaging
10 seeds + threshold optimization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

results = []

def main():
    global results
    
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    results.append(f"Train: {train.shape}, Test: {test.shape}")
    
    test_ids = test['PassengerId'].copy()
    
    y = train["Transported"].astype(int).values
    train["is_train"] = 1
    test["is_train"] = 0
    full = pd.concat([train, test], ignore_index=True)
    
    # Feature Engineering
    pid_split = full["PassengerId"].str.split("_", expand=True)
    full["Group"] = pid_split[0].astype(int)
    full["GroupSize"] = full.groupby("Group")["PassengerId"].transform("count")
    full["IsAlone"] = (full["GroupSize"] == 1).astype(int)
    
    cabin_split = full["Cabin"].str.split("/", expand=True)
    full["Deck"] = cabin_split[0]
    full["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    full["Side"] = cabin_split[2]
    
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    full["TotalSpend"] = full[spend_cols].sum(axis=1)
    full["HasSpend"] = (full["TotalSpend"] > 0).astype(int)
    full["NumAmenitiesUsed"] = full[spend_cols].gt(0).sum(axis=1)
    
    # Group-aware Imputation
    group_mode_cols = ["HomePlanet", "Destination", "Deck", "Side", "CryoSleep", "VIP"]
    for col in group_mode_cols:
        full[col] = full.groupby("Group")[col].transform(
            lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
        )
    
    for col in ["Age", "CabinNum"]:
        full[col] = full.groupby("Group")[col].transform(lambda x: x.fillna(x.median()))
        full[col] = full[col].fillna(full[col].median())
    
    for col in spend_cols:
        full[col] = full[col].fillna(0)
    
    cs_na = full["CryoSleep"].isna()
    full.loc[cs_na & (full["TotalSpend"] == 0), "CryoSleep"] = True
    full.loc[cs_na & (full["TotalSpend"] > 0), "CryoSleep"] = False
    full["CryoSleep"] = full["CryoSleep"].fillna(False)
    full["VIP"] = full["VIP"].fillna(False)
    
    for col in ["HomePlanet", "Destination", "Deck", "Side"]:
        full[col] = full[col].fillna(full[col].mode()[0])
    
    # Derived Features
    full["IsChild"] = (full["Age"] < 18).astype(int)
    full["AgeBin"] = pd.cut(full["Age"], bins=[-1, 12, 18, 25, 40, 60, 200], labels=False).astype(int)
    full["CabinNumEven"] = (full["CabinNum"] % 2 == 0).astype(int)
    full["LogTotalSpend"] = np.log1p(full["TotalSpend"])
    
    for col in spend_cols:
        full["Log_" + col] = np.log1p(full[col])
    
    # Encoding
    full_features = full.drop(columns=["Cabin", "Name", "Group"])
    
    cat_cols = full_features.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        full_features[col] = LabelEncoder().fit_transform(full_features[col].astype(str))
    
    bool_cols = full_features.select_dtypes(include="bool").columns.tolist()
    for col in bool_cols:
        full_features[col] = full_features[col].astype(int)
    
    feature_cols = [c for c in full_features.columns if c not in ["Transported", "is_train", "PassengerId"]]
    train_processed = full_features[full_features["is_train"] == 1].copy()
    test_processed = full_features[full_features["is_train"] == 0].copy()
    
    X = train_processed[feature_cols].astype(float)
    X_test = test_processed[feature_cols].astype(float)
    
    results.append(f"Features: {len(feature_cols)}")
    
    # Extended seed averaging - 10 seeds
    seeds = [42, 123, 456, 789, 2024, 1337, 7777, 8888, 9999, 13]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_test_proba = []
    all_oof_proba = []
    
    results.append(f"--- Seed Averaging ({len(seeds)} seeds) ---")
    
    for seed in seeds:
        oof_proba = np.zeros(len(X))
        test_proba = np.zeros(len(X_test))
        
        for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]
            
            model = CatBoostClassifier(
                n_estimators=500,
                learning_rate=0.03,
                depth=5,
                subsample=0.6,
                l2_leaf_reg=3.0,
                min_data_in_leaf=40,
                random_state=seed,
                verbose=0
            )
            model.fit(X_tr, y_tr)
            oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]
            test_proba += model.predict_proba(X_test)[:, 1] / cv.n_splits
        
        cv_acc = accuracy_score(y, (oof_proba >= 0.5).astype(int))
        results.append(f"  Seed {seed}: CV={cv_acc:.4f}")
        
        all_oof_proba.append(oof_proba)
        all_test_proba.append(test_proba)
    
    # Average predictions
    avg_oof = np.mean(all_oof_proba, axis=0)
    avg_test = np.mean(all_test_proba, axis=0)
    
    avg_cv = accuracy_score(y, (avg_oof >= 0.5).astype(int))
    results.append(f"Averaged CV (thr=0.5): {avg_cv:.4f}")
    
    # Threshold optimization on OOF
    best_thr, best_acc = 0.5, avg_cv
    for thr in np.arange(0.45, 0.55, 0.005):
        acc = accuracy_score(y, (avg_oof >= thr).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    
    results.append(f"Best threshold: {best_thr:.3f} with CV={best_acc:.4f} ({best_acc*100:.2f}%)")
    
    # Submissions with different thresholds
    submission_05 = pd.DataFrame({
        "PassengerId": test_ids,
        "Transported": (avg_test >= 0.5).astype(bool)
    })
    submission_05.to_csv("submission_thr05.csv", index=False)
    
    submission_best = pd.DataFrame({
        "PassengerId": test_ids,
        "Transported": (avg_test >= best_thr).astype(bool)
    })
    submission_best.to_csv("submission.csv", index=False)
    
    results.append(f"Saved submissions with thr=0.5 and thr={best_thr:.3f}")
    
    # Write and print results
    with open('results.txt', 'w', encoding='utf-8') as f:
        for r in results:
            f.write(r + '\n')
    
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
