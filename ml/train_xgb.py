import sys
import os
import glob
import json
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

def load_and_prepare_data(data_dir="D:/synth_data/processed_slices", sample_frac=1.0):
    files = glob.glob(f"{data_dir}/*.npz")
    assert files, "Нет .npz файлов"
    
    X, y = zip(*[(d['vectors'], d['labels']) for d in (np.load(f) for f in files)])
    X, y = np.vstack(X), np.concatenate(y)
    
    if sample_frac < 1.0:
        X, _, y, _ = train_test_split(X, y, train_size=sample_frac, stratify=y, random_state=42)
    return X, y

def search():
    print("Запуск GridSearchCV на специальной выборке...")
    X, y = load_and_prepare_data()
    
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    
    np.random.seed(42)
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    half_pos = max(1, len(pos_idx) // 2)
    
    # Train set for GridSearch: Первая половина препятствий + 3x пустых
    pos_sampled_idx = pos_idx[:half_pos]
    n_neg_needed = min(len(pos_sampled_idx) * 3, len(neg_idx))
    neg_sampled_idx = neg_idx[:n_neg_needed]
    
    selected_idx = np.concatenate([pos_sampled_idx, neg_sampled_idx])
    np.random.shuffle(selected_idx)
    
    X_search, y_search = X[selected_idx], y[selected_idx]
    
    param_grid = {
        'max_depth': [2, 3, 4],
        'n_estimators': [50, 100],
        'scale_pos_weight': [sum(y_search==0)/max(1, sum(y_search==1))],
        'gamma': [0.1, 1.0, 5.0],
        'colsample_bytree': [0.6, 0.8]
    }
    
    grid = GridSearchCV(XGBClassifier(random_state=42), param_grid, scoring='precision', cv=3, n_jobs=-1)
    grid.fit(X_search, y_search)
    print(f"Лучшие параметры: {grid.best_params_}")
    
    import json
    with open("best_params.json", "w") as f:
        json.dump(grid.best_params_, f)
    print("Параметры сохранены в best_params.json")

def train():
    print("Обучение финальной модели...")
    X, y = load_and_prepare_data()
    
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    
    np.random.seed(42)
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    half_pos = max(1, len(pos_idx) // 2)
    
    # Test set: Вторая половина препятствий + 3x пустых (соотношение 1:3)
    pos_test_idx = pos_idx[half_pos:]
    n_neg_test = min(len(pos_test_idx) * 3, len(neg_idx))
    neg_test_idx = neg_idx[:n_neg_test]
    
    # Train set: Первая половина препятствий + ВСЕ оставшиеся пустые
    pos_train_idx = pos_idx[:half_pos]
    neg_train_idx = neg_idx[n_neg_test:]
    
    train_idx = np.concatenate([pos_train_idx, neg_train_idx])
    test_idx = np.concatenate([pos_test_idx, neg_test_idx])
    
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    params = {
        'max_depth': 3,
        'n_estimators': 100,
        'gamma': 1.0,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    import os, json
    if os.path.exists("best_params.json"):
        with open("best_params.json", "r") as f:
            params.update(json.load(f))
            
    # Устанавливаем scale_pos_weight для дисбаланса финальной обучающей выборки
    params['scale_pos_weight'] = sum(y_train==0) / max(1, sum(y_train==1))
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    print("--- Статистика на тестовой выборке (остальная половина препятствий + 3/4 пустых) ---")
    print(classification_report(y_test, model.predict(X_test)))
    model.save_model("xgb_obstacle.json")

if __name__ == "__main__":
    search()
    train()
