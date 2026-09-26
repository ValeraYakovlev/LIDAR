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
    print("Запуск GridSearchCV на 10% данных...")
    X, y = load_and_prepare_data(sample_frac=0.1)
    
    param_grid = {
        'max_depth': [2, 3, 4],
        'n_estimators': [50, 100],
        'scale_pos_weight': [sum(y==0)/max(1, sum(y==1))],
        'gamma': [0.1, 1.0, 5.0],
        'colsample_bytree': [0.6, 0.8]
    }
    
    grid = GridSearchCV(XGBClassifier(random_state=42), param_grid, scoring='f1', cv=3, n_jobs=-1)
    grid.fit(X, y)
    print(f"Лучшие параметры: {grid.best_params_}")
    
    with open("best_params.json", "w") as f:
        json.dump(grid.best_params_, f)
    print("Параметры сохранены в best_params.json")

def train():
    print("Обучение на 100% данных...")
    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    params = {
        'max_depth': 3,
        'n_estimators': 100,
        'gamma': 1.0,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    if os.path.exists("best_params.json"):
        with open("best_params.json", "r") as f:
            params.update(json.load(f))
            
    # Всегда используем точный вес для полных данных
    params['scale_pos_weight'] = sum(y_train==0)/max(1, sum(y_train==1))
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    print(classification_report(y_test, model.predict(X_test)))
    model.save_model("xgb_obstacle.json")

if __name__ == "__main__":
    search()
    train()
