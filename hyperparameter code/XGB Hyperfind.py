import xgboost as xgb
import optuna
import joblib 
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time

train_df = pd.read_csv("clean_df.csv")
X = train_df.drop(columns=['class_e', 'class_p'])
y = train_df['class_e']

def mcc_eval(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(preds)  
    return 'MCC', matthews_corrcoef(labels, preds)

def objective(trial):
    param = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.25),
        "n_estimators": trial.suggest_int("n_estimators", 200, 550),
        "max_depth": trial.suggest_int("max_depth", 7, 13),
        "subsample": trial.suggest_float("subsample", 0.8, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 3),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
        "lambda": trial.suggest_float("lambda", 5, 10),
        "alpha": trial.suggest_float("alpha", 0, 5),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 5),
        "random_state": 16,
    }

    trial_number = trial.number
    print(f"\nStarting Trial #{trial_number + 1}")
    print(f"Hyperparameters: {param}")

    start_time = time.time()

    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.01, random_state=16)

    model = xgb.XGBClassifier(**param, use_label_encoder=False)
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=100)

    y_pred = model.predict(val_x)
    mcc = matthews_corrcoef(val_y, y_pred)

    print(f"Trial #{trial_number + 1} MCC: {mcc}")
    
    if not hasattr(objective, "best_mcc") or mcc > objective.best_mcc:
        objective.best_mcc = mcc
        joblib.dump(model, "best_xgboost_model.pkl")  
        print(f"New best model saved with MCC: {mcc}")
    print(f"Best MMC: {objective.best_mcc}")
    end_time = time.time()
    print(f"Trial #{trial_number + 1} completed in {end_time - start_time:.2f} seconds\n")
    
    return mcc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)

print("\nOptimization completed.")
print(f"Best MCC: {study.best_value}")
print(f"Best Hyperparameters: {study.best_params}")