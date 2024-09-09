import optuna
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import matthews_corrcoef, make_scorer
import pandas as pd
import joblib


def objective(trial):
    print(f"Starting Trial {trial.number}")
    C = trial.suggest_float('C', 1e-6, 1e+6, log=True)
    loss = trial.suggest_categorical('loss', ['hinge', 'squared_hinge'])
    max_iter = trial.suggest_int('max_iter', 1000, 5000)
    
    model = LinearSVC(C=C, loss=loss, max_iter=max_iter, verbose=True, random_state=16)
    
    scorer = make_scorer(matthews_corrcoef)
    scores = cross_val_score(model, train_x, train_y, cv=3, scoring=scorer, n_jobs=-1)
    score_mean = scores.mean()
    
    print(f"Trial {trial.number}: Score = {score_mean:.4f}")
    
    return score_mean

train_df = pd.read_csv("clean_df.csv")
X = train_df.drop(columns=['class_e', 'class_p'])
y = train_df['class_e']

train_x, temp_x, train_y, temp_y = train_test_split(X, y, test_size=0.9, random_state=16)
drop_x, val_x, drop_y, val_y = train_test_split(temp_x, temp_y, test_size=0.02, random_state=16)
print(train_x.shape)
print(val_x.shape)

def print_trial_results(study, trial):
    print(f"Completed trial {trial.number} with value: {trial.value:.4f}")
    print(f"Best score so far: {study.best_value:.4f}")

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20, callbacks=[print_trial_results])

print("Best Parameters:", study.best_params_)
print("Best Cross-Validation MCC Score:", study.best_value)

best_params = study.best_params_
best_model = LinearSVC(C=best_params['C'], loss=best_params['loss'], max_iter=best_params['max_iter'], random_state=42)
best_model.fit(train_x, train_y)

y_pred = best_model.predict(val_x)
test_mcc = matthews_corrcoef(val_y, y_pred)
print("Test Set MCC:", test_mcc)

joblib.dump(best_model, 'best_linear_svc_model_optuna.pkl')
print("Best model saved as 'best_linear_svc_model_optuna.pkl'.")