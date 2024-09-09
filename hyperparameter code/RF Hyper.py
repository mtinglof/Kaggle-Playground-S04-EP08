import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, matthews_corrcoef
import optuna
import joblib

train_df = pd.read_csv("clean_df2.csv")

categorical_columns = train_df.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('class')

onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')  # Updated for future versions
X_encoded = onehot_encoder.fit_transform(train_df[categorical_columns])
X_encoded_df = pd.DataFrame(X_encoded, columns=onehot_encoder.get_feature_names_out(categorical_columns))
X_non_categorical = train_df.drop(columns=categorical_columns + ['class'])

X = pd.concat([X_non_categorical.reset_index(drop=True), X_encoded_df], axis=1)

pca = PCA(n_components=0.95)  
X_reduced = pca.fit_transform(X)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_df['class'])

train_x, temp_x, train_y, temp_y = train_test_split(X_reduced, y, test_size=0.9, random_state=24)
drop_x, val_x, drop_y, val_y = train_test_split(temp_x, temp_y, test_size=0.02, random_state=24)

print(train_x.shape)
print(val_x)

best_mcc_score = 0  

def objective(trial):
    global best_mcc_score

    n_estimators = trial.suggest_int('n_estimators', 100, 250)
    max_depth = trial.suggest_int('max_depth', 20, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    max_features = trial.suggest_float('max_features', 0.1, 1.0)
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=16,
        n_jobs=-1,
        verbose=1
    )
    
    mcc_scorer = make_scorer(matthews_corrcoef)
    score = cross_val_score(clf, train_x, train_y, cv=5, scoring=mcc_scorer).mean()
    
    if score > best_mcc_score:
        best_mcc_score = score
        print(f"New best MCC score: {best_mcc_score:.4f}")

    print(f"Best MCC score after trial {trial.number}: {best_mcc_score:.4f}\n")

    return score

print("Creating the Optuna study...")
study = optuna.create_study(direction='maximize') 
print("Starting the optimization process...")
study.optimize(objective, n_trials=20)

print("Optimization finished!")
print("Best hyperparameters: ", study.best_params)

print("Training the model with the best hyperparameters...")
best_clf = RandomForestClassifier(**study.best_params, random_state=16, verbose=1)
best_clf.fit(train_x, train_y)

print("Saving the best model...")
joblib.dump(best_clf, 'best_random_forest_model_optuna.pkl')

print("Evaluating the best model on the test set...")
y_pred = best_clf.predict(val_x)
mcc_test = matthews_corrcoef(val_y, y_pred)
print(f"MCC on Test Set: {mcc_test:.4f}")
