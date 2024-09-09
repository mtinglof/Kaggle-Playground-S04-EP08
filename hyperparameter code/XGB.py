import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt


train_df = pd.read_csv("clean_df.csv")
X = train_df.drop(columns=['class_e', 'class_p'])
y = train_df['class_e']

train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.01, random_state=16)

dtrain = xgb.DMatrix(train_x, label=train_y)
dval = xgb.DMatrix(val_x, label=val_y)

def mcc_eval(preds, dtrain):
    labels = dtrain.get_label()
    preds = [1 if p > 0.5 else 0 for p in preds]
    mcc = matthews_corrcoef(labels, preds)
    return 'MCC', mcc

params = {
    'objective': 'binary:logistic',
    'learning_rate': 0.05,
    'max_depth': 10,
    'eval_metric': 'logloss',
    'verbosity': 2,
    'seed': 16
}

evals = [(dtrain, 'train'), (dval, 'eval')]
bst = xgb.train(params, dtrain, num_boost_round=100, evals=evals, custom_metric=mcc_eval, verbose_eval=True)

y_pred = bst.predict(dval)
y_pred = [1 if y > 0.5 else 0 for y in y_pred]

mcc_score = matthews_corrcoef(val_y, y_pred)
print(f'Matthews Correlation Coefficient: {mcc_score:.4f}')

xgb.plot_importance(bst, importance_type='weight', max_num_features=10)
plt.show()