import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import matthews_corrcoef


train_df = pd.read_csv("clean_df.csv")
X = train_df.drop(columns=['class_e', 'class_p'])
y = train_df['class_e']

train_x, temp_x, train_y, temp_y = train_test_split(X, y, test_size=0.3, random_state=16)
drop_x, val_x, drop_y, val_y = train_test_split(temp_x, temp_y, test_size=0.015, random_state=16)


print("Training...")
model = RandomForestClassifier(
    n_estimators=50, 
    max_depth=15, 
    min_samples_split=10, 
    min_samples_leaf=5, 
    random_state=16
)
model.fit(train_x, train_y)
y_pred = model.predict(val_x)

print(classification_report(val_y, y_pred))
mcc = matthews_corrcoef(val_y, y_pred)
print(f'Matthews Correlation Coefficient: {mcc:.5f}')

feature_importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(feature_importances)[::-1]

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.title("Feature Importance - Random Forest")
plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()