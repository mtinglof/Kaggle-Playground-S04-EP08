import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


train_df = pd.read_csv("clean_df.csv")
X = train_df.drop(columns=['class_e', 'class_p'])
y = train_df['class_e']

train_x, temp_x, train_y, temp_y = train_test_split(X, y, test_size=0.3, random_state=16)
drop_x, val_x, drop_y, val_y = train_test_split(temp_x, temp_y, test_size=0.015, random_state=16)

model = LinearSVC(random_state=16, verbose=True)
model.fit(train_x, train_y)

y_pred = model.predict(val_x)

# Evaluate the Model
print("Accuracy:", accuracy_score(val_y, y_pred))
print("Confusion Matrix:\n", confusion_matrix(val_y, y_pred))
print("Classification Report:\n", classification_report(val_y, y_pred))