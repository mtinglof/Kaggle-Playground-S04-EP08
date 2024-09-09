import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


train_df = pd.read_csv("clean_df2.csv", keep_default_na=False)

categorical_columns = train_df.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('class')

onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')  # Updated for future versions
X_encoded = onehot_encoder.fit_transform(train_df[categorical_columns])
X_encoded_df = pd.DataFrame(X_encoded, columns=onehot_encoder.get_feature_names_out(categorical_columns))
X_non_categorical = train_df.drop(columns=categorical_columns + ['class'])

X = pd.concat([X_non_categorical.reset_index(drop=True), X_encoded_df], axis=1)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_df['class'])

train_data_x, temp_data_x, train_data_y, temp_data_y = train_test_split(X, y, test_size=0.02, random_state=16)
dev_data_x, test_data_x, dev_data_y, test_data_y = train_test_split(temp_data_x, temp_data_y, test_size=.5, random_state=16)

dev_data_x.to_csv("pca_encoded_dev.csv", index=False)
test_data_x.to_csv("pca_encoded_test.csv", index=False)

# pca = PCA(n_components=0.95)  
# X_reduced = pca.fit(X)

# joblib.dump(pca, 'pca_model.pkl')



# train_df = pd.read_csv("test_df2.csv", keep_default_na=False)

# categorical_columns = train_df.select_dtypes(include=['object']).columns.tolist()

# onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')  # Updated for future versions
# X_encoded = onehot_encoder.fit_transform(train_df[categorical_columns])
# X_encoded_df = pd.DataFrame(X_encoded, columns=onehot_encoder.get_feature_names_out(categorical_columns))
# X_non_categorical = train_df.drop(columns=categorical_columns)

# X = pd.concat([X_non_categorical.reset_index(drop=True), X_encoded_df], axis=1)
# X.to_csv("pca_encoded_test.csv", index=False)