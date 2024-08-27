import pandas as pd
from sklearn.preprocessing import MinMaxScaler

train_df = pd.read_csv("train.csv")

def remove_num_na(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].fillna(0)
    return df

def rewrite_values(df, threshold=0.1):
    for column in df.columns:
        if df[column].dtype == 'object':
            total_count = df[column].notna().sum()
            value_counts = df[column].value_counts()
            replace_dict = {
                value: 'not' if count / total_count < (threshold / 100) else value for value, count in value_counts.items()}
            df[column] = df[column].map(replace_dict).fillna('null')
    return df

def normalize(df): 
    scaler = MinMaxScaler()
    continuous_cols = df.select_dtypes(include=['float64']).columns
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
    return df

def encode(df): 
    df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, dtype=int)
    return df
    
def data_pipeline(df):
    df = remove_num_na(df)
    df = rewrite_values(df)
    df = normalize(df)
    df = encode(df)
    return df

clean_df = train_df.drop(['id'], axis=1)
clean_df = data_pipeline(clean_df)
clean_df.to_csv('clean_df.csv', index=False)