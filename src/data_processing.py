import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# 1. Aggregate Features
def create_aggregate_features(df):
    agg_df = df.groupby('CustomerId').agg(
        total_transaction_amount=('Amount', 'sum'),
        avg_transaction_amount=('Amount', 'mean'),
        transaction_count=('TransactionId', 'count'),
        transaction_amount_std=('Amount', 'std')
    ).reset_index()
    return agg_df

# 2. Datetime Features
def extract_datetime_features(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year
    return df

# 3. Categorical Encoding
def encode_categorical(df):
    # One-hot encoding for ProductCategory
    one_hot = pd.get_dummies(df['ProductCategory'], prefix='ProductCategory')
    df = pd.concat([df, one_hot], axis=1)
    
    # Label encoding for ProviderId
    le = LabelEncoder()
    df['ProviderId_encoded'] = le.fit_transform(df['ProviderId'].astype(str))
    
    return df

# 4. Handle Missing Values
def handle_missing_values(df):
    df = df.fillna({
        'CountryCode': df['CountryCode'].mode()[0],
        'ProviderId': 'Unknown',
        'TransactionStartTime': df['TransactionStartTime'].mode()[0]
    })
    return df

# 5. Scale Numeric Features
def scale_numerical(df, columns_to_scale):
    scaler = StandardScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df

# 6. Full Preprocessing Pipeline
def build_preprocessing_pipeline():
    numeric_features = ['Amount', 'Value']
    categorical_features = ['ProductCategory', 'ChannelId']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor
