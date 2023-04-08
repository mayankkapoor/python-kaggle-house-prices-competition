import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split


def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data


def preprocess_data(train_data, test_data):
    numerical_columns = train_data.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = train_data.select_dtypes(include=['object']).columns

    # Remove 'SalePrice' from the numerical_columns list
    numerical_columns = numerical_columns.drop('SalePrice')

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    X_train = train_data.drop('SalePrice', axis=1)
    y_train = train_data['SalePrice']
    X_test = test_data

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_train_preprocessed, y_train, X_test_preprocessed


def log_rmse(y_true, y_pred):
    log_y_true = np.log(y_true + 1)
    log_y_pred = np.log(y_pred + 1)
    return np.sqrt(mean_squared_error(log_y_true, log_y_pred))


def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)
    return model


def validate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    log_rmse_score = log_rmse(y_val, y_pred)
    print(f"Log RMSE: {log_rmse_score}")


def create_submission_file(model, X_test, test_data, file_name='submission.csv'):
    y_test_pred = model.predict(X_test)
    submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': y_test_pred})
    submission.to_csv(file_name, index=False)
    print(f"Submission file saved as {file_name}")


def main():
    train_data, test_data = load_data('data/train.csv', 'data/test.csv')

    X_train_preprocessed, y_train, X_test_preprocessed = preprocess_data(train_data, test_data)

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_preprocessed, y_train, test_size=0.2, random_state=42)

    model = train_model(X_train_split, y_train_split)

    validate_model(model, X_val_split, y_val_split)

    create_submission_file(model, X_test_preprocessed, test_data)


if __name__ == '__main__':
    main()
