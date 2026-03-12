import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn

mlflow.set_experiment("retail_demand_forecasting")


def load_data(path):

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    return df


def train_test_split(df):

    df = df.sort_values("Date")

    split_date = df["Date"].quantile(0.8)

    train = df[df["Date"] <= split_date]
    test = df[df["Date"] > split_date]

    return train, test


def prepare_features(train, test):

    target = "Weekly_Sales"

    features = [
        "Store",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Holiday_Flag",
        "lag_1",
        "lag_2",
        "lag_4",
        "rolling_mean_4",
        "rolling_std_4",
        "day_of_week",
        "month",
        "week_of_year",
        "is_weekend"
    ]

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):

    models = {}

    models["linear_regression"] = LinearRegression()

    models["random_forest"] = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    models["gradient_boosting"] = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    trained_models = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models


def evaluate_models(models, X_test, y_test):

    results = {}

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            preds = model.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            rmse = mean_squared_error(y_test, preds)**0.5

            mlflow.log_param("model", name)

            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)

            mlflow.sklearn.log_model(model, name)

            results[name] = {
                "MAE": mae,
                "RMSE": rmse
            }

            print(name)
            print("MAE:", mae)
            print("RMSE:", rmse)
            print("---------------")
            
    return results


def select_best_model(models, results):

    best_model_name = min(results, key=lambda x: results[x]["RMSE"])

    best_model = models[best_model_name]

    print("Best model:", best_model_name)

    return best_model_name, best_model


def save_model(model, name):

    path = f"models/{name}.pkl"

    with open(path, "wb") as f:
        pickle.dump(model, f)

    print("Model saved:", path)


if __name__ == "__main__":

    data_path = "data/processed/features_sales.csv"

    df = load_data(data_path)

    train, test = train_test_split(df)

    X_train, X_test, y_train, y_test = prepare_features(train, test)

    models = train_models(X_train, y_train)

    results = evaluate_models(models, X_test, y_test)

    best_name, best_model = select_best_model(models, results)

    save_model(best_model, best_name)