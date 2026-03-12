import pandas as pd


def load_data(path):

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    return df


def sort_data(df):

    df = df.sort_values(["Store", "Date"])

    return df


def create_lag_features(df):

    df["lag_1"] = df.groupby("Store")["Weekly_Sales"].shift(1)
    df["lag_2"] = df.groupby("Store")["Weekly_Sales"].shift(2)
    df["lag_4"] = df.groupby("Store")["Weekly_Sales"].shift(4)

    return df


def create_rolling_features(df):

    df["rolling_mean_4"] = (
        df.groupby("Store")["Weekly_Sales"]
        .shift(1)
        .rolling(4)
        .mean()
    )

    df["rolling_std_4"] = (
        df.groupby("Store")["Weekly_Sales"]
        .shift(1)
        .rolling(4)
        .std()
    )

    return df


def create_time_features(df):

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["week_of_year"] = df["Date"].dt.isocalendar().week

    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

    return df


def drop_missing_rows(df):

    df = df.dropna()

    return df


def save_features(df, path):

    df.to_csv(path, index=False)


if __name__ == "__main__":

    input_path = "data/processed/clean_sales.csv"
    output_path = "data/processed/features_sales.csv"

    df = load_data(input_path)

    df = sort_data(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = create_time_features(df)

    df = drop_missing_rows(df)

    save_features(df, output_path)

    print("Feature engineering complete.")