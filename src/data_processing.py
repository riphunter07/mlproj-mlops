import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):

    df["Date"] = pd.to_datetime(df["Date"],format="%d-%m-%Y")

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["week"] = df["Date"].dt.isocalendar().week

    df = df.fillna(method="ffill")

    q1 = df["Weekly_Sales"].quantile(0.25)
    q3 = df["Weekly_Sales"].quantile(0.75)

    iqr = q3 - q1
    
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    df = df[(df["Weekly_Sales"] >= lower) & (df["Weekly_Sales"] <= upper)]

    return df


def save_data(df, path):
    df.to_csv(path, index=False)


if __name__ == "__main__":

    raw_path = "data/raw/walmart_sales.csv"
    processed_path = "data/processed/clean_sales.csv"

    df = load_data(raw_path)

    df = preprocess_data(df)

    save_data(df, processed_path)

    print("Data processing complete.")