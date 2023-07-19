import pandas as pd

def remove_rows_with_missing_ratings(df):
    df.dropna(inplace = True)
    return df


def combine_description_strings(df):
    df = df.dropna(subset=['Description'])
    df['Description'] = df['Description'].str.replace('About this space', '')
    df['Description'] = df['Description'].str.replace("''", '')
    df['Description'] = df['Description'].apply(eval)
    df['Description'] = df['Description'].apply(lambda x: ' '.join(x))
    return df


def set_default_feature_values(df):
    df['guests'] = df['guests'].fillna(1)
    df['beds'] = df['beds'].fillna(1)
    df['bathrooms'] = df['bathrooms'].fillna(1)
    df['bedrooms'] = df['bedrooms'].fillna(1)
    return df


def clean_tabular_data(raw_df):
    cleaned_df = raw_df.copy()
    cleaned_df = combine_description_strings(cleaned_df)
    cleaned_df = set_default_feature_values(cleaned_df)
    return cleaned_df


if __name__ == "__main__":
    raw_data = pd.read_csv("C:/Users/Sofaglobe/Desktop/AiCore_Projects/Modelling Airbnb's property listing dataset/airbnb-property-listings/tabular_data/AirBnbData.csv")
    processed_data = clean_tabular_data(raw_data)
    processed_data.to_csv("C:/Users/Sofaglobe/Desktop/AiCore_Projects/Modelling Airbnb's property listing dataset/airbnb-property-listings/tabular_data/Cleaned_AirBnbData.csv")


def load_airbnb(df, label_col):
    labels = df[label_col]
    features = df.drop(label_col, axis=1)
    return (features, labels)


