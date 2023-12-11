import pandas as pd
from pandasgui import show

def remove_rows_with_missing_ratings(df):
    col_with_missing_values = ["Cleanliness_rating", "Accuracy_rating", "Communication_rating", "Location_rating", "Check-in_rating", "Value_rating"]
    df.dropna(subset = col_with_missing_values, inplace = True)
    return df

    
def combine_description_strings(df):
    df = df.dropna(subset=['Description'])
    df['Description'] = df['Description'].str.replace('About this space', '')
    df['Description'] = df['Description'].str.replace("''", '')
    #df['Description'] = df['Description'].apply(eval)
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
    cleaned_df = remove_rows_with_missing_ratings(cleaned_df)
    cleaned_df = combine_description_strings(cleaned_df)
    cleaned_df = set_default_feature_values(cleaned_df)
    cleaned_df = cleaned_df.drop(columns=["Unnamed: 19"])
    print(cleaned_df.columns)
    #cleaned_df = cleaned_df.drop(columns=["Unnamed: 0"])
    # string_columns = cleaned_df.select_dtypes(include=['object']).columns
    # cleaned_df = cleaned_df.drop(string_columns, axis=1)
    return cleaned_df


if __name__ == "__main__":
    raw_data = pd.read_csv("D:/AiCore/Projects/AirBnb/airbnb-property-listings/tabular_data/AirBnbData.csv")
    processed_data = clean_tabular_data(raw_data)
    processed_data.loc[586, 'guests'] = 1
    processed_data.loc[586, 'bedrooms'] = 1

    processed_data['guests'] = processed_data['guests'].astype('int64')
    processed_data['bedrooms'] = processed_data['bedrooms'].astype('int64')
    processed_data = processed_data.replace(0.0, 1)
    processed_data = processed_data.replace(200, 1)
    print(processed_data.info())

    processed_data.to_csv("D:/AiCore/Projects/AirBnb/airbnb-property-listings/tabular_data/Cleaned_AirBnbData_3.csv")
    show(processed_data)

def load_airbnb(df, label_col):
    labels = df[label_col]
    features = df.drop(label_col, axis=1)
    return (features, labels)


