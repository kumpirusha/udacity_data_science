import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function reads .csv files, assignes them to a variable and merges data
    into a single set.
        :param messages_filepath: A file path to the messages .csv file
        :param cetegories_filepath: A file path to the categories .csv file
        :return: A merged dataframe contaning both files
    """
    # load data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, how='outer', on='id')

    return (df)


def clean_data(df):
    """
    The function cleans that input dataframe by:
        - splitting the values in 'categories' column on ';' delimiter and saving to a new variable
        - uses the new variable to extract names for new variables from 'categories' columm
        - converts all other columns to type 'int32'
        - drops the original 'categories' column from the df
        - concats the new categories df on the existing one and drops and duplicated values
        * df: A pandas dataframe
    :return: Clean dataframe with 'categories' column parsed as individual columns for each categoty type
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype('int32')

    # drop the original categories column from `df`
    df.drop(labels='categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True) if sum(df.duplicated()) > 0 else df

    return df


def save_data(df, database_filename):
    """
    The function saves the dataframe 'df' into an SQL database.
        :param df: A pandas dataframe
        :param database_filename: A string name for the database
        :return: Nothing
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()