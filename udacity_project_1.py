import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns

"""
Goals of the analysis:

    1. Find out which features impact listing price
    2. Check if there are any differences between regular hosts and Superhosts
    (earnings, ratings)
    3. Find out which listings are the most booked and why
    
"""

# os.chdir(r'C:\Users\jaka.vrhovnik\Desktop\Jaka_archive\Python\udacity_data_science\project_1')
os.chdir(r'D:\Udacity')

calendar_csv = pd.read_csv(r'calendar.csv')
listings_csv = pd.read_csv(r'listings.csv')
# Reviews will not be used in the analysis due to limited usability


# check the imports for missing data
miss_calendar = calendar_csv.isnull().mean() * 100
miss_listings = pd.DataFrame(listings_csv.isnull().mean() * 100, columns=['missing'])

# remove columns with more than half of the data missing and merge the two databases
listings_removed_cols = miss_listings[miss_listings.missing > 50]
listings_subset = listings_csv.loc[:, ~listings_csv.columns.isin(listings_removed_cols.index)]
listings_full = pd.merge(left=calendar_csv, right=listings_subset, how='left', left_on='listing_id',
                         right_on='id')

# check missing values
miss_full = listings_full.isnull().mean() * 100

# remove URL columns which are redundant
url_cols = listings_full.filter(like='url')
listings_full = listings_full.loc[:, ~listings_full.columns.isin(url_cols.columns)]

# rename columns
listings_full.rename(columns={'price_x': 'price'}, errors='raise', inplace=True)

# replace values with meaningful descriptors
cols_to_replace = {
    'host_is_superhost': {'t': 'superhost', 'f': 'regular'},
    'available': {'t': 'free', 'f': 'booked'},

}

for k, v in cols_to_replace.items():
    listings_full[k].replace(
        to_replace=v,
        inplace=True
    )

listings_full['year'] = pd.DatetimeIndex(listings_full['date']).year
listings_full['month'] = pd.DatetimeIndex(listings_full['date']).month

listings_full.info(verbose=True, null_counts=True)

# drop cols with single unique value
cols_len_one = []

for i in listings_full.columns:
    if len(listings_full[i].unique()) == 1:
        cols_len_one.append(i)

listings_full = listings_full.loc[:, ~listings_full.columns.isin(cols_len_one)]

# reformat price columns into a float dtype
for col in ['price_x', 'weekly_price', 'cleaning_fee', 'extra_people']:
    listings_full[col] = listings_full[col].str.replace(',', '').str.replace('$', '').astype(float)

# reformat 'host_response_rate'
listings_full.host_response_rate = listings_full.host_response_rate.str.replace('%', '').astype(float)/100



listings_full.info()

for i in listings_full:
    print(i, '---', listings_full[i].unique(), '\n','*' * 50)

# further remove redundant columns
cols_to_drop = ['name', 'summary', 'space', 'description', 'neighborhood_overview', 'notes', 'transit',
                'host_name', 'host_location', 'host_about', 'street', 'neighborhood_overview', 'zipcode',
                'smart_location', 'latitude', 'longitude', 'price_y', 'calendar_updated', 'neighbourhood',
                'host_acceptance_rate', 'state', 'city', 'is_location_exact']

listings_full = listings_full.loc[:, ~listings_full.columns.isin(cols_to_drop)]

##############################################################################################################
##############################################################################################################
# Find out which features impact listing price

# Create a subset of relevant variables
value_cols = [i for i in range(5)] + [i for i in range(13, 46)] + [i for i in range(47, 76)]
value_listings_db = listings_full.iloc[:, value_cols]

# change categorical values to correct format
value_listings_db.room_type = value_listings_db.room_type.astype('category')
value_listings_db.review_scores_value = value_listings_db.review_scores_value.astype('category')

# Correlation
num_cols = []

for i in value_listings_db:

    if value_listings_db[i].dtype in ('float64', 'int64'):
        num_cols.append(i)

value_cor_db = value_listings_db.iloc[:, value_listings_db.columns.isin(num_cols)]
value_cor_miss = value_cor_db.isnull().mean() * 100

# handpick relevant variables
value_cor_hp = value_listings_db.iloc[:,
               [2, 3, 9, 10, 12, 14, 17, 18, 19, 20, 30, 31, 32, 33, 34, 35, 36, 40, 41, 49, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 64]]

pd.get_dummies(value_cor_hp, columns=['available', 'host_response_time','host_is_superhost',
                                      'host_has_profile_pic', 'host_identity_verified'])

for i in value_cor_hp:
    print(i, '---', value_cor_hp[i].unique(), '\n','*' * 50)


# check for correlation between variables
value_cor = value_cor_hp.corr()
matrix = np.triu(value_cor_hp.corr())
sns.heatmap(value_cor, annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, cmap='coolwarm',
            mask=matrix)

# Plot variables to check for differences and predictive value
# property type
value_listings_db.pivot_table(index=['date'], values='price', columns='property_type',
                              aggfunc=np.mean).plot()

# room type
value_listings_db.pivot_table(index=['date'], values='price', columns='room_type',
                              aggfunc=np.mean).plot()

# review scores
value_listings_db.pivot_table(values='listing_id', index='review_scores_rating',
                              aggfunc=pd.Series.nunique).plot(kind='bar')

# people it accomodates
value_listings_db.pivot_table(values='listing_id', index='accommodates', aggfunc=pd.Series.nunique).plot(
    kind='bar')

value_listings_db.pivot_table(values='price', index='accommodates', aggfunc=np.mean).plot(kind='bar')

# host response time
value_listings_db.pivot_table(index=['year', 'month'],
                              values='price',
                              columns=['cancellation_policy'],
                              aggfunc=[np.mean]).plot(grid=True)

# apply a multiple regression model
model = LinearRegression

# # pivot the table to get some basic insights into the data
# fig, ax = plt.subplots()
# ax.plot(listings_full[['year', 'month', 'price']].groupby(by=['year', 'month'], dropna=True).mean(),
#         label='Mean')
# ax.plot(listings_full[['year', 'month', 'price']].groupby(by=['year', 'month']).max(), label='Max')
# ax.plot(listings_full[['year', 'month', 'price']].groupby(by=['year', 'month']).min(), label='Min')
# # ax.set_yscale('log')
# ax.set_xlabel('Date')
# ax.set_ylabel('Price')
# ax.set_title('Test plot')
# ax.legend()
# plt.show()


# Superhost
listings_full.pivot_table(index=['year', 'month'], values='review_scores_accuracy', aggfunc=np.count)

listings_full.pivot_table(
    index=['host_is_superhost', 'available'], values='price', aggfunc=[np.mean, len]
)
