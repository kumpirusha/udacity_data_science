import pandas as pd
import os
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
for col in ['price', 'weekly_price', 'cleaning_fee', 'extra_people']:
    listings_full[col] = listings_full[col].str.replace(',', '').str.replace('$', '').astype(float)

# reformat 'host_response_rate'
listings_full.host_response_rate = listings_full.host_response_rate.str.replace('%', '').astype(float) / 100


# further remove redundant columns
cols_to_drop = ['name', 'summary', 'space', 'description', 'neighborhood_overview', 'notes', 'transit',
                'host_name', 'host_location', 'host_about', 'street', 'neighborhood_overview', 'zipcode',
                'smart_location', 'latitude', 'longitude', 'price_y', 'calendar_updated', 'neighbourhood',
                'host_acceptance_rate', 'state', 'city', 'is_location_exact']

listings_full = listings_full.loc[:, ~listings_full.columns.isin(cols_to_drop)]

listings_full.info()


##############################################################################################################
##############################################################################################################
# Find out which features impact listing price

# Remove NA from predictive value
price = listings_full.iloc[:,[3,1,4]].dropna(axis=0)

value_listings_db = pd.merge(left=listings_full, right=price, how='inner', on=['date','id'])
value_listings_db.drop(axis=1, labels='price_y', inplace=True)
value_listings_db.rename(columns={'price_x': 'price'}, errors='raise', inplace=True)


# Correlation
def correlation_matrix():
    '''
    Creates a correlation plot from numeric variables and plots values which are correlated > 0.6 with
    price
    '''

    num_cols = []

    for i in value_listings_db:

        if value_listings_db[i].dtype in ('float64', 'int64'):
            num_cols.append(i)

    value_cor_db = value_listings_db.iloc[:, value_listings_db.columns.isin(num_cols)]

    # for i in value_cor_db:
    #     print(i, '---', value_cor_db[i].unique(), '\n', '*' * 50)

    # check for correlation between variables
    value_cor = value_cor_db.corr()
    matrix = np.triu(value_cor_db.corr())

    print(
        sns.heatmap(value_cor[value_cor > .6], annot=True, fmt='.1g', vmin=-1, vmax=1, center=0,
                    cmap='coolwarm', mask=matrix)
    )

correlation_matrix()

# Plot variables to check for differences and predictive value
# property type
value_listings_db.pivot_table(index=['date'], values='price', columns='property_type', aggfunc=np.mean).plot()

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
dummy_cols = ['available',
              'host_response_time',
              'host_is_superhost',
              'host_has_profile_pic',
              'host_identity_verified',
              'property_type',
              'room_type',
              'bed_type',
              'instant_bookable',
              'cancellation_policy',
              'require_guest_profile_picture',
              'require_guest_phone_verification']

value_listings_lr = pd.get_dummies(value_listings_db, columns=dummy_cols)

# [i for i in value_listings_lr if value_listings_lr[i].dtype =='object']

test_cols = [2, 19, 20, 21, 31, 47, 48] + [i for i in range(13,16)]
lr_test_set = value_listings_lr.iloc[:, lambda value_listing_lr: test_cols]


##############################################################################################################
##############################################################################################################
# Check for differences between regular hosts and superhosts

# Superhost price diff
listings_full.pivot_table(index=['year', 'month'], values='price', columns='host_is_superhost',
                          aggfunc=np.mean).plot()
plt.show()

# differences in room availability
sh_available = listings_full.pivot_table(
    columns=['host_is_superhost', 'available'], values='price', aggfunc=[len], index=['year', 'month']
)

sh_available['regular % booked'] = (sh_available.iloc[:, 0] / (
        sh_available.iloc[:, 1] + sh_available.iloc[:, 1])) * 100
sh_available['superhost % booked'] = (sh_available.iloc[:, 2] / (
        sh_available.iloc[:, 2] + sh_available.iloc[:, 3])) * 100

sh_available.iloc[:, [4, 5]].plot(xlabel='Yearmonth', ylabel='% listings booked')
plt.show()

listings_full.pivot_table(
    values='review_scores_rating', index=['date'], aggfunc=sum
)

listings_full.review_scores_rating.hist()
plt.show()
listings_full.iloc[:, 20:24]
listings_full.iloc[:, 20:24].dropna(axis=0)