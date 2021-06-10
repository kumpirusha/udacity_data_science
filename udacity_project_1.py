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

##############################################################################################################
##############################################################################################################
# Find out which features impact listing price

# Remove NA from predictive value
price = listings_full.iloc[:, [3, 1, 4]].dropna(axis=0)

value_listings_db = pd.merge(left=listings_full, right=price, how='inner', on=['date', 'id'])
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
value_listings_db.pivot_table(index=['date'], values='listing_id', columns='property_type',
                              aggfunc=pd.Series.nunique).plot()

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

# cancellation policy
value_listings_db.pivot_table(index=['year', 'month'],
                              values='price',
                              columns=['cancellation_policy'],
                              aggfunc=[np.mean]).plot(grid=True, ylabel='Mean price')

value_listings_db.pivot_table(index=['year', 'month'],
                              values='listing_id',
                              columns=['cancellation_policy'],
                              aggfunc=[pd.Series.nunique]).plot(grid=True, ylabel='# of listings')

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

value_listings_lr.info()

test_cols = [2, 13, 81]  # 20, 48, 69,  77]
test_cols_2 = [2, 13, 81, 20, 48, 69, 77]
lr_test_set = value_listings_lr.iloc[:, lambda value_listing_lr: test_cols_2]
lr_test_clean = lr_test_set.dropna(axis=0)

model = sm.OLS(lr_test_clean.price, lr_test_clean.drop(axis=1, columns='price'))
results = model.fit()
results.summary()

##############################################################################################################
##############################################################################################################
# Check for differences between regular hosts and superhosts
# Superhost price diff
value_listings_db.pivot_table(index=['year', 'month'], values='price', columns='host_is_superhost',
                              aggfunc=np.mean).plot()

# superhost booking difference for most common properties
value_listings_db[value_listings_db['property_type'].isin(['House', 'Apartment'])].pivot_table(
    index=['month', 'year'], values='price',
    columns=['property_type', 'host_is_superhost'],
    aggfunc=np.mean).plot()

# differences in room availability
sh_available = listings_full.pivot_table(
    columns=['host_is_superhost', 'available'], values='price', aggfunc=[len], index=['year', 'month']
)

sh_available['regular % booked'] = (sh_available.iloc[:, 0] / (
        sh_available.iloc[:, 0] + sh_available.iloc[:, 1])) * 100
sh_available['superhost % booked'] = (sh_available.iloc[:, 2] / (
        sh_available.iloc[:, 2] + sh_available.iloc[:, 3])) * 100

sh_available.iloc[:, [4, 5]].plot(xlabel='Yearmonth', ylabel='% listings booked')
plt.show()

# independent sample t-test - compare the prices of superhosts vs. regular hosts
# check the distributions of samples
superhost = value_listings_db[value_listings_db['host_is_superhost'].isin(['superhost'])].price
regular_host = value_listings_db[value_listings_db['host_is_superhost'].isin(['regular'])].price

sns.kdeplot(superhost, shade=True)
sns.kdeplot(regular_host, shade=True)
plt.title("Independent Sample T-Test")

t_test = sm.stats.ttest_ind(superhost, regular_host)
t_test

##############################################################################################################
##############################################################################################################
# Check which listings are the most popular
booked_listings = listings_full[listings_full['available'].isin(['booked'])]
bookings = listings_full.pivot_table(index='id', columns='available', values='listing_id', aggfunc=len)
bookings_db = pd.merge(left=listings_csv, right=bookings, how='left', on=['id'])
bookings_db.fillna(value={'booked': 0, 'free': 0}, inplace=True)
bookings_db['booked_ratio'] = bookings_db.booked / (bookings_db.booked + bookings_db.free)

# reformat price columns into a float dtype
for col in ['price', 'weekly_price', 'cleaning_fee', 'extra_people']:
    bookings_db[col] = bookings_db[col].str.replace(',', '').str.replace('$', '').astype(float)

# Plot variables to check for differences and predictive value
# property type
bookings_db.pivot_table(index=['property_type'], values='booked_ratio', aggfunc=np.mean).sort_values(
    'booked_ratio').plot(kind='bar')

# room type
bookings_db.pivot_table(index=['room_type'], values='booked_ratio', aggfunc=np.mean).sort_values(
    'booked_ratio').plot(kind='bar')

# neighbourhood
bookings_db.pivot_table(index=['neighbourhood_cleansed'], values='booked_ratio', aggfunc=np.mean).sort_values(
    'booked_ratio', ascending=False)[:10]

# amenities
bookings_db.pivot_table(index=['host_response_rate'], values='booked_ratio', aggfunc=np.mean).plot(kind='bar')

pd.get_dummies(data=bookings_db.amenities)

# review scores
listings_full.pivot_table(values='number_of_reviews', index='review_scores_rating',
                          aggfunc=sum).plot(kind='bar')

# people it accomodates
value_listings_db.pivot_table(values='listing_id', index='accommodates', aggfunc=pd.Series.nunique).plot(
    kind='bar')
value_listings_db.pivot_table(values='price', index='accommodates', aggfunc=np.mean).plot(kind='bar')

# cancellation policy
value_listings_db.pivot_table(index=['year', 'month'],
                              values='price',
                              columns=['cancellation_policy'],
                              aggfunc=[np.mean]).plot(grid=True, ylabel='Mean price')

value_listings_db.pivot_table(index=['year', 'month'],
                              values='listing_id',
                              columns=['cancellation_policy'],
                              aggfunc=[pd.Series.nunique]).plot(grid=True, ylabel='# of listings')
