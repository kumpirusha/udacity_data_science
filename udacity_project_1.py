import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""
Goals of the analysis:

    1. Find out which listings are the most profitable and why
    2. Check if there are any differences between regular hosts and SUperhosts
    (earnings, ratings)
    3. Find out which features impact listing price
    
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

# reformat price columns into a float dtype
for col in ['price_x', 'weekly_price', 'cleaning_fee', 'extra_people']:
    listings_full[col] = listings_full[col].str.replace(',', '').str.replace('$', '').astype(float)

# remove URL columns which are redundant
url_cols = listings_full.filter(like='url')
listings_full = listings_full.loc[:, ~listings_full.columns.isin(url_cols.columns)]

# rename columns
listings_full.rename(columns={'price_x': 'price'}, errors='raise', inplace=True)

# replace values with meaningful descriptors
cols_to_replace = {
    'host_is_superhost': {'t': 'superhost', 'f': 'regular'},
    'available': {'t': 'free', 'f': 'booked'}
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


##############################################################################################################
##############################################################################################################
# Which listings are the most profitable and why

# Create a subset of relevant variables
value_cols = [i for i in range(5)] + [i for i in range(12, 46)] + [i for i in range(47, 76)]
value_listings_db = listings_full.iloc[:, value_cols]

num_cols = []

for i in value_listings_db:

    if value_listings_db[i].dtype in ('float64', 'int64'):
        num_cols.append(i)

value_cor_db = value_listings_db.iloc[:, value_listings_db.columns.isin(num_cols)]
value_cor_miss = value_cor_db.isnull().mean()*100

value_cor = value_cor_db.corr()
matrix = np.triu(value_cor_db.corr())
sns.heatmap(value_cor[value_cor > 0.6], annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, cmap='coolwarm',
            mask=matrix)


# pivot the table to get some basic insights into the data
fig, ax = plt.subplots()
ax.plot(listings_full[['year', 'month', 'price']].groupby(by=['year', 'month'], dropna=True).mean(),
        label='Mean')
ax.plot(listings_full[['year', 'month', 'price']].groupby(by=['year', 'month']).max(), label='Max')
ax.plot(listings_full[['year', 'month', 'price']].groupby(by=['year', 'month']).min(), label='Min')
# ax.set_yscale('log')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Test plot')
ax.legend()
plt.show()

# Superhost
listings_full.pivot_table(index=['year', 'month'], values='review_scores_accuracy', aggfunc=np.count)

listings_full.pivot_table(index=['year', 'month'], values='price', columns='host_is_superhost',
                          aggfunc=[np.mean, min, max])

test_user = listings_full[listings_full['listing_id']==241032][['date', 'available', 'price', 'listing_id']]

listings_full.pivot_table(
    index=['host_is_superhost', 'available'], values='price', aggfunc=[np.mean, len]
)

listings_full.pivot_table(index=['date'], values='availability_30', columns='host_is_superhost',
                          aggfunc=np.sum).plot()
