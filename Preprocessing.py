import pandas as pd

# Inserting the data from the files and cleaning it

train_data = pd.read_csv('D:/Python Projects/Python-Projects/Machine_Learning_Projects_Discord/Property_Price/train.csv')
test_data = pd.read_csv('D:/Python Projects/Python-Projects/Machine_Learning_Projects_Discord/Property_Price/test.csv')

# Replacing the na values
col_na = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
           'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

col0 = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
col_none = ['MasVnrType']
col_SBrkr = ['Electrical']

numeric_col=train_data.select_dtypes(include=['number']).columns[0:]
catergle_col=train_data.select_dtypes(include=['object']).columns[0:]
numeric_mean=train_data[numeric_col].mean()

train_data[numeric_col]=train_data[numeric_col].fillna(numeric_mean)
train_data[catergle_col]=train_data[catergle_col].fillna('NA')


train_data[col_none] = train_data[col_none].fillna('None')
train_data[col_SBrkr] = train_data[col_SBrkr].fillna('SBrkr')

col_num = ['MSSubClass']
for col in col_num:
    train_data[col] = train_data[col].astype(str)

# Encode categorical columns and join with numericals
train_encoded = pd.get_dummies(
    train_data[catergle_col], 
    drop_first=True
).join(train_data[numeric_col])

# Compute correlation matrix
corr_matrix = train_encoded.corr()

imp_features = {}
# Extract the 'SalePrice' correlation series
sale_price_corr = corr_matrix['SalePrice']

# Iterate over feature names (index) and their correlation values
for feature, corr in sale_price_corr.items():
    if corr > 0.5 and feature != 'SalePrice':  # Exclude SalePrice itself
        imp_features[feature] = corr

print(imp_features)

train_data.to_csv('D:/Python Projects/Python-Projects/Machine_Learning_Projects_Discord/Property_Price/train.csv')

# Making the similar preprocessing steps to the test dataset
numeric_col1=test_data.select_dtypes(include=['number']).columns[0:]
catergle_col1=test_data.select_dtypes(include=['object']).columns[0:]
numeric_mean=test_data[numeric_col1].mean()

test_data[numeric_col1]=test_data[numeric_col1].fillna(numeric_mean)
test_data[catergle_col1]=test_data[catergle_col1].fillna('NA')


test_data[col_none] = test_data[col_none].fillna('None')
test_data[col_SBrkr] = test_data[col_SBrkr].fillna('SBrkr')

col_num = ['MSSubClass']
for col in col_num:
    test_data[col] = test_data[col].astype(str)

test_data.to_csv('D:/Python Projects/Python-Projects/Machine_Learning_Projects_Discord/Property_Price/test.csv')