import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# Inserting the data from the files and cleaning it

train_data = pd.read_csv(r'train.csv')
# train_df = pd.DataFrame(train_data)

# Replacing the na values
col_na = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
           'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

col0 = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
col_none = ['MasVnrType']
col_SBrkr = ['Electrical']

# for col in col_na:
#     train_df[col] = train_df[col].fillna('NA')
# for col in col0:
#     train_df[col] = train_df[col].fillna(0)

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

# # Define columns
# col_en = [
#     'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 
#     'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 
#     'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 
#     'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 
#     'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
#     'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
#     'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
#     'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
#     'SaleType', 'SaleCondition' , # SalePrice removed
# ]

# numerical_cols = [
#     'Id', 'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 
#     'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
#     '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
#     'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 
#     'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 
#     'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 
#     'MiscVal', 'MoSold', 'YrSold', 'SalePrice', 'OverallQual', 'OverallCond'  # SalePrice stays here
# ]

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

# print(train_data[catergle_col].dtypes)








# print(train_data.head(10))
# y= train_df['SalePrice']
