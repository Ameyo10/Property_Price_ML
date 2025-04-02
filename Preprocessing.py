import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Inserting the data from the files and cleaning it

train_data = pd.read_csv(r'D:\Python Projects\Python-Projects\Machine_Learning_Projects_Discord\Property_Price\train.csv')
train_df = pd.DataFrame(train_data)

# print(train_df.head(10))
y= train_df['SalePrice']

# The columns that contains the NaN value
contains_nan = train_df.columns[train_df.isna().any()].tolist()
print(contains_nan)
# # Finding the correlation between different features in the data given
# corr = train_df.corr()
# sns.heatmap(corr, annot = True, cmap= 'coolwarm', linewidths= 0.5)
# plt.title('Correlation map')
# plt.show()
