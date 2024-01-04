import pandas as pd
from pyECLAT import ECLAT

market_base = pd.read_csv('../../../Databases/market1.csv', header=None)
print(market_base, '\n============================================================')

eclat = ECLAT(data=market_base)

print(eclat.df_bin)
print(eclat.uniq_, '\n============================================================')

indexes, support = eclat.fit(min_support=0.3, min_combination=1, max_combination=3)
print(indexes, '\n\n', support, '\n============================================================')