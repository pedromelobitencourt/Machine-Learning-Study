import pandas as pd
from apyori import apriori

base = pd.read_csv('market1.csv', header=None)

# print(base) # DataFrame

transactions = []

for i in range(len(base)):
    # print(base.values[i])
    transactions.append([str(base.values[i, j]) for j in range(base.shape[1])])

print('\n\n', transactions, '\n')

combinations = apriori(transactions, min_support=0.3, min_confidence=0.8, min_lift=2)

# Set of items: It has the combinations of each item, its support value, its confidence value and its lift value
combinations = list(combinations) 

# for combination in combinations:
#     print(combination, '\n')

r = combinations[2]

A = [] # If
B = [] # Then
support = []
confidence = []
lift = []

for combination in combinations:
    s = combination[1]
    rules = combination[2]

    for rule in rules:
        a = list(rule[0])
        b = list(rule[1])
        c = rule[2]
        l = rule[3]

        A.append(a)
        B.append(b)
        support.append(s)
        confidence.append(c)
        lift.append(l)
    # print(s)

rules_df = pd.DataFrame({'If': A, 'Then': B, 'Support': support, 'Confidence': confidence, 'lift': lift})
rules_df = rules_df.sort_values(by='lift', ascending=False)

print(rules_df)