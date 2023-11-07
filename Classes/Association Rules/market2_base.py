import pandas as pd
from apyori import apriori

base = pd.read_csv('market2.csv', header=None)
print(base.shape)

transactions = []

for i in range(base.shape[0]):
    # print(base.values[i])
    
    transactions.append([str(base.values[i, j]) for j in range(base.shape[1])])

print(len(transactions))


# Products that are sold, at least, 4 times a day
# 4 * 7 (this base represents the market sales in a week) = 28
# 28 / 7501 (how many lines)
combinations = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3)

# for combination in combinations:
#     print(combination)

A = []
B = []
support = []
confidence = []
lift = []

for combination in combinations:
    s = combination[1]
    rule_set = combination[2]

    for rule in rule_set:
        # print(rule)
        a = rule[0]
        b = rule[1]
        c = rule[2]
        l = rule[3]

        A.append(a)
        B.append(b)
        support.append(s)
        confidence.append(c)
        lift.append(l)

rules_df = pd.DataFrame({'If': A, 'Then': B, 'Support': support, 'Confidence': confidence, 'Lift': lift})
rules_df = rules_df.sort_values(by='Lift', ascending=False)

print(rules_df)