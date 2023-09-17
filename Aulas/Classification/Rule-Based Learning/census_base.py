import Orange

census_base = Orange.data.Table('census_rules.csv')
divided_base = Orange.evaluation.testing.sample(census_base, n=0.25)

trainment_base = divided_base[1]
test_base = divided_base[0]

cn2 = Orange.classification.rules.CN2Learner()
census_rules = cn2(trainment_base)

for rule in census_rules.rule_list:
    print(rule)
print()

print(census_base.domain.class_var.values)

forecast = Orange.evaluation.testing.TestOnTestData(trainment_base, test_base, [lambda testdata: census_rules])
print(Orange.evaluation.CA(forecast))