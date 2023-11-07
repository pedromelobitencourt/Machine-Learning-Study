import Orange

credit_base = Orange.data.Table('credit_rules.csv') # We put 'i#' to ignore the attribute
print(credit_base.domain)

divided_base = Orange.evaluation.testing.sample(credit_base, n=0.25) # An array with two elements

trainment_base = divided_base[1] # 75%
test_base = divided_base[0] # 25%

cn2 = Orange.classification.rules.CN2Learner()
credit_base_rules = cn2(trainment_base)

for rule in credit_base_rules.rule_list:
    print(rule)
print()

print(credit_base.domain.class_var.values)

forecast = Orange.evaluation.testing.TestOnTestData(trainment_base, test_base, [lambda testdata: credit_base_rules])
print(Orange.evaluation.CA(forecast))