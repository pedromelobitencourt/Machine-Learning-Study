import Orange

credit_risk_base = Orange.data.Table('credit_risk.csv') # Format: Table

# The classes (attributes) will have a 'c#' as a preffixe in the csv file

print(credit_risk_base) # [[predictors | classes],
                            #[     ]   ]

print(credit_risk_base.domain) # It shows the column names; It separates the predictors and the classes as well with a '|'

cn2 = Orange.classification.rules.CN2Learner()
credit_risk_rules = cn2(credit_risk_base) # Trainment
print()

for rule in credit_risk_rules.rule_list:
    print(rule)
print()

# credit_report: good, debt: high, guarantee: none, income: > 35
# credit_report: bad, debt: high, guarantee: adequate, income: <= 35
forecast = credit_risk_rules([['boa', 'alta', 'nenhuma', 'acima_35'], ['ruim', 'alta', 'adequada', '0_15']])

print(credit_risk_base.domain.class_var.values) # ('alto', 'baixo', 'moderado'); 0 == alto, 1 == baixo, 2 == moderado

forecast2 = []

for i in forecast:
    forecast2.append(credit_risk_base.domain.class_var.values[i])                                                                 

print(forecast)
print(forecast2)