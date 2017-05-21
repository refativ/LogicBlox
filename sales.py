import numpy as np
import pandas as pd
import patsy
import re
import math
import numbers
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def sales():
    # load dataset
    sales = pd.read_csv('realData\\technion_retail_obfuscated.csv')
    for i, row in sales.iterrows():
        sales.set_value(i, 'higherThanOne', 1 if row['sales'] > 1 else 0)
    sales_copy = sales

    #sales_copy = pd.read_csv('realData\\numeric.csv')

    cols = sales_copy.columns.values.tolist()
    dict = {}
    counters = {}
    for col in cols:
        dict[col] = {}
        counters[col] = 1
    sales_copy = sales_copy.fillna(0)
    for i, row in sales_copy.iterrows():
        print i
        for col in cols:
            if not isinstance(row[col], numbers.Number):
                if not row[col] in dict[col]:
                    dict[col][row[col]] = counters[col]
                    counters[col] = counters[col] + 1
                sales_copy.set_value(i, col, dict[col][row[col]])

    #sales_copy.to_csv('realData\\numeric.csv')

    # create dataframes with an intercept column
    str = "higherThanOne" + " ~ " + " + ".join([x for x in cols if not x in ["higherThanOne", "sales"]])
    conj1 = " + I(calendar_month_of_year > 5) * I(location_type == %d)" % dict['location_type']['High-street']
    conj2 = " + I(preholidaykernel == %d)" % dict['preholidaykernel']['CHRISTMAS-DAY']
    print conj1
    print conj2
    str = str + conj1 + conj2
    print str
    y, X = patsy.dmatrices(str, sales_copy, return_type="dataframe")

    # flatten y into a 1-D array
    y = np.ravel(y)

    # instantiate a logistic regression model, and fit with X and y
    model = LogisticRegression()
    model = model.fit(X, y)

    # check the accuracy on the training set
    print "accuracy: "
    print model.score(X, y)

    # what percentage is higher than one sale?
    print "the percentage of higher than one sale: "
    print y.mean()

    # examine the coefficients
    print "coefficients: "
    print pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))

    # evaluate the model by splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model2 = LogisticRegression()
    model2.fit(X_train, y_train)

    # predict class labels for the test set
    predicted = model2.predict(X_test)
    print "predicted: "
    print predicted

    # generate class probabilities
    probs = model2.predict_proba(X_test)
    print "probabilities: "
    print probs

    print "area under the curve: "
    print roc_auc_score(y, model.predict(X)) #area under the curve
