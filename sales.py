import numpy as np
import pandas as pd
import patsy
import re
import math
import numbers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def sales():

    # load dataset
    sales = pd.read_csv('realData\\technion_retail_obfuscated.csv')

    for i, row in sales.iterrows():
        sales.set_value(i, 'higherThanOne', 1 if row['sales'] > 1 else 0)
    cols = sales.columns.values.tolist()

    dict = {}
    counters = {}
    for col in cols:
        dict[col] = {}
        counters[col] = 1
    sales = sales.fillna(-1)
    for i, row in sales.iterrows():
        print i
        for col in cols:
            if not isinstance(row[col], numbers.Number):
                if not row[col] in dict[col]:
                    dict[col][row[col]] = counters[col]
                    counters[col] = counters[col] + 1
                sales.set_value(i, col, dict[col][row[col]])

    print sales

    #for i, row in sales.iterrows():
    #    for col in cols:
    #        if not isinstance(row[col], numbers.Number):
    #            print dict[col][row[col]]
     #       else:
    #            print row[col]
    # create dataframes with an intercept column
    # s = [key for key in dict(sales.dtypes) if dict(sales.dtypes)[key] not in ['float64', 'int64']]
    #sales_with_dummies = pd.get_dummies(sales).fillna(-1)
    # print sales_with_dummies
    #sales_with_dummies.rename(columns=lambda x: re.sub('[\{ | \} | \( | \) | / | \[ | \] | ^]', '_', x), inplace=True)
    #cols = sales_with_dummies.columns.values.tolist()
    #print cols
    # str = "higherThanOne" + " ~ " + " + ".join(cols[1:])
    #str = "higherThanOne" + " ~ " + " + ".join(x for x in cols if not x in ["sales"])
    #print str
    # y, X = patsy.dmatrices(str, sales_with_dummies, return_type="dataframe")

    # flatten y into a 1-D array
    #y = np.ravel(y)

    # instantiate a logistic regression model, and fit with X and y
    #model = LogisticRegression()
    #model = model.fit(X, y)

    # check the accuracy on the training set
    #print model.score(X, y)

    # average rating
    #print y.mean()

    # examine the coefficients
    #print pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))

    # evaluate the model by splitting into train and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #model2 = LogisticRegression()
    #model2.fit(X_train, y_train)

    # predict class labels for the test set
   # predicted = model2.predict(X_test)
    #print predicted

    # generate class probabilities
    #probs = model2.predict_proba(X_test)
    #print probs
