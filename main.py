import numpy as np
import pandas as pd
import patsy
import re
import math
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
#from sklearn import metrics
#from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import matplotlib

def main():

    # load dataset
    ratings = pd.read_csv('data\\ratings.csv')
    movies = pd.read_csv('data\\movies.csv')

    # movies['rating'] = ratings.query('movieId == @movies.movieId')['rating'].mean()
    # movies['rating'] = (ratings[ratings['movieId'] == movies.movieId])['rating'].mean()
    # movies['ratingCount'] = ratings[ratings['movieId'] == movies.movieId].shape[0]

    geners = ['Comedy', 'Adventure', 'Animation', 'Children', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime',
              'Thriller', 'Documentary', 'Mystery', 'Sci-Fi', 'Musical', 'Horror']

    for i, row in movies.iterrows():
        b = ratings[(ratings['movieId'] == row['movieId'])]
        movies.set_value(i, 'num_rating', b.shape[0])
        # rating = 0 if math.isnan(b['rating'].mean()) else int(b['rating'].mean())
        rating = b['rating'].mean()
        movies.set_value(i, 'rating', rating)

        movies.set_value(i, 'higherThanThree', 1 if rating > 3.0 else 0)

        for g in geners:
            movies.set_value(i, g, int(g in row['genres']))

        s = row['title']
        x = re.search('\(([0-9]{4})\)', s)
        movies.set_value(i, 'Year', 0 if x is None else int(s[x.start()+1:x.end()-1]))

    #print movies.head(100)

    # create dataframes with an intercept column
    y, X = patsy.dmatrices('higherThanThree ~  num_rating + Year + Comedy + Adventure + Animation + Children + Fantasy + \
                       Romance + Drama + Action + Crime + Thriller + Documentary + Mystery + Musical + Horror',
                       movies, return_type="dataframe")

    # flatten y into a 1-D array
    y = np.ravel(y)

    # instantiate a logistic regression model, and fit with X and y
    model = LogisticRegression()
    model = model.fit(X, y)

    # check the accuracy on the training set
    print model.score(X, y)

    # average rating
    print y.mean()

    # examine the coefficients
    pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))

    # evaluate the model by splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model2 = LogisticRegression()
    model2.fit(X_train, y_train)

    # predict class labels for the test set
    predicted = model2.predict(X_test)
    print predicted
    
    # generate class probabilities
    #probs = model2.predict_proba(X_test)
    #print probs

if __name__ == "__main__":
    main()
