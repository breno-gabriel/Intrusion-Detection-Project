"""Train and test bigram classifier"""
from dga_classifier.other_data import generate_database
# from keras.layers.core import Dense
from keras.models import Sequential
import sklearn
import numpy as np 
from sklearn import feature_extraction
# from sklearn.cross_validation import train_test_split
import sklearn.metrics
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from keras import ops


def build_model(max_features):
    """Builds logistic regression model"""
    model = Sequential()
    model.add(layers.Dense(1, input_dim=max_features, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam')

    return model


def run(max_epoch=50, nfolds=10, batch_size=128):
    """Run train/test on logistic regression model"""
    X, labels = generate_database()

    # print(len(X))
    # print(len(labels))

    # Extract data and labels
    # X = [x[1] for x in indata]
    # labels = [x[0] for x in indata]

    # Create feature vectors
    print ("vectorizing data")
    ngram_vectorizer = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2, 2))
    count_vec = ngram_vectorizer.fit_transform(X)

    max_features = count_vec.shape[1]

    # Convert labels to 0-1
    y = [0 if x == 'benigno' else 1 for x in labels]

    final_data = []

    for fold in range(nfolds):
        print ("fold %u/%u" % (fold+1, nfolds))
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(count_vec, y,
                                                                           labels, test_size=0.05)

        print ('Build model...')
        model = build_model(max_features)

        print ("Train...")
        X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05)
        best_iter = -1
        best_auc = 0.0
        out_data = {}

        X_train = X_train.toarray()  # Converta a matriz esparsa para uma matriz densa
        y_train = np.array(y_train)

        for ep in range(max_epoch):
            model.fit(X_train, y_train, batch_size=batch_size)

            t_probs = model.predict(X_holdout.todense())
            t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs)

            print ('Epoch %d: auc = %f (best=%f)' % (ep, t_auc, best_auc))

            if t_auc > best_auc:
                best_auc = t_auc
                best_iter = ep

                probs = model.predict(X_test.todense())

                out_data = {'y':y_test, 'labels': label_test, 'probs':probs, 'epochs': ep,
                            'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, probs > .5)}

                print( sklearn.metrics.confusion_matrix(y_test, probs > .5))
            else:
                # No longer improving...break and calc statistics
                if (ep-best_iter) > 5:
                    break

        final_data.append(out_data)

    return final_data
