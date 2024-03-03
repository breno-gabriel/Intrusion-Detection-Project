"""Train and test LSTM classifier"""
from dga_classifier.data import get_data
import numpy as np
# import keras 
# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.layerss import Embedding
# from keras.layers import LSTM
import sklearn.metrics
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from keras import ops


def build_model(max_features, maxlen):
    """Build LSTM model"""
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=max_features, output_dim=128))  # Remova input_length da camada Embedding
    model.add(layers.LSTM(128, input_shape=(maxlen,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')

    return model

def run(max_epoch=25, nfolds=10, batch_size=128):
    """Run train/test on logistic regression model"""
    # indata = data.get_data()

    X, labels = get_data()

    # Extract data and labels
    # X = [x[1] for x in indata]
    # labels = [x[0] for x in indata]

    # Generate a dictionary of valid characters
    valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}

    max_features = len(valid_chars) + 1
    maxlen = np.max([len(x) for x in X])


    # Convert characters to int and pad
    X = [[valid_chars[y] for y in x] for x in X]
    X = keras.preprocessing.sequence.pad_sequences(X, maxlen=maxlen)

    # Convert labels to 0-1
    y = [0 if x == 'benign' else 1 for x in labels]

    final_data = []

    for fold in range(nfolds):
        print ("fold %u/%u" % (fold+1, nfolds))
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels, 
                                                                           test_size=0.05)

        print ('Build model...')
        model = build_model(max_features, maxlen)

        print ("Train...")
        X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05)
        best_iter = -1
        best_auc = 0.0
        out_data = {}

        # Convert lists to NumPy arrays
        X_train = np.array(X_train)
        X_holdout = np.array(X_holdout)
        y_train = np.array(y_train)
        y_holdout = np.array(y_holdout)

        for ep in range(max_epoch):
            model.fit(X_train, y_train, batch_size=batch_size)

            t_probs = model.predict(X_holdout)

            t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs)


            print ('Epoch %d: auc = %f (best=%f)' % (ep, t_auc, best_auc))

            if t_auc > best_auc:
                best_auc = t_auc
                best_iter = ep

                probs = model.predict(X_test)

                out_data = {'y':y_test, 'labels': label_test, 'probs':probs, 'epochs': ep,
                            'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, probs > .5)}

                print(sklearn.metrics.confusion_matrix(y_test, probs > .5))
            else:
                # No longer improving...break and calc statistics
                if (ep-best_iter) > 2:
                    break

        final_data.append(out_data)

    return final_data

