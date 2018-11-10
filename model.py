import tensorflow
import keras
import numpy as np
import pandas as pd

df = pd.read_csv('train_E6oV3lV.csv')

count_class_0, count_class_1 = df['label'].value_counts()

df_class_0 =  df.query('label==0')
df_class_1 =  df.query('label==1')
df_class_0_under = df_class_0.sample(count_class_1)
df_under = pd.concat([df_class_0_under, df_class_1],ignore_index=True ,axis=0)

X = df_under['tweet']
Y = df_under['label']
Y_org = Y
X.values

def find_features(tweet):
    rtweet = []
    from keras.preprocessing.text import Tokenizer
    max_features = 10000
    tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', split=' ', lower=True, char_level=False, oov_token=None)
    tokenizer.fit_on_texts(X.values)
    rtweet = tokenizer.texts_to_sequences(tweet.split(' '))
    rtweet = np.array(rtweet)
    rtweet = rtweet.reshape(1,-1)
    
    from keras.preprocessing.sequence import pad_sequences
    rtweet = pad_sequences(rtweet, maxlen=95)
    return rtweet



from keras.models import load_model
new_model = load_model("rnn_classifier.h5")

def ret_probabilities(tweet):
    features = find_features(tweet)
    features = np.reshape(features, (features.shape[0],1, features.shape[1]))
    prob = new_model.predict(features)
    return prob
