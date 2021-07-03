from numpy.lib.function_base import append
import pandas as pd
import numpy as np

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.models import Sequential
import keras.utils as ku

data = pd.read_json("reddit_jokes.json")
print(data.shape)
data.head()

# Dropping duplicates and creating a list containing all the title
title = data['title'].drop_duplicates()
print(f"Total Unique title: {title.shape}")

body = data['body'].drop_duplicates()
print(f"Total Unique title: {title.shape}")
title = list(title)
body = list(body)
x = 0

for i in range(len(body) - 1, -1, -1):
    if (len(body[i].split(' ')) > 10 or len(title[i].split(' ')) > 10):
        del body[i]
        del title[i]
# Considering only top 3000 title
print(len(title))
title_filt = title[:3000]
all_title = list(title_filt)
all_title[:2]


# Considering only top 3000 bodys
body_filt = body[:3000]
all_body = list(body_filt)
all_body[:2]
print( max([len(i.split(' ')) for i in all_body]))
# Tokeinization
tokenizer = Tokenizer()

def prep_data(data_features, data_labels):
    tokenizer.fit_on_texts(data_features)
    tokenizer.fit_on_texts(data_labels)
    total_words = len(tokenizer.word_index) + 1
    print(f"Total unique words in the text corpus: {total_words}")
    tokenized_data = []
    label_max = max([len(i.split(' ')) for i in data_labels])
    features_max = max([len(i.split(' ')) for i in data_features])
    for x in range(len(data_features)):
            print(x)
            seq = tokenizer.texts_to_sequences([data_labels[x]])[0]
            for i in range(1, len(seq)):
                ngram_seq = seq[:i]
                array = (pad_sequences([tokenizer.texts_to_sequences([data_features[x]])[0]], maxlen=features_max, padding="pre"))
                array = np.append(array, pad_sequences([ngram_seq], maxlen=label_max+1)[0])
                tokenized_data.append(array.tolist())
    return  tokenized_data, total_words, label_max, features_max
tokenized_data, total_words, label_max, features_max = prep_data(all_title, all_body)
print("tokenized done")
print(f"data length:{label_max, features_max}")
# Generating predictors and labels from the padded sequences
def generate_input_sequence(input_sequences):
    maxlen = max([len(x) for x in input_sequences])
    predictors, label = [item[:-1] for item in input_sequences], [item[-1] for item in input_sequences]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, maxlen

predictors, label, maxlen = generate_input_sequence(tokenized_data)
print("data loading done")
predictors, label = np.array(predictors), np.array(label)

# Building the model
embedding_dim = 64

def create_model(maxlen, embedding_dim, total_words):
    model = Sequential()
    model.add(layers.Embedding(total_words, embedding_dim, input_length = maxlen - 1))
    model.add(layers.LSTM(128, dropout=0.2))
    model.add(layers.Dense(total_words, activation='softmax'))
    
    # compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

model = create_model(maxlen, embedding_dim, total_words)
model.summary()

# Training the model
model.fit(predictors, label, epochs=5000, batch_size=64)
# Save the model for later use
model.save("title_generator_test_super_over_fit.h5")

# Text generating function
def generate_quote(seed_text, num_words, model, maxlen):
    
    for _ in range(num_words):
        tokens = tokenizer.texts_to_sequences([seed_text])[0]
        tokens = pad_sequences([tokens], maxlen=maxlen, padding='pre')
        
        predicted = model.predict_classes(tokens)
        
        output_word = ''
        
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text = seed_text + " " + output_word
    
    return seed_text