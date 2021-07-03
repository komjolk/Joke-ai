from numpy.lib.function_base import append
import pandas as pd
import numpy as np

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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
# amount of titles
title_filt = title[:300]
all_title = list(title_filt)
all_title[:2]


# amount of bodys, should be same as title
body_filt = body[:300]
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
    label_max = max([len(i.split(' ')) for i in data_labels])
    features_max = max([len(i.split(' ')) for i in data_features])
    return label_max, features_max
label_max, features_max = prep_data(all_title, all_body)
print("tokenized done")
print(f"data length:{label_max, features_max}")



from keras.models import load_model
#the model you wanna use
Quotes_gen = load_model("title_generator_test_extra_over_fit.h5")
# Text generating function
def generate_quote(title, num_words, model, title_max, body_max):
    body = ""
    for _ in range(num_words):
        tokens = (pad_sequences([tokenizer.texts_to_sequences([title])[0]], maxlen=title_max, padding="pre"))
        tokens = np.array([np.append(tokens, pad_sequences([tokenizer.texts_to_sequences([body])[0]], maxlen=body_max)[0])])
        predicted = np.argmax(model.predict(tokens), axis=-1)
        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        body = body + " " + output_word
    return title + "\n" + body

#write the joke title in the quote
print(generate_quote("", num_words = 10, model= Quotes_gen, title_max=features_max, body_max=label_max))

