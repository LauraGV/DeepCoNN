
# coding: utf-8

# In[ ]:


import json
import pandas as pd
from keras.preprocessing.text import text_to_word_sequence


def get_list_of_dicts(fname): return [json.loads(i) for i in open(fname, "rt")]


def add_user_reviews(x):
    ur = user_reviews.loc[x["reviewerID"]].drop(x["asin"])
    mr = movie_reviews.loc[x["asin"]].drop(x["reviewerID"])
    x["userReviews"] = ur["reviewText"].tolist()
    x["movieReviews"] = mr["reviewText"].tolist()
    return x


def clean(text):
    return text_to_word_sequence(text,
                                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                 lower=True, split=" ")


def clean_one(x):
    return list(map(clean, x[2:-2].split()))


def clean_review_text(x):
    x["reviewText"] = clean_each(x["reviewText"])
    return x


raw_data = get_list_of_dicts("../data/reviews_Amazon_Instant_Video_5.json")

data = pd.DataFrame(raw_data).loc[:,
                                  ["reviewerID",
                                   "reviewText",
                                   "asin",
                                   "overall"]]


# In[ ]:


cleaned_text = data.loc[:, ["reviewerID", "asin", "overall"]]
cleaned_text.loc[:, "reviewText"] = data.loc[:, "reviewText"].apply(clean)


# In[ ]:


cleaned_text.to_csv("../data/cleaned_reviews.csv")

