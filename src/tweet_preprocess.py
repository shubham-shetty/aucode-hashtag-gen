import pandas as pd
import numpy as np
import re

# Import Tweet processing utils from ekphrasis
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

# Define text preprocessing pipeline
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag"},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

# Preprocess raw tweets using ekphrasis preprocessing pipeline
def preprocess_tweet(tweet):
    tweet_clean = " ".join(text_processor.pre_process_doc(tweet))
    return tweet_clean

# Identify hashtags annotated within processed tweets
def find_hashtags(tweet):
    hashtags = re.findall(r"(?<=<hashtag>)(.+?)(?=</hashtag>)", tweet)
    return hashtags

# Return dataframe containing original tweet, preprocessed tweet, and list of hashtags
# INPUT: Pandas df containing tweets, column name for tweet content
# OUTPUT: df appended with preprocessed tweet and list of hashtags
def get_hashtags(tweet_df, content):
    tweet_df['cleaned_tweet'] = tweet_df[content].map(preprocess_tweet)
    tweet_df['hashtags'] = tweet_df['clean_tweet'].map(find_hashtags)
    return tweet_df