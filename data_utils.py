import traceback
import pandas as pd
import tweepy as tw
import numpy as np
import os

import preprocessor as p
p.set_options(p.OPT.URL, p.OPT.EMOJI)



'''
Preprocessing Function.
Preprocesses raw dataset in place
Args:
    data (pd.DataFrame): Dataframe for raw data collected from dataset source, using shared-format column names
'''
def preprocess(data):
    p.set_options(p.OPT.URL, p.OPT.EMOJI)
    
    # Preprocess Tweet Text
    if "tweet_text" in data.columns:
        data["tweet_text"] = data["tweet_text"].apply(p.clean)
        
    # Preprocess Tweet Label (0 for REAL, 1 for FAKE)
    if "tweet_label" in data.columns:
        def label_preprocess(label):
            if str(label).lower() == "real":
                return 0.0
            if str(label).lower() == "fake":
                return 1.0
            return label
        data["tweet_label"] = data["tweet_label"].apply(label_preprocess)






'''
Utilities for downloading tweet content from tweet ID
'''
class TwitterRetriever:
    API_KEY = ""
    API_KEY_SECRET = ""
    BEARER_TOKEN = ""
    ACCESS_TOKEN = ""
    ACCESS_TOKEN_SECRET = ""

    def __init__(self):
        self.client = tw.Client(TwitterRetriever.BEARER_TOKEN, TwitterRetriever.API_KEY, TwitterRetriever.API_KEY_SECRET, TwitterRetriever.ACCESS_TOKEN, TwitterRetriever.ACCESS_TOKEN_SECRET, wait_on_rate_limit=True)
    
    def get_tweet_info(self, id):
        text = ""
        profile_id = ""
        profile_name = ""
        profile_username = ""
        profile_description = ""
        profile_image = ""
        
        response = self.client.get_tweet(id, expansions=["author_id"], user_fields=["id", "name", "username", "profile_image_url", "description", "public_metrics"])
        
        if response.data is None:
            # Ignorable error
            if response.errors is not None:
                if response.errors[0]["title"] == "Authorization Error":
                    print("Skip inaccessible tweet")
                    return None
                if response.errors[0]["title"] == "Not Found Error":
                    print("Skip ID not found")
                    return None
            
            print(response)
            raise ValueError(response)
        
        user = response.includes['users'][0]
        return {
            "tweet_text": response.data.text,
            "profile_id": user.id,
            "profile_name": user.name,
            "profile_username": user.username,
            "profile_description": user.description,
            "profile_image": user.profile_image_url
        }