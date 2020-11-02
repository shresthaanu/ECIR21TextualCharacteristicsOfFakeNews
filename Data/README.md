# How to download PolitiFact and GossipCop datasets

In order to download Politifact and GossipCop datasets, we used FakeNewsNet repository https://github.com/KaiDMML/FakeNewsNet.
This repository contains codes to download news articles from published websites and relevant social media data from Twitter. Following are the stepwise process to use this repository,

Step 1:

  Download scripts from FakeNewsNet repository

Step 2:

  As mentioned in the Requirements part of the FakeNewsNet repository, the minimalistic files to collect dataset is provided in dataset folder that contains news ids, URL, News Headline and tweet ids. 
  
Step 3:

  Install all the libraries in requirements.txt using the given command in Requirements part of the FakeNewsNet repository. The script use twitter API keys that should be updated in tweet_keys_file.json file to extract social media data from twitter. However, for our analysis we have not considered social media data. Thus, it is not required to update those API keys. 

Step 4:

  Since we are not using social media data, the code to collect certain features of the dataset should be updated. In other words, as mentioned in Configuration part the FakeNewsNet repository, the config.json file should be updated to extract only news articles by updating data_features_to_collect with this line of code, 
  
  "data_features_to_collect" : ["news_articles"]


