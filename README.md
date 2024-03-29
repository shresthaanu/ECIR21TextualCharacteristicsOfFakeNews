# Analysis of News Title and Body to Detect Fake News

This repository contains the code for the paper 
"Textual Characteristics of News Title and Bodyto Detect Fake News: A Reproducibility Study"


# Data Sources
BuzzFeedNews dataset is available at https://zenodo.org/record/1239675#.X5riw0JKgXA (Download Article.zip)

FakeNewsNet (Politifact and Gossipcop) dataset can be downloaded using the code provided at https://github.com/KaiDMML/FakeNewsNet

For the details about how to download the dataset that we have used in paper, please follow the instructions provided here https://github.com/shresthaanu/ECIR21TextualCharacteristicsOfFakeNews/tree/main/Data

# Content

* Script to generate LIWC features : ```step1_liwc_features.ipynb```
* Script to generate complexity, stylistic and emotion features : ```step2_extract_complexity_emotion_and_stylistic_features.ipynb```
* Script to merge all features including LIWC and remaining features (complexity, stylistci, emotion) : ```step3_combine_LIWC_and_remaining_features.ipynb```
* Script for statistical test (one way ANOVA and Wilcoxon rank sum test) : ```step4_statistical_test.ipynb```
* Script for performing classification and reproduce all the results of paper : ```step5_classification.ipynb```
* Helper files: ```emotion_features.py```, ```liwc_features.py```, ```redability.py```, ```statistical_test.py```


# Citation
If you use our code please cite our work.

```{ 
@inproceedings{shrestha2021textual,
  title={Textual characteristics of news title and body to detect fake news: a reproducibility study},
  author={Shrestha, Anu and Spezzano, Francesca},
  booktitle={European Conference on Information Retrieval},
  pages={120--133},
  year={2021},
  organization={Springer}
  }
