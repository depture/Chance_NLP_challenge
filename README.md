# Coding Challenge - Data Scientist - NLP

This repository was created to host the challenge for Data Scientist - NLP roles at
Chance.

## Evaluation

This is a non-exaustive list of the aspects that we will consider:

* Code organization
* Problem solving
* Code readability
* Chosen solutions
* Version control practices


## Problem - MBTI prediction

On this task you will have to create a model to predict MBTI personality types
from posts. If you don't know what MBTI is, see [MBTI basics]( 
http://www.myersbriggs.org/my-mbti-personality-type/mbti-basics/home.htm?bhcp=1)

### Details

For this task you can use any language of your choice, but we recommend the use
of some of the following: python, jupyter, scikit-learn. You will have
to :
* Create a local git repository
* Download the dataset for this task [here](
    https://www.kaggle.com/datasnaek/mbti-type)
* Create an NLP machine learning algorithm that determines a person personality type based on a set of posts. You can use the model of your preference.
* Train your model, expose it through an API (function), and describe
    how to access it.

## Usage

1. Start a git repository with ```git init```
1. Do your magic on your local machine, trying to commit often
1. Add at the end of this README a small descriptions about your choices.
1. Run ```git bundle create YOURNAME.bundle HEAD ```
1. Send the generated file by email to tech-interview@chance.co

## Description 

An ipython notebook describe step by step the choice i took to implement the algorithm.
Ii is implemented in Python 3.6.

The main steps are :

1. Treat data
* Load data using pandas
* Split each row by type | comments | urls posted | count youtube videos
* Cleaning each comments by removing numbers and spaces and join them into a single text

2. Learning phase
* Run tf-idf and bag of words vectorization
* Try pca (no improvements, if more data investigate feature reduction to improve speed)
* Try Multinomial Naive Bayes (F1-score ~ 54) [5 fold stratified cross validation]
* Try XGboost (F1-score ~ 64) [5 fold stratified cross validation]
* Save vectorization and models parameters for later usage
3. Application
* The function MBTI\_XGB loads precalculated vectorizer and boosting model to train new data. 


This is a simple model which trains relatively quickly. The cross validation outputed an f1-score of ~64. It is possible to search for hyper paramaters with bayesian optimisation to improve accuracy. Also, a pre-trained word embeddings like glove in a neural model may bring more insight. I only add the count of youtube videos per link in comments. A lot of them are dead links but it would be interesting to scrap the different urls to extract more information.

TO DO:
- Hyper Parameter Optimization
- Convolutional Network with glove embedding
