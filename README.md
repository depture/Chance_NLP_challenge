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

### Description 

An ipython notebook describe step by step the choice i took to implement the algorithm.
Ii is implemented in Python 3.6.

The main steps are :

1. Treat data
* Load data using pandas
* Split each row by type | comments | urls posted | count youtube videos
* Cleaning each comments by removing numbers and spaces and join them into a single text
* Run tf-idf and bag of words vectorization
* Add pca 

1. Learning phase
* Try Multinomial Naive Bayes (F1-score ~ 54) [5 fold stratified cross validation]
* Try XGboost (F1-score ~ 64) [5 fold stratified cross validation]
* Save vectorization and models parameters for later usage

It is possible to treat new data with the function MBTI_XGB.py.
