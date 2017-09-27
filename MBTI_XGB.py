# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import pickle
import re
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def check_remove(FileName):
    if os.path.isfile(FileName):
        os.remove(FileName)

		dataset = '/Users/Dupi/venv_py_3/Chance/Chance_NLP_challenge/data/mbti_1.csv'
def MBTI_XGB(dataset):

	# Load data
	print("Loading data")
	data = pd.read_csv(dataset)  # dtype = {'type': str,'post': str}

	##### Encode each type to an int

	unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
						'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
	lab_encoder = LabelEncoder().fit(unique_type_list)

	##### Compute list of subject with Type | list of comments | list of url | count youtube videos

	list_subject = []

	for row in data.iterrows():
		list_comment = []
		list_url = []
		list_youtube = []
		posts = row[1].posts
		for post in posts.split("|||"):
			urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', post)
			if urls:
				post = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', post)
				list_url += urls
			if any(post):
				list_comment.append(post)
		list_youtube += [sum([1 for s in list_url if "youtube" in s])]

		type_labelized = lab_encoder.transform([row[1].type])[0]
		list_subject.append([type_labelized, list_comment, list_url, list_youtube])

	del data
	subject_type = np.array([subject[0] for subject in list_subject])
	# subject_comments = ["".join(subject[1]) for subject in list_subject]

	##### Remove and clean comments

	# Remove numbers
	subject_comments_1 = [re.sub("[^a-zA-Z]", " ", " ".join(sentence[1])).split(' ') for sentence in list_subject]

	# Remove spaces
	subject_comments_1 = [re.sub(' +', ' ', " ".join(comments)) for comments in subject_comments_1]

	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import TfidfVectorizer


	##### Vectorize with vectors in memory
	print("Vectorizing words")

	tfVec = pickle.load(open("data/tfidf.pickle.dat", "rb"))
	X_tfidf = tfVec.transform(subject_comments_1).todense()

	CountVect = pickle.load(open("data/CountVect.pickle.dat", "rb"))
	X_vectorized = CountVect.transform(subject_comments_1).toarray()

	count_youtube = [c[3][0] for c in list_subject]
	X_concat = np.column_stack((X_tfidf, X_vectorized, count_youtube))

	# load model from file
	loaded_model = pickle.load(open("data/model.pickle.dat", "rb"))

	# Confusion plot

	def plot_confusion_matrix(cm, classes,
							  normalize=False,
							  title='Confusion matrix',
							  cmap=plt.cm.Blues):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
		else:
			print('Confusion matrix, without normalization')

		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')


	# XGboost

	xg_test = xgb.DMatrix(X_concat)

	# make predictions for test data
	preds = loaded_model.predict(xg_test)

	if any(subject_type):
		print(classification_report(subject_type, preds))

		# get prediction
		error_rate = np.sum(preds != subject_type) / subject_type.shape[0]
		print('Test error using softmax = {}'.format(error_rate))

		# Compute confusion matrix
		cnf_matrix = confusion_matrix(subject_type, preds)
		np.set_printoptions(precision=2)

		# Plot normalized confusion matrix
		plt.figure()
		plot_confusion_matrix(cnf_matrix, classes=lab_encoder.inverse_transform(range(16)), normalize=True,
							  title='Confusion matrix')

	print("\nSaving preds in data/MBTI_XGB_predictions.txt")

	check_remove('data/MBTI_XGB_predictions.txt')
	with open('data/MBTI_XGB_predictions.txt', 'a') as outfile:
		outfile.write("\n".join(lab_encoder.inverse_transform(preds.astype(int))))


if __name__ == '__main__':
	MBTI_XGB("data/mbti_1.csv")


