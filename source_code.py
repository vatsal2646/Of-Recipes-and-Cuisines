#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#LOADING TRAINING AND TESTING DATASET
train = pd.read_json('/Users/vatsa/Downloads/Downloads/CS537/train.json')
test = pd.read_json('/Users/vatsa/Downloads/Downloads/CS537/test.json')

#CLEANING AND PREPROCESSING DATA
train['ingredients_clean'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', array)) for array in rows]).strip() for rows in train['ingredients']]
train['ingredients_clean'] = train['ingredients_clean'].str.lower()
test['ingredients_clean'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', array)) for array in rows]).strip() for rows in test['ingredients']]
test['ingredients_clean'] = test['ingredients_clean'].str.lower()

#DATA VISUALIZATION
figure = train.cuisine.value_counts().plot(kind='bar')
figure = figure.get_figure()
figure.savefig('Number of recipes per cuisine.png')
plt.show()
trainingredients = WordCloud(background_color='white', max_words=500, max_font_size=70, relative_scaling=0).generate(str(train['ingredients_clean']))
print(trainingredients)
fig = plt.figure(2)
plt.imshow(trainingredients)
plt.axis('off')
plt.show()
fig.savefig("trainingredients.png", dpi=3000)
traincuisine = WordCloud(background_color='white', max_words=16, max_font_size=70, relative_scaling=0).generate(str(train['cuisine']))
print(traincuisine)
fig = plt.figure(3)
plt.imshow(traincuisine)
plt.axis('off')
plt.show()
fig.savefig("traincuisine.png", dpi=3000)
testingredients = WordCloud(background_color='white', max_words=5000, max_font_size=70, relative_scaling=0).generate(str(test['ingredients_clean']))
print(testingredients)
fig = plt.figure(4)
plt.imshow(testingredients)
plt.axis('off')
plt.show()
fig.savefig("testingredients.png", dpi=3000)

#EXTRACTING FEATURES FROM DATA
train_vector = TfidfVectorizer(stop_words='english', ngram_range = ( 1 , 1 ),analyzer="word", max_df = .57, binary=False , token_pattern=r'\w+', sublinear_tf=False)
train_feat = train['ingredients_clean']
a=train_vector.fit_transform(train_feat).todense()
b = train['cuisine']
test_feat = test['ingredients_clean']
test_vector = TfidfVectorizer(stop_words='english')
c=train_vector.transform(test_feat)

#TRAINING THE MODEL WITH CROSS-VALIDATION
'''
Below is the list of classifiers that were tried before finding out that linear SVC provided the best accuracy among all the other classifiers.
clf = BernoulliNB()
clf = MultinomialNB()
scaler = StandardScaler()
scaler.fit(a_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
a_train = scaler.transform(a_train)
a_test = scaler.transform(a_test)
clf = MLPClassifier(hidden_layer_sizes=(20,20,20),max_iter=500)
clf = LogisticRegression()
'''
a_train, a_test, b_train, b_test  =   train_test_split(a, b, test_size=.2)
clf = LinearSVC()
parameters = {'C':[1, 10, 100, 1000]}
classifier = GridSearchCV(clf, parameters)
classifier=classifier.fit(a_train, b_train)
predictions_train=classifier.predict(a_test)
print(accuracy_score(b_test, predictions_train))
print(confusion_matrix(b_test, predictions_train))
print(classification_report(b_test, predictions_train))

#PLOTTING THE PERFORMANCE OF VARIOUS CLASSIFIERS USING MATPLOTLIB
objects = ('Linear\nSVC', 'Logistic\nRegression', 'Naive Bayes\nBernoulli', 'Multilayer\nPerceptron', 'Naive Bayes\nMultinomial')
y_pos = np.arange(len(objects))
performance = [0.7942, 0.7716, 0.7307, 0.6815, 0.6727]
plt.bar(y_pos, performance, align='center')
plt.xticks(y_pos, objects)
plt.ylabel('Classification Accuracy')
plt.title('Performance of various classifiers')
plt.show()

#GENERATING RESULTS FOR TEST DATASET
predictions=classifier.predict(c)
test['cuisine'] = predictions
test[['id' , 'ingredients' , 'cuisine' ]].to_csv("test_data_classification.csv")
