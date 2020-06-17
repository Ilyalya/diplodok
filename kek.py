import PySimpleGUI as sg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log
import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import pickle

def save(obj):
    with open('sis.pickle', 'wb') as f:
        pickle.dump(obj, f)

def load():
    with open('sis.pickle', 'rb') as f:
        obj_new = pickle.load(f)
    return obj_new

def show_spam(spam_words):
    spam_wc = WordCloud(width = 512,height = 512).generate(spam_words)
    plt.figure(figsize = (10, 8), facecolor = 'k')
    plt.imshow(spam_wc)
    plt.axis('off')
    plt.tight_layout(pad = 0)
    plt.show()

def show_ham(ham_words):
    ham_wc = WordCloud(width = 512,height = 512).generate(ham_words)
    plt.figure(figsize = (10, 8), facecolor = 'k')
    plt.imshow(ham_wc)
    plt.axis('off')
    plt.tight_layout(pad = 0)
    plt.show()

oldmails = pd.read_csv('spam.csv', encoding = 'latin-1')
oldmails.head()

mailz = pd.read_csv('messages.csv', encoding = 'latin-1')
mailz.head()

oldmails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
oldmails.head()

mailz.drop(['subject'], axis = 1, inplace = True)
mailz.head()

oldmails.rename(columns = {'v1': 'labels', 'v2': 'message'}, inplace = True)
oldmails.head()

oldmails['labels'].value_counts()

mailz['label'].value_counts()

oldmails['label'] = oldmails['labels'].map({'ham': 0, 'spam': 1})
oldmails.head()

oldmails.drop(['labels'], axis = 1, inplace = True)
oldmails.head()

mails = pd.concat((mailz, oldmails), ignore_index=True)

totalMails = (int(len(mails))-1)
trainIndex, testIndex = list(), list()
for i in range(mails.shape[0]):
    if np.random.uniform(0, 1) < 0.75:
        trainIndex += [i]
    else:
        testIndex += [i]
trainData = mails.loc[trainIndex]
testData = mails.loc[testIndex]

trainData.reset_index(inplace = True)
trainData.drop(['index'], axis = 1, inplace = True)
trainData.head()

testData.reset_index(inplace = True)
testData.drop(['index'], axis = 1, inplace = True)
testData.head()

trainData['label'].value_counts()

testData['label'].value_counts()

spam_words = ' '.join(list(mails[mails['label'] == 1]['message']))
ham_words = ' '.join(list(mails[mails['label'] == 0]['message']))

trainData.head()

trainData['label'].value_counts()

testData.head()

testData['label'].value_counts()

def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
        print(words)
    return words


class SpamClassifier(object):
    def __init__(self, trainData, method='tf-idf'):
        self.mails, self.labels = trainData['message'], trainData['label']
        self.method = method

    def train(self):
        self.calc_TF_and_IDF()
        if self.method == 'tf-idf':
            self.calc_TF_IDF()
        else:
            self.calc_prob()

    def calc_prob(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word] + 1) / (self.spam_words + \
                                                               len(list(self.tf_spam.keys())))
        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word] + 1) / (self.ham_words + \
                                                             len(list(self.tf_ham.keys())))
        self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails

    def calc_TF_and_IDF(self):
        noOfMessages = self.mails.shape[0]
        self.spam_mails, self.ham_mails = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_mails = self.spam_mails + self.ham_mails
        self.spam_words = 0
        self.ham_words = 0
        self.tf_spam = dict()
        self.tf_ham = dict()
        self.idf_spam = dict()
        self.idf_ham = dict()
        for i in range(noOfMessages):
            message_processed = process_message(self.mails[i])
            count = list()  # To keep track of whether the word has ocured in the message or not.
            # For IDF
            for word in message_processed:
                if self.labels[i]:
                    self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                    self.spam_words += 1
                else:
                    self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                    self.ham_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels[i]:
                    self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                else:
                    self.idf_ham[word] = self.idf_ham.get(word, 0) + 1

    def calc_TF_IDF(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        self.sum_tf_idf_spam = 0
        self.sum_tf_idf_ham = 0
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word]) * log((self.spam_mails + self.ham_mails) \
                                                              / (self.idf_spam[word] + self.idf_ham.get(word, 0)))
            self.sum_tf_idf_spam += self.prob_spam[word]
        for word in self.tf_spam:
            self.prob_spam[word] = (self.prob_spam[word] + 1) / (
                        self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))

        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word]) * log((self.spam_mails + self.ham_mails) \
                                                            / (self.idf_spam.get(word, 0) + self.idf_ham[word]))
            self.sum_tf_idf_ham += self.prob_ham[word]
        for word in self.tf_ham:
            self.prob_ham[word] = (self.prob_ham[word] + 1) / (self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))

        self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails

    def classify(self, processed_message):
        pSpam, pHam = 0, 0
        for word in processed_message:
            if word in self.prob_spam:
                pSpam += log(self.prob_spam[word])
            else:
                if self.method == 'tf-idf':
                    pSpam -= log(self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
                else:
                    pSpam -= log(self.spam_words + len(list(self.prob_spam.keys())))
            if word in self.prob_ham:
                pHam += log(self.prob_ham[word])
            else:
                if self.method == 'tf-idf':
                    pHam -= log(self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))
                else:
                    pHam -= log(self.ham_words + len(list(self.prob_ham.keys())))
            pSpam += log(self.prob_spam_mail)
            pHam += log(self.prob_ham_mail)
        return pSpam >= pHam

    def predict(self, testData):
        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message))
        return result

def metrics(labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels[i] == 1 and predictions[i] == 1)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 1)
        false_neg += int(labels[i] == 1 and predictions[i] == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    return precision, recall, Fscore, accuracy


df = mails

df['message'] = df.message.map(lambda x: x.lower())
df['message'] = df.message.str.replace('[^\w\s]', '')
df['message'] = df['message'].apply(nltk.word_tokenize)

stemmer = PorterStemmer()

df['message'] = df['message'].apply(lambda x: [stemmer.stem(y) for y in x])

df['message'] = df['message'].apply(lambda x: ' '.join(x))

count_vect = CountVectorizer()
counts = count_vect.fit_transform(df['message'])

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)

X_train, X_test, y_train, y_test = train_test_split(counts, df['label'], test_size=0.1, random_state=69)

model = MultinomialNB().fit(X_train, y_train)

predicted = model.predict(X_test)

layout = [
    [sg.Button('Обучение'), sg.Button('Показать спам слова'), sg.Button('Показать не спам слова')],
    [sg.Text('Введите сообщение для проверки на спамовость')],
    [sg.Input(size=(50, 30), key='-IN-')],
    [sg.Button('Проверить'), sg.Button('Выход'), sg.Button('Посчитать метрики')]
    ]

window = sg.Window('Настройка классификатора', layout)
while True:
    # The Event Loop
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Выход':
        break
    if event == 'Посчитать метрики':
        sc_tf_idf = load()
        preds_tf_idf = sc_tf_idf.predict(testData['message'])
        precision, recall, Fscore, accuracy = metrics(testData['label'], preds_tf_idf)
        sg.popup('Метрики',
                 "Точность:", precision,
                 "Полнота:", recall,
                 "F-мера:", Fscore,
                 "Численная оценка качества алгоритма:", accuracy,
                 "Точность классификации для тестового набора данных:", np.mean(predicted == y_test),
                 "Размер тестовой выборки:", len(y_test),
                 "Количество легитимных писем попавших в не спам:", confusion_matrix(y_test, predicted)[0][0],
                 "Количество легитимных писем попавших в спам:", confusion_matrix(y_test, predicted)[0][1],
                 "Количество спам писем попавших в не спам:", confusion_matrix(y_test, predicted)[1][0],
                 "Количество спам писем попавших в спам:", confusion_matrix(y_test, predicted)[1][1])
    if event == 'Проверить':
        text_input = values['-IN-']
        pm = process_message(text_input)
        sc_tf_idf = load()
        sc_tf_idf.classify(pm)
        if sc_tf_idf.classify(pm) == True:
            sg.popup('Спам')
        else:
            sg.popup('Не спам')
    if event == 'Показать спам слова':
        show_spam(spam_words)
    if event == 'Показать не спам слова':
        show_ham(ham_words)
    if event == 'Обучение':
        sc_tf_idf = SpamClassifier(trainData, 'tf-idf')
        sc_tf_idf.train()
        save(sc_tf_idf)
window.close()