import re
import pymorphy2
import pandas as pd
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from nltk.corpus import stopwords
from razdel import sentenize, tokenize
from threading import Thread

from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,

    PER,
    NamesExtractor,
    Doc)


# текст делим на предложения
# по одному преждложунию грузим в doc
# находим именованные сущности
# убираем именованные чущности
# получение аббревиатур и удаление их из предложения
# чистка от знаков препинания
####на выходе массив именованных сущностей и массив предложений
# затем чистим предложения от пунктуации стоп слов и лемматизируем

class KeyWords(object):
    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        self.names_extractor = NamesExtractor(self.morph_vocab)


def init_doc(self, text):
    doc = Doc(text)
    doc.segment(self.segmenter)
    doc.tag_ner(self.ner_tagger)
    return doc


def get_sentance(self, doc):
    sentences = []

    for sentence in doc.sents:
        sentences.append(sentence.text)
    return sentences


def get_ner(self, doc):
    ner = []
    for span in doc.spans:
        ner.append(span.text)
    return ner


def clean_text_from_ner(self, text, ner):
    for word in ner:
        text = text.replace(word, '')
    return text


def get_abbr(self, text):
    abbr = re.findall(r"\b[0-9]*[-]*[A-ZА-Я]{1,}[a-zа-я]*[A-ZА-Я]{1,}[-]*[a-zA-Zа-яА-Я0-9]*\b", text)
    for word in abbr:
        text = text.replace(word, '')
    return text, abbr


def cleanText(self, text):
    rep = re.compile("[^a-zA-Zа-яА-я]")
    cleanStr = rep.sub(" ", text).lower()
    cleanWord = re.findall(r'\b[а-яa-z]{3,15}\b', cleanStr)
    return ' '.join(cleanWord)


def tokenization(self, text):
    tokens = []

    for sent in sentenize(text):
        tokens = [_.text for _ in tokenize(sent.text)]
    return tokens


def delete_stop_words(self, words):
    stop_words = stopwords.words('russian')
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words


def lemmaPymorphy(self, words):
    morph = pymorphy2.MorphAnalyzer()
    normal_words = []
    for word in words:
        p = morph.parse(word)[0]
    normal_words.append(p.normal_form)
    return normal_words


def filter_noun_adfj(self, words):
    morph = pymorphy2.MorphAnalyzer()
    noun_adjf_words = []
    for word in words:
        p = morph.parse(word)[0]
    if 'NOUN' in p.tag or 'ADJF' in p.tag:
        noun_adjf_words.append(p.normal_form)
    return noun_adjf_words


def pair_words(self, words):
    couple_of_words = list(ngrams(words, 2))
    list_of_pairs = []
    for pair in couple_of_words:
        list_of_pairs.append(' '.join(list(pair)))
    return list_of_pairs


def TFIDF(self, tokens):
    cv = CountVectorizer()

    word_count_vector = cv.fit_transform(tokens)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    tf_idf_data = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])
    # sort ascending
    tf_idf_data = tf_idf_data.sort_values(by=['idf_weights'])
    index = tf_idf_data.index.values.tolist()
    values = tf_idf_data['idf_weights'].values.tolist()
    return index, values


def get_pair(self, pairs):
    frequency = {}
    for word in pairs:
        count = frequency.get(word, 0)
    frequency[word] = count + 1

    frequency_list = frequency.keys()
    new_pairs = []
    for words in frequency_list:
        if frequency[words] > 1:
            new_pairs.append(words)
    return new_pairs


def clear_pair(self, sentences_pair_tokens, pair):
    new_sentences_pair_tokens = []
    for sentence in sentences_pair_tokens:
        new_sentence = []
    for pair_of_token in sentence:
        if pair_of_token in pair:
            new_sentence.append(pair_of_token.replace(' ', '_'))
    new_sentences_pair_tokens.append(' '.join(new_sentence))
    print(new_sentences_pair_tokens)
    return new_sentences_pair_tokens


def sum_all_tokens(self, ner, pair, tokens):
    for i in range(len(tokens)):
        tokens[i] += " " + ner[i] + " " + pair[i]
    print(tokens)
    return tokens


def execute(self, text):
    text_doc = self.init_doc(text)

    sentences = self.get_sentance(text_doc)
    print(sentences)
    clean_sentences = []
    name_entities = []
    sentences_abbr = []
    tokens = []
    sentences_tokens = []
    sentences_pair_tokens = []
    all_pair_tokens = []
    for sentence in sentences:
        sentence_doc = self.init_doc(sentence)
    sentence_ner = self.get_ner(sentence_doc)
    clean_sentence = self.clean_text_from_ner(sentence, sentence_ner)
    clean_sentence, sentence_abbr = self.get_abbr(clean_sentence)
    sentences_abbr.append(sentence_abbr)
    clean_sentence = self.cleanText(clean_sentence)
    sentence_token = self.tokenization(clean_sentence)
    sentence_token = self.delete_stop_words(sentence_token)
    sentence_token = self.lemmaPymorphy(sentence_token)
    sentence_ner = self.lemmaPymorphy(sentence_ner)
    sentence_token = self.delete_stop_words(sentence_token)
    sentence_token = self.filter_noun_adfj(sentence_token)
    sentence_pair_token = self.pair_words(sentence_token)
    # tokens.append(sentence_token)
    sentence_token = ' '.join(sentence_token)
    sentences_tokens.append(sentence_token)
    sentences_pair_tokens.append(sentence_pair_token)
    all_pair_tokens += sentence_pair_token
    # clean_sentences.append(clean_sentence)
    name_entities.append(' '.join(sentence_ner + sentence_abbr))

    pair = self.get_pair(all_pair_tokens)
    sentences_pair_tokens = self.clear_pair(sentences_pair_tokens, pair)
    sentences_tokens = self.sum_all_tokens(name_entities, sentences_pair_tokens, sentences_tokens)
    keys, values = self.TFIDF(sentences_tokens)
    return keys, values


def get_key_words(self, text):
    keys, values = self.execute(text)
    data = zip(keys, values)
    return keys, data