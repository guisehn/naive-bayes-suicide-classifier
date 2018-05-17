import nltk
import re
import operator
from functools import reduce

class SuicidePhraseClassifier:
  def __init__(self):
    self.classifier = None
    self.load_stopwords()

  def load_stopwords(self):
    self.stopwords = nltk.corpus.stopwords.words('portuguese')
    self.stopwords.append('vou')

  def split_words(self, phrase):
    phrase = phrase.strip()
    phrase = re.sub(r'\s+', ' ', phrase)
    return phrase.split(' ')

  def remove_stopwords(self, words):
    cleaned = [word for word in words if word not in self.stopwords]
    return cleaned

  def apply_stemmer(self, words):
    stemmer = nltk.stem.RSLPStemmer()
    cleaned = [stemmer.stem(word) for word in words]
    return cleaned

  def prepare_phrase(self, phrase):
    words = self.split_words(phrase)
    words = self.remove_stopwords(words)
    words = self.apply_stemmer(words)
    return words

  def get_word_frequency_list(self, database):
    wordlist = [item[0] for item in database]
    wordlist = reduce(operator.add, wordlist)
    frequency_list = nltk.FreqDist(wordlist)
    return frequency_list

  def get_word_list(self, frequency_list):
    return frequency_list.keys()

  def extract_features(self, phrase_words):
    return {word: (word in phrase_words) for word in self.database_word_list}

  def train(self, database):
    database = [(self.prepare_phrase(item[0]), str(item[1])) for item in database]

    frequency_list = self.get_word_frequency_list(database)
    self.database_word_list = self.get_word_list(frequency_list)

    nltk_features = nltk.classify.apply_features(self.extract_features, database)
    self.classifier = nltk.NaiveBayesClassifier.train(nltk_features)

  def is_suicidal(self, phrase):
    if self.classifier == None:
      raise Exception("Suicide phrase classifier is not yet trained")
    words = self.prepare_phrase(phrase)
    features = self.extract_features(words)
    return self.classifier.classify(features) == 'True'
