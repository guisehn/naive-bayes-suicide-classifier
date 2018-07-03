import nltk
import re
import operator
import unicodedata
from functools import reduce

class SuicidePhraseClassifier:
  def __init__(self):
    self.classifier = None
    self.load_stopwords()

  def load_stopwords(self):
    self.stopwords = nltk.corpus.stopwords.words('portuguese')
    self.stopwords.append('vou')

  def clean_phrase(self, phrase):
    phrase = phrase.lower().replace('-', '')
    return phrase

  def remove_accents(self, phrase):
    phrase = unicodedata.normalize('NFD', phrase)
    phrase = phrase.encode('ascii', 'ignore')
    phrase = phrase.decode('utf-8')
    return str(phrase)

  def remove_extra_spaces(self, phrase):
    phrase = phrase.strip()
    phrase = re.sub(r'\s+', ' ', phrase)
    return phrase

  def glue_related_words(self, phrase):
    phrase = phrase.replace('me matar', 'mematar')
    phrase = phrase.replace('te matar', 'matalo')
    phrase = phrase.replace('matar voce', 'matalo')
    phrase = phrase.replace('nunca mais', 'nuncamais')
    phrase = phrase.replace('para sempre', 'parasempre')
    phrase = phrase.replace('não ', 'não') # junta o não com a palavra para criar uma associação
    phrase = phrase.replace('quero viver', 'queroviver')
    return phrase

  def split_words(self, phrase):
    return phrase.split(' ')

  def remove_stopwords(self, words):
    cleaned = [word for word in words if word not in self.stopwords]
    return cleaned

  def apply_stemmer(self, words):
    stemmer = nltk.stem.RSLPStemmer()
    cleaned = [stemmer.stem(word) for word in words]
    return cleaned

  def prepare_phrase(self, phrase):
    phrase = self.clean_phrase(phrase)
    phrase = self.remove_accents(phrase)
    phrase = self.remove_extra_spaces(phrase)
    phrase = self.glue_related_words(phrase)
    words = self.split_words(phrase)
    words = self.remove_stopwords(words)
    words = self.apply_stemmer(words)
    #print(words)
    return words

  def get_word_frequency_list(self, database):
    wordlist = [item[0] for item in database]
    wordlist = reduce(operator.add, wordlist)
    frequency_list = nltk.FreqDist(wordlist)
    return frequency_list

  def get_word_list(self, frequency_list):
    return frequency_list.keys()

  def extract_features(self, phrase_words, word_list):
    return {word: (word in phrase_words) for word in word_list}

  def apply_features(self, database):
    database = [(self.prepare_phrase(item[0]), str(item[1])) for item in database]
    frequency_list = self.get_word_frequency_list(database)
    database_word_list = self.get_word_list(frequency_list)

    extract_features = lambda words: self.extract_features(words, database_word_list)
    nltk_features = nltk.classify.apply_features(extract_features, database)

    return { 'word_list': database_word_list, 'nltk_features': nltk_features }

  def train(self, database):
    result = self.apply_features(database)

    self.database_word_list = result['word_list']
    self.classifier = nltk.NaiveBayesClassifier.train(result['nltk_features'])

  def calculate_accuracy(self, test_database):
    if self.classifier == None:
      raise Exception("Suicide phrase classifier is not yet trained")

    test_database = [(self.prepare_phrase(item[0]), str(item[1])) for item in test_database]

    extract_features = lambda words: self.extract_features(words, self.database_word_list)
    test_database_features = nltk.classify.apply_features(extract_features, test_database)

    return nltk.classify.accuracy(self.classifier, test_database_features)

  def is_suicidal(self, phrase):
    if self.classifier == None:
      raise Exception("Suicide phrase classifier is not yet trained")
    words = self.prepare_phrase(phrase)
    features = self.extract_features(words, self.database_word_list)
    return self.classifier.classify(features) == 'True'
