from suicide_phrase_classifier import SuicidePhraseClassifier
from database import TRAINING_DATABASE

print('Treinando...')

classifier = SuicidePhraseClassifier()
classifier.train(TRAINING_DATABASE)

print('Pronto!')

while True:
  print('\nDigite uma frase (ou apenas enter para fechar o programa):')
  phrase = input().strip()

  if phrase != '':
    result = classifier.is_suicidal(phrase)
    print('Resultado: ' + ('Suicida' if result else 'Não-suicida'))
  else:
    print('Tchau!')
    break


