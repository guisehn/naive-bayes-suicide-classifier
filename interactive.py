from suicide_phrase_classifier import SuicidePhraseClassifier
from report_generator import ReportGenerator
from database import TRAINING_DATABASE

print('Treinando...')

classifier = SuicidePhraseClassifier()
classifier.train(TRAINING_DATABASE)

test_database = []

print('Pronto!')
while True:
  print('\nDigite uma frase (ou apenas enter para encerrar):')
  phrase = input().strip()

  if phrase != '':
    result = classifier.is_suicidal(phrase)
    print('Resultado: ' + ('Suicida' if result else 'Não-suicida'))

    is_correct = ''
    while is_correct != 'S' and is_correct != 'N':
      print('Estava correto (S/N)? ')
      is_correct = input().strip().upper()

    test_database.append((phrase, result if is_correct == 'S' else not result))
  else:
    if test_database:
      print('\nResultado\n===================================')
      report_generator = ReportGenerator(classifier)
      report = report_generator.generate(test_database)
      print(report)
    break

