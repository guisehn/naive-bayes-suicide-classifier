class ReportGenerator:
  def __init__(self, classifier):
    self.classifier = classifier

  def generate(self, test_database):
    lines = []

    accuracy = self.classifier.calculate_accuracy(test_database)
    lines.append('Acurácia: ' + str(accuracy * 100) + '%\n')

    confusion_matrix = self.classifier.mount_confusion_matrix(test_database, 'Suicida', 'Não-suicida')
    lines.append('Matriz de confusão:\n' + str(confusion_matrix))

    if accuracy < 1:
      lines.append('Falhas na predição:')

      for (phrase, expected_class) in test_database:
        result = self.classifier.is_suicidal(phrase)
        if result != expected_class:
          lines.append('\nFrase: ' + phrase)
          lines.append('Resultado do classificador: ' + ('Suicida' if result else 'Não-suicida'))
          lines.append('Esperado na base de testes: ' + ('Suicida' if expected_class else 'Não-suicida'))

    return '\n'.join(lines)
