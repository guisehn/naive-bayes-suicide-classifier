# Naive Bayes Suicide Classifier

Project for Special Topics in Computing Module (2018/1) - University of Santa Cruz do Sul (UNISC)

This program attempts to classify whether or not a given phrase has suicidal bias using a Naive Bayes
classifier with the NLTK module.

This is only an experiment for testing data mining in natural language and may not be suitable for real life usage.

It it trained and adapted for the Portuguese language only as this is a homework for a brazilian university.

## Requirements

- [Python 3](https://www.python.org/download/releases/3.0/)
- [Natural Language tooklkit (NLTK)](https://www.nltk.org/)

## How to run

There are two ways to run the program:

- Use `python3 main.py` to test the classifier with phrases inserted directly into the code (edit `main.py` with the phrases to be tested before running)
- Use `python3 interactive.py` if you want to test the phrases using an interactive command line interface

## Interactive example

```
$ python3 interactive.py
Treinando...
Pronto!

Digite uma frase (ou apenas enter para encerrar):
Nada faz sentido em minha vida, quero acabar logo com isso
Resultado: Suicida
Estava correto (S/N)?
S

Digite uma frase (ou apenas enter para encerrar):
Nunca me senti tão pleno como agora
Resultado: Não-suicida
Estava correto (S/N)?
S

Digite uma frase (ou apenas enter para encerrar):
Foi bom te conhecer amigo, até logo
Resultado: Suicida
Estava correto (S/N)?
S

Digite uma frase (ou apenas enter para encerrar):
Sexta-feira é dia de tomar uma gelada
Resultado: Não-suicida
Estava correto (S/N)?
N

Digite uma frase (ou apenas enter para encerrar):


Resultado
===================================
Acurácia: 75.0%

Matriz de confusão:
            |   N |
            |   ã |
            |   o |
            |   - |
            | S s |
            | u u |
            | i i |
            | c c |
            | i i |
            | d d |
            | a a |
------------+-----+
    Suicida |<2>1 |
Não-suicida | .<1>|
------------+-----+
(row = reference; col = test)

Falhas na predição:

Frase: Sexta-feira é dia de tomar uma gelada
Resultado do classificador: Não-suicida
Esperado na base de testes: Suicida
```
