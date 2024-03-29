from suicide_phrase_classifier import SuicidePhraseClassifier
from report_generator import ReportGenerator
from database import TRAINING_DATABASE

print('Treinando...')

classifier = SuicidePhraseClassifier()
classifier.train(TRAINING_DATABASE)

print('Pronto!')

# True = suicida, False = não-suicida
TEST_DATABASE = [
  ('estou planejando me matar amanhã', True),
  ('só queria sumir para sempre', True),
  ('vou tomar uma cartela inteira de remédio para acabar logo com isso', True),
  ('nada faz sentido nessa vida vazia', True),
  ('será que se eu me entupir de paracetamol eu consigo sumir desse planeta?', True),
  ('quero me jogar do nono andar', True),
  ('vou me matar', True),
  ('vou te matar e depois cometer suicídio', True),
  ('nada mais faz sentido na vida, me sinto um inútil', True),
  ('queria dormir e nunca mais acordar', True),
  ('quero morrer', True),
  ('foi bom conhece-lo amigo, até mais', True),
  ('eu não consigo mais aguentar tudo isso', True),
  ('eu não pertenço mais a este mundo', True),
  ('a vida não tem mais graça', True),
  ('me sinto muito sozinho', True),
  ('eu choro todo o dia pois minha vida é horrível', True),

  ('vou te matar', False), # intenção dessa frase é mostrar o tratamento feito na ordem das palavras
  ('me sinto muito bem', False),
  ('estou planejando viajar para o nordeste', False),
  ('a economia do brasil vai mal', False),
  ('o neymar é muito cai cai', False),
  ('me ferrei na prova porque estudei pouco, espero que sirva de lição para mim mesmo no futuro', False),
  ('amanhã eu vou jogar futebol com meus amigos, nunca mais joguei com eles', False),
  ('no dia do lixo eu me entupi de hamburguer coca cola', False),
  ('sexta-feira é dia de tomar uma gelada', False),
  ('adoro ficar no meu quarto olhando televisão', False),
  ('quero viver', False),
  ('eu te amo muito, quero viver com você pra sempre', False),
  ('pombos gostam de se alimentar na sombra das arvores', False),
  ('raposas são criaturas assustadoras', False),
  ('tenho medo de levantar sozinho pra tomar água no meio da noite', False)
]

print('\nResultado\n===================================')
report_generator = ReportGenerator(classifier)
report = report_generator.generate(TEST_DATABASE)
print(report)
