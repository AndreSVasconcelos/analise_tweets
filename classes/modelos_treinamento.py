import spacy
import random
from spacy.training import Example

class ModeloTreinamento:
    def __init__(self, bd_treinamento):
        self.bd_treinamento = bd_treinamento
        self.historico = []
        self.modelo = self.criar_modelo()


    def criar_modelo(self):
        try:
            modelo_carregado = spacy.load("modelo_tweets")
        except:
            new_model = spacy.blank('pt')
            textcat = new_model.add_pipe("textcat")
            textcat.add_label("POSITIVO")
            textcat.add_label("NEGATIVO")
            new_model.begin_training()
            for epoca in range(200):
                random.shuffle(self.bd_treinamento)
                losses = {}
                batches = spacy.util.minibatch(self.bd_treinamento, 30)
                for batch in batches:
                    textos = [new_model(texto) for texto, entities in batch]
                    annotations = [{'cats': entities} for texto, entities in batch]
                    examples = [Example.from_dict(doc, annotation) for doc, annotation in zip(textos, annotations)]
                    new_model.update(examples, losses=losses)
                if epoca % 3 == 0:
                    print(losses)
                    self.historico.append(losses)
            new_model.to_disk("modelo_tweets")
            return new_model
        else:
            return modelo_carregado
            
    
    def modelo_treinado(self, modelo_to_test):
        try:
            modelo_to_test
        except NameError:
            return False
        else:
            return True