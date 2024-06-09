import string
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS

class BaseTeste:
    def __init__(self, bd_teste):
        self.bd_base = bd_teste
        self.bd_teste = self.__preprocessar__(bd_teste)

    def __preprocessar__(self, bd_prep):
        bd_prep['tweet_text'] = bd_prep['tweet_text'].apply(self.__preprocessamento__)
        return bd_prep
    
    def __preprocessamento__(self, texto):
        print('Preprocessando base teste...')
        pln = spacy.load('pt_core_news_sm')
        documento = pln(texto.lower())
        lista = []
        for token in documento:
            lista.append(token.lemma_)
        lista = [palavra for palavra in lista if palavra not in STOP_WORDS and palavra not in string.punctuation] #removendo stopwords e pontuação 
        lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()]) #removendo digitos
        return lista