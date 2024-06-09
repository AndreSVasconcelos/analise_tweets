import string
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS

class BaseTreinamento:
    def __init__(self, bd_treinamento):
        self.bd_preprocessado = self.__preprocessar__(bd_treinamento)
        self.bd_treinamento = self.__tratar_categorias__()

    def __preprocessar__(self, bd_prep):
        bd_prep['tweet_text'] = bd_prep['tweet_text'].apply(self.__preprocessamento__)
        return bd_prep
    
    def __preprocessamento__(self, texto):
        print('Preprocessando base treinamento...')
        pln = spacy.load('pt_core_news_sm')
        documento = pln(texto.lower())
        lista = []
        for token in documento:
            lista.append(token.lemma_)
        lista = [palavra for palavra in lista if palavra not in STOP_WORDS and palavra not in string.punctuation] #removendo stopwords e pontuação 
        lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()]) #removendo digitos
        return lista
    
    def __tratar_categorias__(self):
        print('Tratando categorias...')
        return_list = []
        for tweet_text, sentiment in zip(self.bd_preprocessado['tweet_text'], self.bd_preprocessado['sentiment']):
            if sentiment == 1:
                dic = ({'POSITIVO': True, 'NEGATIVO': False})
            elif sentiment == 0:
                dic = ({'POSITIVO': False, 'NEGATIVO': True})
            return_list.append([tweet_text, dic.copy()])
        return return_list