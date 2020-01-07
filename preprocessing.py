from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("english")

def preprocess(sentence):
    return sentence.replace('.','').replace(',','')

def preprocess_batch(sentences):
    new_sentences = []
    for sentence in sentences:
        new_sentences.append(preprocess(sentence.lower()))
    return new_sentences
    
if __name__ == '__main__':
    pass