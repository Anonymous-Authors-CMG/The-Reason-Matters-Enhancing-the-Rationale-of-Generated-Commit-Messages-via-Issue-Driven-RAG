from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk.data
import nltk.stem
import re
import math

# 自动下载 nltk 数据
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# 加载 punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# 加载 stopwords
stop_words = set(stopwords.words('english'))

stemmer = nltk.stem.SnowballStemmer('english')


def RemoveHttp(text):
    httpPattern = r'[a-zA-Z]+://[^\s]*'
    return re.sub(httpPattern, ' ', text)


def RemoveTag(text, key):
    keys = key.split("/")
    patterns = [k + '-[0-9]*' for k in keys]
    pattern = "|".join(patterns)
    return re.sub(pattern, ' ', text)


def clean_en_text(text):
    # keep English letters, digits and spaces
    comp = re.compile(r'[^A-Za-z0-9 ]')
    return comp.sub(' ', text)


def RemoveGit(text):
    gitPattern = r'[Gg]it-svn-id'
    return re.sub(gitPattern, ' ', text)


def textProcess(text, key):
    final = []
    if not isinstance(text, str):
        text = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    text = RemoveHttp(text)
    text = RemoveHttp(text)
    if "#" in key:
        text = re.sub(r'#-[0-9]+', ' ', text)
    else:
        text = RemoveTag(text, key)
    text = RemoveGit(text)

    sentences = tokenizer.tokenize(text)
    for sentence in sentences:
        sentence = clean_en_text(sentence)
        word_tokens = word_tokenize(sentence)
        word_tokens = [word for word in word_tokens if word.lower() not in stop_words]
        for word in word_tokens:
            final.append(stemmer.stem(word))

    if len(final) == 0:
        return ' '
    return ' '.join(final)


def codeMatch(word):
    identifier_pattern = r'''[A-Za-z]+[0-9]*_.*|
                             [A-Za-z]+[0-9]*[\.].+|
                             [A-Za-z]+.*[A-Z]+.*|
                             [A-Z0-9]+|
                             _+[A-Za-z0-9]+.+|
                             [a-zA-Z]+[:]{2,}.+'''
    identifier_pattern = re.compile(identifier_pattern)
    return bool(identifier_pattern.match(word))


def diffProcess(text):
    identifiers = []
    sentences = tokenizer.tokenize(text)
    for sentence in sentences:
        word_tokens = word_tokenize(sentence)
        for word in word_tokens:
            if codeMatch(word):
                identifiers.append(word)
    if len(identifiers) == 0:
        return ' '
    return ' '.join(identifiers)
