


def extract_no_stopwords(tokens):
    from nltk.corpus import stopwords
    out = list()
    out = [w for w in tokens if not w in stopwords.words('english')]
    return out
