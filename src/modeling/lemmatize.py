import pymorphy2

morph = pymorphy2.MorphAnalyzer()


def lemmatize_word(word):
    """Лемматизация одного слова"""
    return morph.parse(word)[0].normal_form


def preprocess_text(text):
    """Предобработка текста с лемматизацией"""
    text = text.lower().strip()
    words = text.split()
    
    return ' '.join(map(lemmatize_word, words))