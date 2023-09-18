from normalizer import Normalizer
from remover import Remover


def process(text: str):
    """
    Normalize substrings from Vietnamese text.
    :param text:              Vietnamese text.
    :return:                  Processed text.
    """
    text = text.lower()
    
    text = Remover().url(text)
    
    text = Remover().mail(text)
            
    text = Remover().tag(text)
    
    text = Remover().remove_mixed_word_number(text)
    
    text = Remover().special_character(text)
    
    text = Remover().remove_emoji(text)
    
    text = Remover().remove_duplicates(text)
            
    text = Normalizer().normalize_abbreviation(text)
    
    text = Normalizer().normalize_abbreviation_special(text)
    
    text = Normalizer().normalize_kk_abbreviation(text)
    
    text = Normalizer().abbreviation_predict(text)
            
    text = Normalizer().tokenize(text)
    
    text = Remover().remove_stopwords(text)
    
    return text
