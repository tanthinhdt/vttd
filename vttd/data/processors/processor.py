from normalizer import Normalizer
from remover import Remover
import pandas as pd 


def process(text: str):
    """
    Normalize substrings from Vietnamese text.
    :param text:              Vietnamese text.
    :return:                  Processed text.
    """
    text = text.lower()
    
    text = Remover().remove_url(text)
    
    text = Remover().remove_mail(text)
            
    text = Remover().remove_tags(text)
    
    text = Remover().remove_mixed_word_number(text)
    
    text = Remover().remove_special_character(text)
    
    text = Remover().remove_emoji(text)
    
    text = Remover().remove_repeated_character(text)
            
    text = Normalizer().normalize_abbreviation(text)
    
    text = Normalizer().normalize_abbreviation_special(text)
    
    text = Normalizer().normalize_kk_abbreviation(text)
    
    text = Normalizer().abbreviation_predict(text)
            
    text = Normalizer().tokenize(text)
    
    text = Remover().remove_stopwords(text)
    
    return text


data_test = pd.read_csv(r"raw\test.csv")
data_test["Free_text"] = data_test["Free_text"].apply(process)
data_test.to_csv("processed/data_test_processed_withh_stopwords.csv", index = False)