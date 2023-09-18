import re
from base import Processor
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO


class Remover(Processor):

    """
    Remove unwanted substrings from Vietnamese text.
    """

    def __init__(self):
        super().__init__()
        
    
    def remove_stopwords(self, text):
        '''
        Remove stopwords from Vietnamese text.
        :param text:          Vietnamese text.
        :return:              Stopwords-removed text.
        '''
        f =  open("resources/vietnamese_stop_words/vietnamese-stopwords-dash.txt", 'r', encoding="utf-8")
        stopwords = []
        for line in f:
            stopwords.append(line.strip())
        
        words = text.split(" ")

        filtered_words = []
        for word in words:
            if word not in stopwords:
                filtered_words.append(word)
        text = " ".join(filtered_words)
        return text
    
    # Remove emoji and emoticons
    def remove_emoji(self, text):
        for emot in UNICODE_EMOJI:
            text = str(text).replace(emot, ' ')
        text = re.sub('  +', ' ', text).strip()
        return text


    # Remove url
    def url(self, text):
        text = re.sub(r'https?://\S+|www\.\S+', ' ', str(text))
        text = re.sub('  +', ' ', text).strip()
        return text

    
    # remove special character
    def special_character(self, text):
        text = re.sub(r'\d+', lambda m: " ", text)
        # text = re.sub(r'\b(\w+)\s+\1\b',' ', text) #remove duplicate number word
        text = re.sub("[~!@#$%^&*()_+{}“”|:\"<>?`´\-=[\]\;\\\/.,]", " ", text)
        text = re.sub('  +', ' ', text).strip()
        return text


    def remove_duplicates(self, input_str):
        output_str = input_str[0]
        for char in input_str[1:]:
            if char != output_str[-1]:
                output_str += char
        return output_str


    def mail(self, text):
        text = re.sub(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'," ", text)
        return text


    # remove mention tag and hashtag
    def tag(self, text):
        text = re.sub(r"(?:\@|\#|\://)\S+", " ", text)
        text = re.sub('  +', ' ', text).strip()
        return text


    def remove_emoji(self, text):
        """
        Remove emoji.
        :param text:   text needed to remove emoji.
        :return:           Emoji-removed text.
        """
        for emot in UNICODE_EMOJI:
            text = str(text).replace(emot, ' ')
        text = re.sub('  +', ' ', text).strip()
        return text

    
    def remove_mixed_word_number(self, text):
        """
        Remove all mixed words and numbers from Vietnamese text.
        :param text:                            Vietnamese text.
        :return:            Mixed_word_removed text.
        """
        text = ' '.join(s for s in text.split() if not any(c.isdigit() for c in s))
        text = re.sub('  +', ' ', text).strip()
        return text
    
