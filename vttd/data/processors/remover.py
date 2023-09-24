import re
from base import Processor
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
import pandas as pd


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
    

    def remove_url(self, text):
        """
        Remove url.
        :param text:   text needed to remove url.
        :return:           url-removed text.
        """
        text = re.sub(r'https?://\S+|www\.\S+', ' ', str(text))
        return text

    
    def remove_special_character(self, text):
        """
        Remove special character.
        :param text:   text needed to remove special character.
        :return:           special_character-removed text.
        """
        text = re.sub(r'\d+', lambda m: " ", text)
        text = re.sub("[~!@#$%^&*()_+{}“”|:\"<>?`´\-=[\]\;\\\/.,]", " ", text)
        return text
    

    def remove_repeated_character(self, text):
        """
        Remove repeated character.
        :param text:   text needed to remove repeated character.
        :return:           repeated_character-removed text.
        """
        words = text.split()
        for i in range(len(words)):
            if len(words[i])==2:
                continue
            words[i] = re.sub(r'(\w)\1+', r'\1', words[i])
        return ' '.join(words)


    def remove_mail(self, text):
        """
        Remove mails.
        :param text:   text needed to remove mails.
        :return:           mails-removed text.
        """
        text = re.sub(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'," ", text)
        return text


    def remove_tags(self, text):
        """
        Remove tag and hashtag.
        :param text:   text needed to remove tags and hashtags.
        :return:           tags-removed text.
        """
        text = re.sub(r"(?:\@|\#|\://)\S+", " ", text)
        return text


    def remove_emoji(self, text):
        """
        Remove emoji.
        :param text:   text needed to remove emoji.
        :return:           Emoji-removed text.
        """
        character2emoji = pd.read_excel('resources\dictionary\character2emoji.xlsx')
        
        for i in range(character2emoji.shape[0]):
            text = text.replace(character2emoji.at[i, 'character'], " " + character2emoji.at[i, 'emoji'] + " ")
        
        for emot in UNICODE_EMOJI:
            text = str(text).replace(emot, ' ')
        return text

    
    def remove_mixed_word_number(self, text):
        """
        Remove all mixed words and numbers from Vietnamese text.
        :param text:        text needed to remove mixed word number.
        :return:            Mixed_word_removed text.
        """
        text = ' '.join(s for s in text.split() if not any(c.isdigit() for c in s))
        return text
    
