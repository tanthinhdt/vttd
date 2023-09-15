import re
from base import Processor
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO


class Remover(Processor):

    """
    Remove unwanted substrings from Vietnamese text.
    """

    def __init__(self):
        super().__init__()

    def add_stopwords_dash(self, file_path: str):
        """
        Add stopwords from file.
        :param file_path:   Path to file containing stopwords.
        :return:            Self.
        """
        with open(file_path, 'r', encoding="utf8", errors="ignore") as f:
            self.stopwords = f.read().split()
        return self
    
    
    def remove_stopwords(self, text):
        '''
        Remove stopwords from Vietnamese text.
        :param text:          Vietnamese text.
        :return:              Stopwords-removed text.
        '''
        f =  open("vietnamese-stopwords-dash.txt", 'r', encoding="utf-8")
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


    def mixed_word_number(self, text):
        """
        Remove all mixed words and numbers from Vietnamese text.
        :param text:                            Vietnamese text.
        :return:            Mixed_word_removed text.
        """
        text = ' '.join(s for s in text.split() if not any(c.isdigit() for c in s))
        text = re.sub('  +', ' ', text).strip()
        return text
    
    
    def process(self, text: str):
        """
        Remove the unwanted substrings from Vietnamese text.
        :param text:                        Vietnamese text.
        :return:                            Processed text.
        """
        text = self.remove_stopwords(text)

        text = self.remove_emoji(text)

        text = self.mixed_word_number(text)
        
        text = text.strip()

        text = " ".join(text.split())

        return text