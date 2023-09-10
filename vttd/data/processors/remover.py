import re
from vttd.data.processors.base import BaseProcessor


class Remover(BaseProcessor):
    """
    Remove unwanted substrings from Vietnamese text.
    """
    def __init__(self):
        super().__init__()
        self.stopwords_re = re.compile(r'\b(' + r'|'.join(self.stopwords) + r')\b\s*')
        self.hashtags_re = re.compile(r'#\w+')
        self.mentions_re = re.compile(r'@\w+')
        self.urls_re = re.compile(r'http\S+')
        self.trailing_whitespace_re = re.compile(r'\s+$')

    def add(self, file_path: str):
        """
        Add stopwords from file.
        :param file_path:   Path to file containing stopwords.
        :return:            Self.
        """
        with open(file_path, 'r') as f:
            self.stopwords = f.read().split()
        return self

    def process(self, text: str,
                remove_stopwords: bool = False,
                remove_hashtags: bool = False,
                remove_mentions: bool = False,
                remove_urls: bool = False,
                remove_trailing_whitespace: bool = False,):
        """
        Remove the unwanted substrings from Vietnamese text.
        :param text:                        Vietnamese text.
        :param remove_stopwords:            Remove stopwords.
        :param remove_hashtags:             Remove hashtags.
        :param remove_mentions:             Remove mentions.
        :param remove_urls:                 Remove URLs.
        :param remove_trailing_whitespace:  Remove trailing whitespace.
        :return:                            Processed text.
        """
        if remove_stopwords:
            text = self.stopwords_re.sub('', text)
        if remove_hashtags:
            text = self.hashtags_re.sub('', text)
        if remove_mentions:
            text = self.mentions_re.sub('', text)
        if remove_urls:
            text = self.urls_re.sub('', text)
        if remove_trailing_whitespace:
            text = self.trailing_whitespace_re.sub('', text)
        return text
