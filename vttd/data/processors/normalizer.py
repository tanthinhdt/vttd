import re
import os
import joblib
import unicodedata
import pandas as pd
import numpy as np
import regex as re
from sklearn import preprocessing
from scipy import sparse
from base import Processor
from pyvi import ViTokenizer
from sklearn import preprocessing
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class Normalizer(Processor):
    """
    Normalize Vietnamese text.
    """
    def __init__(self):
        super().__init__()    
        self.tags_re = re.compile(r'(?:\@|\#|\://)\S+')
        self.urls_re = re.compile(r'https?://\S+|www\.\S+')
        self.special_character_re = re.compile(r'[~!@#$%^&*()_+{}“”|:\"<>?`´\-=[\]\;\\\/.,]')
    
    
    def remove_repeated_character(self, text):
        """
        Remove all repeated_characters from Vietnamese text.
        :param text:                            Vietnamese text.
        :return:            repeated_character_removed text.
        """
        text = re.sub(r'(\w)\1+', r'\1', text)
        text = re.sub('  +', ' ', text).strip()
        return text
    

    def normalize_unicode(text):
        """
        Normalize Unicode text from Vietnamese text.
        :param text                 Vietnamese text.
        :return:                    Processed text.
        """
        normalized_text = unicodedata.normalize('NFC', text)
        return normalized_text
    
    
    def normalize_abbreviation(self, text):
        """
        Normalize abbrevaition character from Vietnamese text.
        :param text:                          Vietnamese text.
        :return:                              Processed text.
        """
        adn_path = os.path.join(os.getcwd(), 'resources\dictionary\abb_dict_normal.xlsx')
        abb_dict_normal = pd.read_excel(adn_path)
        text = str(text)
        temp = ''
        for word in text.split():
            for i in range(abb_dict_normal.shape[0]):
                if str(abb_dict_normal.at[i, 'abbreviation']) == str(word):
                    word = str(abb_dict_normal.at[i, 'meaning'])
            temp = temp + ' ' + word
        text = temp
        text = re.sub('  +', ' ', text).strip()
        return text
    
    
    def normalize_abbreviation_special(self, text):
        """
        Nomarlize abbreviation character from Vietnamese text.
        :param text:                          Vietnamese text.
        :return:                              Processed text.
        """
        ads_path = os.path.join(os.getcwd(), 'resources\dictionary\abb_dict_special.xlsx')
        abb_dict_special = pd.read_excel(ads_path)
        text = ' ' + str(text) + ' '
        for i in range(abb_dict_special.shape[0]):
            text = text.replace(' ' + abb_dict_special.at[i, 'abbreviation'] + ' ',
                                ' ' + abb_dict_special.at[i, 'meaning'] + ' ')
        text = re.sub('  +', ' ', text).strip()
        return text
    
    
    def normalize_kk_abbreviation(self, text):
        """
        Normalize kk abbreviation character from Vietnamese text.
        :param text:                             Vietnamese text.
        :return:                                 Processed text.
        """
        text = str(text)
        for t in text.split():
            if 'kk' in t:
                text = text.replace(t, ' ha ha ')
            else:
                if 'kaka' in t:
                    text = text.replace(t, ' ha ha ')
                else:
                    if 'kiki' in t:
                        text = text.replace(t, ' ha ha ')
                    else:
                        if 'haha' in t:
                            text = text.replace(t, ' ha ha ')
                        else:
                            if 'hihi' in t:
                                text = text.replace(t, ' ha ha ')
        text = re.sub('  +', ' ', text).strip()
        return text
    
    
    def tokenize(self, text):
        """
        Tokennize character from Vietnamese text.
        :param text:             Vietnamese text.
        :return:                 Processed text.
        """
        text = str(text)
        text = ViTokenizer.tokenize(text)
        return text
    
    
    def annotations(self, dataset):
        """
        Generate postion list from Vietnamese text.
        :param dataset:               Vietnamese text.
        :return:                   Processed text.
        """
        pos = []
        max_len = 8000
        for i in range(dataset.shape[0]):
            n = len(dataset.at[i, 'cmt'])
            l = [0] * max_len
            s = int(dataset.at[i, 'start_index'])
            e = int(dataset.at[i, 'end_index'])
            for j in range(s, e):
                l[j] = 1
            pos.append(l)
        return pos


    def abbreviation_predict(self, t):
        """
        Predict and normalize abbreviation from Vietnamese text.
        :param t:                               Vietnamese text.
        :return:                                Processed text.
        """
        model_path = os.path.join(os.getcwd(), 'resources\abb_model\abb_model.sav')
        loaded_model = joblib.load(model_path)

        da_path = os.path.join(os.getcwd(), 'resources\dictionary/abbreviation_dictionary_vn.xlsx')
        train_path = os.path.join(os.getcwd(), 'resources\dictionary/train_duplicate_abb_data.xlsx')
        dev_path = os.path.join(os.getcwd(), 'resources\dictionary/dev_duplicate_abb_data.xlsx')
        test_path = os.path.join(os.getcwd(), 'resources\dictionary/test_duplicate_abb_data.xlsx')
        duplicate_abb = pd.read_excel(da_path, sheet_name='duplicate', header=None)
        duplicate_abb = list(duplicate_abb[0])

        train_duplicate_abb_data = pd.read_excel(train_path)
        dev_duplicate_abb_data = pd.read_excel(dev_path)
        test_duplicate_abb_data = pd.read_excel(test_path)
        duplicate_abb_data = pd.concat([train_duplicate_abb_data, dev_duplicate_abb_data, test_duplicate_abb_data],
                                    ignore_index=True)
        duplicate_abb_data = duplicate_abb_data.drop_duplicates(keep='last').reset_index(drop=True)

        X = duplicate_abb_data[['abb', 'start_index', 'end_index', 'cmt']]
        y = duplicate_abb_data['origin']

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        enc = DictVectorizer()
        Tfidf_vect = TfidfVectorizer(max_features=1200)

        temp = self.annotations(X)
        X_pos = sparse.csr_matrix(np.asarray(temp))
        X_abb = enc.fit_transform(X[['abb']].to_dict('records'))
        X_text = Tfidf_vect.fit_transform(X['cmt'])
        X = hstack((X_abb, X_pos, X_text))

        text = str(t)
        max_len = 8000
        if len(t) > max_len:
            text = t[:max_len]

        cmt = ' ' + text + ' '
        for abb in duplicate_abb:
            start_index = 0
            count = 0
            while start_index > -1:
                start_index = cmt.find(' ' + abb + ' ') 
                if start_index > -1:
                    end_index = start_index + len(abb)
                    t = pd.DataFrame([[abb, start_index, end_index, text]],
                                    columns=['abb', 'start_index', 'end_index', 'cmt'], index=None)
                    temp = self.annotations(t)
                    X_pos = sparse.csr_matrix(np.asarray(temp))

                    X_abb = enc.transform(t[['abb']].to_dict('records'))
                    X_text = Tfidf_vect.transform([text])

                    X = hstack((X_abb, X_pos, X_text))
                    predict = loaded_model.predict(X)
                    origin = le.inverse_transform(predict.argmax(axis=1))
                    origin = ''.join(origin)
                    text = text[:start_index + count * (len(origin) - len(abb))] + origin + text[end_index + count * (
                                len(origin) - len(abb)):]
                    text = ''.join(text)
                    count = count + 1
                    for i in range(start_index + 1, end_index + 1): 
                        cmt = cmt[:i] + ' ' + cmt[i + 1:]
        return text
    
    
    def process(self, text: str):
        """
        Normalize substrings from Vietnamese text.
        :param text:              Vietnamese text.
        :return:                  Processed text.
        """
        text = self.tags_re.sub('', text)

        text = self.urls_re.sub('', text)
        
        text = self.remove_repeated_character(text)
        
        text = self.special_character_re.sub('',text)
        
        text = self.normalize_abbreviation(text)
        
        text = self.normalize_abbreviation_special(text)
        
        text = self.normalize_kk_abbreviation(text)
        
        text = self.abbreviation_predict(text)
                
        text = self.tokenize(text)
        
        return text
    
    