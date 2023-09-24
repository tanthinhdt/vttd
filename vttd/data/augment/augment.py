import random
import pandas as pd
from pyvi import ViTokenizer, ViPosTagger


def synonym_substitution(text, synonym_dict):
    words, pos_tags = ViPosTagger.postagging(ViTokenizer.tokenize(text))
    new_text = []
    for word, pos_tag in zip(words, pos_tags):
        if word in synonym_dict and pos_tag == 'N':
            new_word = random.choice(synonym_dict[word])
            new_text.append(new_word)
        else:
            new_text.append(word)
    return ' '.join(new_text)


def random_insertion(text, n_insertions=1, vocabulary=None):
    words = ViTokenizer.tokenize(text).split()
    for _ in range(n_insertions):
        random_word = random.choice(vocabulary) if vocabulary else random.choice(words)
        position = random.randint(0, len(words))
        words.insert(position, random_word)
    return ' '.join(words)


def random_swap(text, n_swaps=1):
    words = ViTokenizer.tokenize(text).split()
    for _ in range(n_swaps):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)


def random_deletion(text, p=0.2):
    words = ViTokenizer.tokenize(text).split()
    remaining_words = [word for word in words if random.uniform(0, 1) > p]
    return ' '.join(remaining_words)

