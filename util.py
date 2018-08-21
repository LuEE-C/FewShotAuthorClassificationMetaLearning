from collections import Counter
from tqdm import tqdm
import string
import numpy as np
import torchtext.vocab as vocab
import os
import torch
import random
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GutenbergConstructor:
    def __init__(self):
        self.author_to_work_dict, self.author_set = self.read_all_gutenberg()
        self.preprocess_guttenberg()
        self.words_to_indexes, self.indexes_to_words, self.n_words = self.map_words_to_indexes()
        self.glove_embedding = self.get_glove_embedding()
        self.split_validation_and_train_author()

    def split_validation_and_train_author(self):
        self.validation_authors = set(random.sample(self.author_set, 10))
        self.author_set = self.author_set - self.validation_authors
        pickle.dump(self.validation_authors, open('validation_authors', 'wb'))
        pickle.dump(self.author_set, open('train_authors', 'wb'))

    def get_n_task(self, tasks=5, examples=10, examples_size=256):
        sampled_author = random.sample(self.author_set, tasks)
        targets = torch.tensor(np.repeat([i for i in range(tasks)], examples), dtype=torch.long, device=device)
        val_targets = torch.tensor(np.repeat([i for i in range(tasks)], examples * 20), dtype=torch.long, device=device)

        texts = []
        for author in sampled_author:
            length_of_work = len(self.author_to_work_dict[author])
            examples_idx_start = np.random.random_integers(0, length_of_work - examples_size - 1, examples)
            for idx in examples_idx_start:
                texts.append(self.author_to_work_dict[author][idx: idx + examples_size])
        texts = torch.tensor(np.array(texts), dtype=torch.long, device=device)

        val_texts = []
        for author in sampled_author:
            length_of_work = len(self.author_to_work_dict[author])
            examples_idx_start = np.random.random_integers(0, length_of_work - examples_size - 1, examples * 20)
            for idx in examples_idx_start:
                val_texts.append(self.author_to_work_dict[author][idx: idx + examples_size])
        val_texts = torch.tensor(np.array(val_texts), dtype=torch.long, device=device)

        return texts, targets, val_texts, val_targets

    def get_validation_task(self, tasks=5, examples=10, examples_size=256):
        sampled_author = random.sample(self.validation_authors, tasks)
        targets = torch.tensor(np.repeat([i for i in range(tasks)], examples), dtype=torch.long, device=device)
        val_targets = torch.tensor(np.repeat([i for i in range(tasks)], examples * 20), dtype=torch.long, device=device)

        texts = []
        for author in sampled_author:
            length_of_work = len(self.author_to_work_dict[author])
            examples_idx_start = np.random.random_integers(0, length_of_work - examples_size - 1, examples)
            for idx in examples_idx_start:
                texts.append(self.author_to_work_dict[author][idx: idx + examples_size])
        texts = torch.tensor(np.array(texts), dtype=torch.long, device=device)

        val_texts = []
        for author in sampled_author:
            length_of_work = len(self.author_to_work_dict[author])
            examples_idx_start = np.random.random_integers(0, length_of_work - examples_size - 1, examples * 20)
            for idx in examples_idx_start:
                val_texts.append(self.author_to_work_dict[author][idx: idx + examples_size])
        val_texts = torch.tensor(np.array(val_texts), dtype=torch.long, device=device)

        return texts, targets, val_texts, val_targets

    # Make a dictionary of author to 1 string containing all of their work
    def read_all_gutenberg(self):
        author_to_book_dict = dict()
        gutenberg_dir = 'Gutenberg/txt/'
        for books in tqdm(os.listdir(gutenberg_dir)[:1000]):
            author = books.split('___')[0]
            if author not in author_to_book_dict:
                author_to_book_dict[author] = []

            with open(gutenberg_dir + books, 'r', encoding='latin1') as f:
                book = f.readlines()
            book = ' '.join(book)
            author_to_book_dict[author].append(book)

        list_of_author = []
        for author in author_to_book_dict:
            list_of_author.append(author)
            author_to_book_dict[author] = ' '.join(author_to_book_dict[author])
        return author_to_book_dict, set(list_of_author)

    def preprocess_guttenberg(self):
        # Translating punctuation to spaces
        table = str.maketrans('', '', string.punctuation)
        for author in tqdm(self.author_to_work_dict):
            self.author_to_work_dict[author] = self.author_to_work_dict[author].replace('\n', '').lower()
            self.author_to_work_dict[author] = self.author_to_work_dict[author].translate(table)

    def map_words_to_indexes(self):
        # First pass to decide on good tokens
        word_to_occurences = Counter()
        minimum_occurences = 10

        for author in tqdm(self.author_to_work_dict):
            word_to_occurences.update(self.author_to_work_dict[author].split())

        # This is kinda backwards,
        # probably should find the non ignored words and work from there
        ignored_word = set()
        for word in word_to_occurences:
            if word_to_occurences[word] < minimum_occurences:
                ignored_word.update([word])
        # Second pass to tokenize and build dicts
        n_words = 1
        words_to_index = {'unknown_token': 0}
        index_to_words = {0: 'unknown_token'}

        for author in tqdm(self.author_to_work_dict):
            new_work = []
            for word in self.author_to_work_dict[author].split():

                if word in ignored_word:
                    new_work.append(0)
                else:
                    if word not in words_to_index:
                        words_to_index[word] = n_words
                        index_to_words[n_words] = word
                        n_words += 1

                    new_work.append(words_to_index[word])
            self.author_to_work_dict[author] = np.array(new_work)
        return words_to_index, index_to_words, n_words

    def get_glove_embedding(self):
        glove = vocab.GloVe(name='6B', dim=100)

        weights_matrix = np.zeros((self.n_words, 100))

        for word in tqdm(self.words_to_indexes):
            idx = self.words_to_indexes[word]
            weights_matrix[idx] = glove[word]
        return weights_matrix

if __name__ == '__main__':
    gut = GutenbergConstructor()
    gut.get_n_task()