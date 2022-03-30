from collections import Counter
import re
from operator import itemgetter
import string
import numpy as np


def get_pair_stats(vocab):
    """Get counts of pairs of consecutive symbols."""

    pairs = {}
    for word, frequency in vocab.items():
        symbols = word.split()

        # count occurrences of pairs
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            current_frequency = pairs.get(pair, 0)
            pairs[pair] = current_frequency + frequency

    return pairs


def merge_vocab(best_pair, vocab_in):
    vocab_out = {}

    # re.escape
    # ensures the characters of our input pair will be handled as is and
    # not get mistreated as special characters in the regular expression.
    pattern = re.escape(' '.join(best_pair))
    replacement = ''.join(best_pair)

    for word_in in vocab_in:
        # replace most frequent pair in all vocabulary
        word_out = re.sub(pattern, replacement, word_in)
        vocab_out[word_out] = vocab_in[word_in]

    return vocab_out


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    return pairs


def tokenizer(text):
    """TODO: replace in the future by something proper"""
    return text.split()


def separator(text):
    """Adds spaces between letters"""
    return ' '.join(list(text))


class extBPE:
    def __init__(self):
        self.fitted = False

    def fit(self, text, max_iter=5, min_subwords=None, lowercase=False):
        """
        TODO: documentation
        """

        # add all text characters, our control sequence and all ascii letters
        text_chars = set(text) | {"@eow", "@low", "@upp", "@cap"}
        text_chars = text_chars | set(string.ascii_letters)
        if " " in text_chars:
            text_chars.remove(" ")

        # replace our control characters
        text = text.replace("#", "&#35").replace("@", "&#64;")

        # tokenize and preprocess text
        text = tokenizer(text)
        text = [separator(word) + " @eow" for word in text]

        # keep original text for casing information
        if lowercase:
            text_str_original = " ".join(
                [word.replace(" ", "").lower() for word in text]
            )
        else:
            text_str_original = " ".join(
                [word.replace(" ", "") for word in text]
            )

        # lowercase everything
        if lowercase:
            text = [word.lower() for word in text]

        # compute the initial vocab+freqs
        vocab = Counter(text)

        # encoding table
        self.bpe_codes = {}

        # perform merging
        for i in range(max_iter):
            pair_stats = get_pair_stats(vocab)
            best_pair = max(pair_stats, key=pair_stats.get)
            vocab = merge_vocab(best_pair, vocab)

            # check whether we would have less subwords than we need
            if min_subwords is not None:
                total_subwords = sum([
                    v * len(k.split())
                    for k, v in vocab.items()
                ])
                if total_subwords <= min_subwords:
                    break

            self.bpe_codes[best_pair] = i

        bpe_codes_new = {}

        # find most common casing for every subword
        for subword, subword_v in self.bpe_codes.items():
            # join pairs
            subword_j = subword[0] + subword[1]

            # find all occurences
            indices = re.finditer(pattern=re.escape(
                subword_j), string=text_str_original)
            indices = [(index.start(), index.end()) for index in indices]

            vocab_case = Counter([
                text_str_original[x[0]:x[1]]
                for x in indices
            ])

            # TODO: add decider that in case of a draw prefer lowercase
            subword_j_most_common = vocab_case.most_common()[0][0]

            # store most common casing in the vocabulary
            bpe_codes_new[(
                subword[0], subword[1],
            )] = (subword_v, subword_j_most_common)

        self.bpe_codes = bpe_codes_new

        self.bpe_codes_join = {
            (k[0] + k[1]): v[0]
            for k, v in self.bpe_codes.items()
        }

        # add single characters at the end of ids
        # TODO: move to the beginning for stability
        for char_i, i in zip(text_chars, range(i + 1, i + 1 + len(text_chars))):
            self.bpe_codes_join[char_i] = i

        self.bpe_codes_join_rev = {
            v: k
            for k, v in self.bpe_codes_join.items()
        }
        # map unknown character to unknown character
        self.bpe_codes_join_rev["#"] = "#"

        self.fitted = True
        self.vocab_words = list(vocab.keys())

    def check_fitted(self):
        if not self.fitted:
            raise Exception("The model has not been fitten yet.")

    def encode(self, text, disable_special=False):
        self.check_fitted()

        # tokenize and preprocess text
        text = tokenizer(text)

        # actual encoding
        text = [self._encode_word(word, disable_special=disable_special) for word in text]
        text_ids = [self._encoded_word_to_ids(word) for word in text]

        return text, text_ids

    def _encode_word(self, word, disable_special):
        word_true = self._encode_word_sub(word)

        if disable_special:
            return word_true

        word_lower = self._encode_word_sub(word.lower())
        word_upper = self._encode_word_sub(word.upper())
        word_capital = self._encode_word_sub(word.capitalize())

        # decide which encoding to use
        min_length = np.argmin([
            len(word_true), 1 + len(word_lower),
            1 + len(word_upper), 1 + len(word_capital)
        ])

        def find_target_op(word, word_bped):
            if word == "".join(word_bped).replace("@eow", "").lower():
                return ["@low"] + word_bped
            elif word == "".join(word_bped).replace("@eow", "").upper():
                return ["@upp"] + word_bped
            elif word == "".join(word_bped).replace("@eow", "").capitalize():
                return ["@cap"] + word_bped
            else:
                # can't reverse opretaions
                return word_true
                
        if min_length == 0:
            return word_true
        elif min_length == 1:
            return find_target_op(word, word_lower)
        elif min_length == 2:
            return find_target_op(word, word_upper)
        elif min_length == 3:
            return find_target_op(word, word_capital)
        
    def _encode_word_sub(self, word):
        if len(word) == 1:
            return word

        word = list(word)
        word.append("@eow")

        while True:
            pairs = get_pairs(word)
            bpe_codes_pairs = [
                (pair, self.bpe_codes[pair])
                for pair in pairs
                if pair in self.bpe_codes
            ]
            if not bpe_codes_pairs:
                break

            # take the value and the count (not the most common form)
            pair_to_merge = min(bpe_codes_pairs, key=lambda x: x[1][0])
            word = self._create_new_word(word, pair_to_merge[0])

        return word

    def _encoded_word_to_ids(self, word):
        self.check_fitted()

        return [
            self.bpe_codes_join[c]
            if c in self.bpe_codes_join
            else "#"
            for c in word
        ]

    def decode(self, text):
        return " ".join([
            self._post_process_word(self._decode_word(word))
            for word in text
        ])

    def decode_raw(self, text):
        self.check_fitted()
        return [
            self._decode_word(word)
            for word in text
        ]

    def _post_process_word(self, word):
        word = word.replace("@eow", "")

        # process special tags and strip them
        if word.startswith("@low"):
            word = word[4:].lower()
        elif word.startswith("@upp"):
            word = word[4:].upper()
        elif word.startswith("@cap"):
            word = word[4:].capitalize()
        return word

    def _decode_word(self, word):
        return ''.join([
            self.bpe_codes_join_rev[c_id]
            for c_id in word
        ])

    def _create_new_word(self, word, pair_to_merge):
        first, second = pair_to_merge
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except ValueError:
                new_word.extend(word[i:])
                break

            if i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(first)
                i += 1

        return new_word
