from collections import Counter
import re
from operator import itemgetter
import string

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

    def fit(self, text, max_iter=5, min_subwords=None):
        """
        TODO: documentation
        """

        # add all text characters, our control sequence and all ascii letters
        text_chars = set(text) | {"@eow"} | set(string.ascii_letters)
        if " " in text_chars:
            text_chars.remove(" ")

        # replace our control characters
        text = text.replace("#", "&#35").replace("@", "&#64;")

        # keep original text for casing information
        text_str_original = str(text)
        text_str_lower = str(text)

        # tokenize and preprocess text
        text = tokenizer(text)
        text = [separator(word) + " @eow" for word in text]

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
                    v*len(k.split())
                    for k,v in vocab.items()
                ])
                if total_subwords <= min_subwords:
                    break

            self.bpe_codes[best_pair] = i

        # find most common casing

        self.bpe_codes_join = {
            (k[0]+k[1]):v
            for k,v in self.bpe_codes.items()
        }

        # add single characters
        for char_i, i in zip(text_chars, range(i+1, i+1+len(text_chars))):
            self.bpe_codes_join[char_i] = i


        self.bpe_codes_join_rev = {
            v:k
            for k,v in self.bpe_codes_join.items()
        }
        # map unknown character to unknown character
        self.bpe_codes_join_rev["#"] = "#"

        self.fitted = True
        self.vocab_words = list(vocab.keys())

    def check_fitted(self):
        if not self.fitted:
            raise Exception("The model has not been fitten yet.")

    def encode(self, text):
        self.check_fitted()

        # tokenize and preprocess text
        text = tokenizer(text)

        # actual encoding
        text = [self._encode_word(word) for word in text]

        text_ids = [self._encoded_word_to_ids(word) for word in text]

        print(text)
        print(text_ids)
        return text, text_ids
            
    def _encode_word(self, original_word):
        if len(original_word) == 1:
            return original_word

        word = list(original_word)
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

            pair_to_merge = min(bpe_codes_pairs, key=itemgetter(1))[0]
            word = self._create_new_word(word, pair_to_merge)

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
        self.check_fitted()
        return [
            self._post_process_word(self._decode_word(word))
            for word in text
        ]

    def _post_process_word(self, word):
        return word.replace("@eow", "")

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

