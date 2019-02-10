import re
from typing import List

from overrides import overrides
import spacy

from allennlp.common import Registrable
from allennlp.common.util import get_spacy_model
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.openai_transformer_byte_pair_indexer import text_standardize

from allennlp.data.tokenizers.word_splitter import WordSplitter

# davedit
from spacy.tokens import Doc
class WhitespaceTokenizer(object):

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

def _remove_spaces(tokens: List[spacy.tokens.Token]) -> List[spacy.tokens.Token]:
    return [token for token in tokens if not token.is_space]

@WordSplitter.register('whitespacy')
class SpacyWordSplitter(WordSplitter):
    """
    A ``WordSplitter`` that uses spaCy's tokenizer, modified to use a white space
    tokenizer.  It's fast and reasonable - this is the
    recommended ``WordSplitter``.
    """
    def __init__(self,
                 language: str = 'en_core_web_sm',
                 pos_tags: bool = False,
                 parse: bool = False,
                 ner: bool = False) -> None:
        self.spacy = get_spacy_model(language, pos_tags, parse, ner)
            # Modify default to use white space tokenizer, defined above
        self.spacy.tokenizer = WhitespaceTokenizer(self.spacy.vocab)

    @overrides
    def batch_split_words(self, sentences: List[str]) -> List[List[Token]]:
        return [_remove_spaces(tokens)
                for tokens in self.spacy.pipe(sentences, n_threads=-1)]

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        # This works because our Token class matches spacy's.
        return _remove_spaces(self.spacy(sentence))
