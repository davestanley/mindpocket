# Modified version of sequence_tagging dataset reader to allow specifying a custom
# tokenizer that operates on each sentence. Allows capturing pos tags, ner tags, and
# dependencies. Using spacy tagger (see https://spacy.io/usage/spacy-101#annotations-pos-deps)



from typing import Dict, List
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
    # davedit
from allennlp.data.tokenizers import Tokenizer, WordTokenizer


# Davedit - load spacey
import spacy
nlp = spacy.load('en_core_web_sm')

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_WORD_TAG_DELIMITER = "###"

@DatasetReader.register("sequence_tagging2")
class SequenceTaggingDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD###TAG [TAB] WORD###TAG [TAB] ..... \n

    and converts it into a ``Dataset`` suitable for sequence tagging. You can also specify
    alternative delimiters in the constructor.

    Parameters
    ----------
    word_tag_delimiter: ``str``, optional (default=``"###"``)
        The text that separates each WORD from its TAG.
    token_delimiter: ``str``, optional (default=``None``)
        The text that separates each WORD-TAG pair from the next pair. If ``None``
        then the line will just be split on whitespace.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    """
    def __init__(self,
                 word_tag_delimiter: str = DEFAULT_WORD_TAG_DELIMITER,
                 token_delimiter: str = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 mytokenizer: Tokenizer = None,           # davedit
                 use_spacy_directly: bool = False) -> None:            #davedit
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._word_tag_delimiter = word_tag_delimiter
        self._token_delimiter = token_delimiter
        self._mytokenizer = mytokenizer or WordTokenizer()
        self._use_spacy_directly = use_spacy_directly

    @overrides
    def _read(self, file_path):
        debug_mode = True

        # def words2text(words):
        #     return ' '.join(join_punctuation(words))
        #
        # def join_punctuation(seq, characters='.,;?!'):
        #     # For joining lists of words together with correct
        #     # punctuation spacings
        #     characters = set(characters)
        #     seq = iter(seq)
        #     current = next(seq)
        #
        #     for nxt in seq:
        #         if nxt in characters:
        #             current += nxt
        #         else:
        #             yield current
        #             current = nxt
        #
        #     yield current
        #
        # if debug_mode:
        #     from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
        #     ws = SpacyWordSplitter()

        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:

            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")

                # skip blank lines
                if not line:
                    continue


                tokens_and_tags = [pair.rsplit(self._word_tag_delimiter, 1)
                                   for pair in line.split(self._token_delimiter)]

                # davedit
                # Get text only
                words = [word for word, tag in tokens_and_tags]
                sentence = ' '.join(words)
                # if debug_mode:
                #     words2 = ws.split_words(sentence);
                    # if not words == words2:
                    #     import pdb;
                    #     pdb.set_trace()

                # sentence = sentence.replace(' "— ','"—')
                # sentence = sentence.replace('— ','—')
                # sentence = sentence.replace(' \'','\'')          # Delete any spaces appearing before a '
                                                                 # Necessary because spacy is stupid
                             # For example, it splits I 'm into I, ', m  (e.g., len=3)
                             # but splits I'm into I, 'm (e.g., len = 2)


                if self._use_spacy_directly:
                    # # # # Usings spacy directly # # # #
                    # Load into spacy
                    doc = nlp(sentence)

                    # Calculate pos
                    spacy_pos = [token.pos_ for token in doc]

                    # Calculate tags
                    spacy_tags = [token.tag_ for token in doc]

                    # Calculate NER tags
                    pos_tags = [token.ent_type_ for token in doc]

                    # Testing only
                    # temp = [(tt[0],tt[1],a,b) for tt,a,b in zip(tokens_and_tags,spacy_pos,spacy_tags)]
                    tokens = [Token(text=tt[0],pos=pos,tag=tag) for tt,pos,tag in zip(tokens_and_tags,spacy_pos,spacy_tags)]
                else:
                    # Use specified mytokenizer
                    tokens = self._mytokenizer.tokenize(sentence)

                #tokens = [Token(text=token) for token, tag in tokens_and_tags]
                tags = [tag for token, tag in tokens_and_tags]

                # Bug checking because spacy is stupid....grrrr


                if not len(tokens) == len(tags):
                    tokens2 = [Token(text=token) for token, tag in tokens_and_tags]

                    import spacy
                    from spacy.tokens import Doc

                    class WhitespaceTokenizer(object):
                        def __init__(self, vocab):
                            self.vocab = vocab

                        def __call__(self, text):
                            words = text.split(' ')
                            # All tokens 'own' a subsequent space character in this tokenizer
                            spaces = [True] * len(words)
                            return Doc(self.vocab, words=words, spaces=spaces)

                    nlp = spacy.load('en_core_web_sm')
                    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
                    doc = nlp(sentence)
                    words2 = [t.pos for t in doc]

                    print(len(tokens))
                    print(len(tags))

                    # For Testing
                    from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
                    ws = SpacyWordSplitter()
                    context_split = ws.split_words(sentence);
                    sentence2 = 'the Karma Kargyu—cannot cannot'
                    sentence3 = 'the Karma Kargyu — cannot cannot'
                    ws.split_words(sentence2)
                    ws.split_words(sentence3)
                    print(ws.split_words(sentence2))
                    print(ws.split_words(sentence3))
                    import pdb
                    pdb.set_trace()

                yield self.text_to_instance(tokens, tags)

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        # import code
        # code.interact(local=locals())
        # import pdb
        # pdb.set_trace()
        # print(len(sequence))
        # print(len(tags))
        # print(sequence)
        if tags is not None:
            fields["tags"] = SequenceLabelField(tags, sequence)
        return Instance(fields)
