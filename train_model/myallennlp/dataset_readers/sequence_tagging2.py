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
        debug_mode = False
        batch_process = True
        print_vocab = False
        import time

        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        # Do full batch
        if batch_process:
            t0 = time.time()
            with open(file_path, "r") as data_file:

                logger.info("Reading instances from lines in file at: %s", file_path)
                sentences_batch = []
                tags_batch = []
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
                    sentences_batch.append(' '.join(words))

                    tags_batch.append([tag for token, tag in tokens_and_tags])

            # Batch process tokens (faster)
            tokens_batch = self._mytokenizer.batch_tokenize(sentences_batch)

            t1 = time.time()
            total = t1-t0
            print("Token batch process time:")
            print(total)

            # View full vocab - used for figuring out the correct embedding to use
            if print_vocab:
                pos_all = [token.pos_ for tokens in tokens_batch for token in tokens]
                pos_all2 = set(pos_all)
                tag_all = [token.tag_ for tokens in tokens_batch for token in tokens]
                tag_all2 = set(tag_all)
                ner_all = [token.ent_type_ for tokens in tokens_batch for token in tokens]
                ner_all2 = set(ner_all)
                dep_all = [token.dep_ for tokens in tokens_batch for token in tokens]
                dep_all2 = set(dep_all)
                print(pos_all2)
                print(tag_all2)
                print(ner_all2)
                print(dep_all2)
                print("Npos {}: ".format(str(len(pos_all2))))
                print("Npos {}: ".format(str(len(tag_all2))))
                print("Nner {}: ".format(str(len(ner_all2))))
                print("Ndep {}: ".format(str(len(dep_all2))))
                # Results of survey of sizes. Use these for embedding:
                # (Pdb) Npos 15:
                # (Pdb) Npos 49:
                # (Pdb) Nner 19:
                # (Pdb) Ndep 45:
                # {'NUM', 'ADJ', 'INTJ', 'SYM', 'DET', 'NOUN', 'CCONJ', 'ADV', 'PART', 'ADP', 'X', 'VERB', 'PUNCT', 'PRON', 'PROPN'}
                # (Pdb) {'RBR', 'POS', 'XX', 'SYM', ':', 'WDT', ',', 'FW', 'JJ', 'RB', '``', 'NFP', '-LRB-', 'CD', 'LS', 'RBS', 'JJR', 'RP', 'PRP$', '.', "''", 'DT', 'PRP', 'CC', 'VBD', 'NNP', 'AFX', 'EX', 'JJS', 'PDT', 'VBN', 'VBP', 'MD', 'WP$', 'VBZ', 'IN', 'NNPS', '$', 'UH', 'HYPH', 'WRB', 'VB', 'WP', 'TO', 'NNS', 'ADD', 'VBG', '-RRB-', 'NN'}
                # (Pdb) {'', 'LANGUAGE', 'LAW', 'DATE', 'ORG', 'NORP', 'PRODUCT', 'PERSON', 'WORK_OF_ART', 'EVENT', 'GPE', 'QUANTITY', 'LOC', 'TIME', 'PERCENT', 'MONEY', 'CARDINAL', 'FAC', 'ORDINAL'}
                # (Pdb) {'csubjpass', 'prep', 'npadvmod', 'attr', 'cc', 'compound', 'nmod', 'amod', 'advcl', 'pobj', 'appos', 'pcomp', 'advmod', 'quantmod', 'dative', 'nsubj', 'acl', 'predet', 'auxpass', 'mark', 'ccomp', 'neg', 'case', 'oprd', 'relcl', 'ROOT', 'aux', 'acomp', 'nummod', 'det', 'dep', 'preconj', 'meta', 'expl', 'punct', 'parataxis', 'poss', 'conj', 'xcomp', 'agent', 'dobj', 'nsubjpass', 'prt', 'csubj', 'intj'}



            for i in range(len(tokens_batch)):
                tokens = tokens_batch[i]
                tags = tags_batch[i]

                # Bug checking because spacy is stupid....grrrr
                if debug_mode:
                    words = sentences_batch[i].split()
                    words2 = [t.text for t in tokens]
                    if not words == words2:
                        import pdb
                        pdb.set_trace()

                # If these don't match up, it will produce an error. In this csae,
                # pre-emptively enter debug mode
                if not len(tokens) == len(tags):
                    # This is the orignal way tokens would have been calculated in sequence_tagging.py
                    tokens2 = [Token(text=token) for token, tag in tokens_and_tags]
                    print("Error -misamatch between tokens and tags")
                    print(tokens)
                    print(tokens2)

                    import pdb
                    pdb.set_trace()

                yield self.text_to_instance(tokens, tags)

        if not batch_process:
            t0 = time.time()
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
                        # Need to be sure to use modified white space tokenizer (registerd as whitespacy)
                        # because the default option doesn't always reproduce the original tokenization
                        # from versoin re-joined by spaces above.
                        tokens = self._mytokenizer.tokenize(sentence)

                    #tokens = [Token(text=token) for token, tag in tokens_and_tags]
                    tags = [tag for token, tag in tokens_and_tags]

                    # Bug checking because spacy is stupid....grrrr
                    if debug_mode:
                        words2 = [t.text for t in tokens]
                        if not words == words2:
                            import pdb
                            pdb.set_trace()

                    # If these don't match up, it will produce an error. In this csae,
                    # pre-emptively enter debug mode
                    if not len(tokens) == len(tags):
                        # This is the orignal way tokens would have been calculated in sequence_tagging.py
                        tokens2 = [Token(text=token) for token, tag in tokens_and_tags]
                        print("Error -misamatch between tokens and tags")
                        print(tokens)
                        print(tokens2)

                        import pdb
                        pdb.set_trace()

                    yield self.text_to_instance(tokens, tags)
            t1 = time.time()
            total = t1-t0
            print("Iterated token processing time:")
            print(total)

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
