import os
import argparse

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import BertTokenizer

from fairseq.binarizer import Binarizer
from fairseq.data import indexed_dataset, Dictionary


class BilingualDataPreprocessor:

    def __init__(self, use_bert_tokenizer=False):
        self.use_bert_tokenizer = use_bert_tokenizer

        if use_bert_tokenizer:
            # Load the BERT tokenizer
            self.en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.en_tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
            self.en_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Amharic tokenizer always uses WordLevel tokenizer
        self.am_tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        self.am_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def train_tokenizer(self, files):
        # If BERT tokenizer is not being used, then train with WordLevel tokenizer
        if not self.use_bert_tokenizer:
            trainer = trainers.WordLevelTrainer(vocab_size=30000,
                                                special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
            self.en_tokenizer.train(files=files, trainer=trainer)
            self.en_tokenizer.add_tokens(["[UNK]"])
            self.am_tokenizer.train(files=files, trainer=trainer)
            self.am_tokenizer.add_tokens(["[UNK]"])

    def tokenize_line_with_bert(self, line):
        # Uses BERT tokenizer to tokenize
        tokens = self.en_tokenizer.tokenize(line)
        return tokens

    def build_dictionary(self, filename, tokenizer):
        dict_obj = Dictionary()
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                # Use provided tokenizer for encoding
                if tokenizer == self.tokenize_line_with_bert:
                    tokens = tokenizer(line.strip())
                else:
                    tokens = tokenizer.encode(line.strip()).tokens
                for token in tokens:
                    dict_obj.add_symbol(token)
        return dict_obj

    def save_dictionary(self, dictionary, filename, pre_fix):
        """Save a dictionary to a file only if conditions are met"""
        if "train.en-am" in pre_fix or 'train.bert.en-am' in pre_fix:
            with open(filename, 'w', encoding='utf-8') as file:
                dictionary.save(file)

    def preprocess_data(self, en_file, am_file, imp, pre_fix):
        am_dict = None
        en_dict = None
        en_file = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), en_file)
        am_file = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), am_file)
        pre_fix = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), pre_fix)

        en_token_file = pre_fix + ".en.tokenized.en"
        am_token_file = pre_fix + ".am.tokenized.am"

        en_dict_filename = "bert-dict.en.txt" if self.use_bert_tokenizer else "dict.en.txt"
        am_dict_filename = "dict.am.txt"  # Assuming you want this name regardless of the tokenizer used for English

        if not os.path.exists(os.path.dirname(en_token_file)):
            os.makedirs(os.path.dirname(en_token_file))

        with open(en_file, 'r', encoding='utf-8') as en, open(en_token_file, 'w', encoding='utf-8') as out:
            for line in en:
                if self.use_bert_tokenizer:
                    tokens = self.tokenize_line_with_bert(line.strip())
                else:
                    tokens = self.en_tokenizer.encode(line.strip()).tokens
                print(' '.join(tokens), file=out)

        # If use_bert_tokenizer is True, avoid tokenizing am_file with BERT
        if not self.use_bert_tokenizer:
            with open(am_file, 'r', encoding='utf-8') as am, open(am_token_file, 'w', encoding='utf-8') as out:
                for line in am:
                    tokens = self.am_tokenizer.encode(line.strip()).tokens
                    print(' '.join(tokens), file=out)

        def consumer_wrapper(data_builder):
            def _add(item):
                data_builder.add_item(item)

            return _add

        if self.use_bert_tokenizer:
            en_dict = self.build_dictionary(en_token_file, self.tokenize_line_with_bert)
        else:
            en_dict = self.build_dictionary(en_token_file, self.en_tokenizer)
            am_dict = self.build_dictionary(am_token_file, self.am_tokenizer)

        # save to dictionary if it is a train file
        if am_dict is not None:
            self.save_dictionary(am_dict, os.path.join(os.path.dirname(pre_fix), am_dict_filename), pre_fix)
        self.save_dictionary(en_dict, os.path.join(os.path.dirname(pre_fix), en_dict_filename), pre_fix)

        if en_dict is not None:
            en_dataset_dest_file = os.path.join(pre_fix + ".en")
            en_ds = indexed_dataset.make_builder(en_dataset_dest_file + ".bin", imp)
            en_consumer = consumer_wrapper(en_ds)
            Binarizer.binarize(en_token_file, en_dict, en_consumer, append_eos=True, reverse_order=False)
            en_ds.finalize(en_dataset_dest_file + ".idx")

        if am_dict is not None:
            am_dataset_dest_file = os.path.join(pre_fix + ".am")
            am_ds = indexed_dataset.make_builder(am_dataset_dest_file + ".bin", imp)
            am_consumer = consumer_wrapper(am_ds)
            Binarizer.binarize(am_token_file, am_dict, am_consumer, append_eos=True, reverse_order=False)
            am_ds.finalize(am_dataset_dest_file + ".idx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess data for fairseq")
    parser.add_argument('--en_file', type=str, required=True, help='Path to the English file')
    parser.add_argument('--am_file', type=str, required=True, help='Path to the Amharic file')
    parser.add_argument('--implementation', type=str, default=None, help='dataset build type')
    parser.add_argument('--data_bin_path', type=str, required=True, help='Path to the data bin directory')
    parser.add_argument('--task_file', type=str, required=True, choices=['train.en-am', 'train.bert.en-am',
                                                                         'test.en-am', 'valid.en-am',
                                                                         'test.bert.en-am', 'valid.bert.en-am'],
                        help='Task file name')
    parser.add_argument('--use_bert_tokenizer', action='store_true',
                        help='Use BERT tokenizer for English data.')

    args = parser.parse_args()

    processor = BilingualDataPreprocessor(use_bert_tokenizer=args.use_bert_tokenizer)

    en_file_absolute = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), args.en_file)
    am_file_absolute = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), args.am_file)

    processor.train_tokenizer([en_file_absolute, am_file_absolute])

    prefix = os.path.join(args.data_bin_path, args.task_file)
    processor.preprocess_data(args.en_file, args.am_file, args.implementation, prefix)
