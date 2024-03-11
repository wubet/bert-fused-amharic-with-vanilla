# Vanilla Transformer Model

This model is a basic Vanilla Transformer architecture developed using the fairseq toolkit, it is tailored for bidirectional English-Amharic sequence translation. It leverages attention mechanisms, shifting from the previously dominant recurrent neural network (RNN) frameworks.

### Requirements and Installation:

PyTorch version >= 1.13.0 \
Python version >= 3.8

Cloning and Building the Repository:

Repo cloning:
```commandline
git clone https://github.com/wubet/bert-fused-amharic-with-vanilla.git
```
Installs all the dependencies
```commandline
pip3 install -r requirements.txt
```

Installs the build module, a modern Python package builder
```commandline
pip3 install build
```

Builds the package, generating distribution archives (wheel and source) in the dist directory
```commandline
python3 -m build
```

Installs the package in editable mode (also known as development mode), allowing changes to the code to be immediately reflected
```commandline
pip3 install --editable .
```

Compiles and builds any extension modules (e.g., C extensions) specified in setup.py directly in the source directory
```commandline
python3 setup.py build_ext –inplace
```

### Data Preparation

Clone the English-Amharic corpus.
```commandline
Git clone https://github.com/wubet/unified-amharic-english-corpus.git
```

Process the corpus data for training and evaluation. Typically, processing encompasses various crucial stages to adeptly navigate the intricacies of language. Such stages encompass breaking down words into subwords or tokens, mapping these tokens to a specific vocabulary, and incorporating special tokens, all aimed at enhancing the model's proficiency in dealing with infrequent words and morphological differences.

Training data:
```commandline
python3 data/bilingual_data_processor.py \
--en_file="unified-amharic-english-corpus/datasets/train.am-en.base.en" \
--am_file="unified-amharic-english-corpus/datasets/train.am-en.transliteration.am" \
--implementation="mmap" \
--data_bin_path="data-bin/vanilla/wmt23_en_am" \
--task_file="train.en-am"
```

Validation Data:
```commandline
python3 data/bilingual_data_processor.py \
--en_file="unified-amharic-english-corpus/datasets/dev.am-en.base.en" \
--am_file="unified-amharic-english-corpus/datasets/dev.am-en.transliteration.am" \
--implementation="mmap" \
--data_bin_path="data-bin/vanilla/wmt23_en_am" \
--task_file="valid.en-am"
```

Test Data:
```commandline
python3 data/bilingual_data_processor.py \
--en_file="unified-amharic-english-corpus/datasets/test.am-en.base.en" \
--am_file="unified-amharic-english-corpus/datasets/test.am-en.transliteration.am" \
--implementation="mmap" \
--data_bin_path="data-bin/vanilla/wmt23_en_am" \
--task_file="test.en-am"
```
Preparing Amharic translitration file for training
```commandline
python3 ../translitration/create_transliteration.py
--source_filenames=unified-amharic-english-corpus/datasets/train.am-en.base.en \
--target_filenames=unified-amharic-english-corpus/datasets/train.am-en.transliteration.am
```

Preparing Amharic translitration file for testing
```commandline
python3 ../translitration/create_transliteration.py
--source_filenames=unified-amharic-english-corpus/datasets/test.am-en.base.en \
--target_filenames=unified-amharic-english-corpus/datasets/test.am-en.transliteration.am
```

### Train The Model
In order to train the model use the following command:
```commandline
python3 fairseq_cli/train
--arch=vanilla_transformer_wmt_en_am \
--source-lang=en \
--target-lang=am \
--share-decoder-input-output-embed \
--optimizer=adam \
--adam-betas="(0.9,0.98)" \
--clip-norm=0.0 \
--lr=5e-4 \
--lr-scheduler=inverse_sqrt \
--warmup-updates=4000 \
--dropout=0.3 \
--weight-decay=0.0001 \
--criterion=label_smoothed_cross_entropy \
--label-smoothing=0.1 \
--max-tokens=4096 \
--max-update=1500000 \
--save-interval-updates=1000 \
--keep-interval-updates=5 \
data-bin/vanilla/wmt23_en_am
```