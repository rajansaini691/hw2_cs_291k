import subword_nmt
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

from io import StringIO
from string import ascii_lowercase, ascii_uppercase
from itertools import chain
import os
import numpy as np
import xml.etree.ElementTree as ET

import torch

OUT_DIR = "./.out/"
TRAIN_CORPUS_PATH = "./opus.ha-en.tsv"
VAL_CORPUS_PATH = "./newsdev2021.en-ha.xml"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

"""
Preprocessing helpers (normally these would be in a separate file, but I wasn't sure whether
this was allowed for this assignment)
"""
def format_tsv_for_tokenization(infile):
    """
    Ensure that the tsv can be tokenized by removing extraneous words such as TICO
    """
    outfile = StringIO()
    for line in infile:
        try:
            sentences = line.strip().split('\t')
            outfile.write(f"{sentences[0]}\t{sentences[1]}\n")
        except IndexError:
            pass
    outfile.seek(0)
    return outfile

def tabs_to_newlines(infile, write_path):
    outfile = open(write_path, "w")
    for line in infile:
        outfile.write(line.replace("\t", "\n"))
    return outfile

def generate_vocab(corpus_file):
    """
    Uses byte-pair encoding to build a vocabulary from the corpus
    Returns a file object pointing to the generated codes file
    """
    path_to_codes_file = OUT_DIR + "codes_file"
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if os.path.exists(path_to_codes_file):
        with open(path_to_codes_file) as codes_file:
            return codes_file

    corpus_file = open(corpus_file.name)
    split_corpus_file = tabs_to_newlines(corpus_file, OUT_DIR + "split_corpus")
    split_corpus_file = open(split_corpus_file.name, "r")
    codes_file = open(path_to_codes_file, "w")
    learn_bpe(split_corpus_file, codes_file, 10000)
    return codes_file


def tokenize_dataset(infile, codes_file, prefix=""):
    """
    Use a vocabulary to tokenize the dataset
    """
    path_to_tokenized_file = OUT_DIR + prefix + "_tokenized_corpus"
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if os.path.exists(path_to_tokenized_file):
        with open(path_to_tokenized_file) as codes_file:
            return codes_file
    tokenized_corpus = open(path_to_tokenized_file, "w")
    codes_file = open(codes_file.name, "r")
    infile.seek(0)
    codes_file.seek(0)

    bpe = BPE(codes_file)
    for line in infile:
        tokenized_line = bpe.process_line(line)
        tokenized_corpus.writelines(tokenized_line)
    return tokenized_corpus

def create_token_dict(codes_file):
    """
    Creates a dictionary mapping subwords to integers
    """
    codes_file = open(codes_file.name, "r")
    codes_file.seek(0)
    next(codes_file)    # Skip first line (contains version info)

    token_dict = dict()

    # Insert alphabet first
    for i, x in enumerate(chain(ascii_lowercase, ascii_uppercase)):
        token_dict[x] = i
    for i, x in enumerate(chain(ascii_lowercase, ascii_uppercase)):
        token_dict[x + '</w>'] = i

    alphabet_len = len(token_dict.keys())

    # Insert bpe subwords
    for i, line in enumerate(codes_file):
        token = line.replace(' ', '').replace('\n', '')
        token_dict[token] = i + alphabet_len

    return token_dict
    
def create_tensor_from_sentence(sentence, token_dict):
    sentence_with_boundary_tokens = list(map(\
        lambda token: token[:-2] if token.endswith("@@") else token + "</w>",
        sentence.replace('\n', '').split(' ')))
    sentence_as_indices = []
    for token in sentence_with_boundary_tokens:
        try:
            sentence_as_indices += [token_dict[token]]
        except KeyError as e:
            pass
    return torch.tensor(sentence_as_indices, dtype=torch.long)

def create_tensors(tokenized_corpus_file, token_dict):
    """
    Create pytorch-processable dataset from corpus
    (return list of seq2seq tensors)
    """
    tokenized_corpus_file = open(tokenized_corpus_file.name)
    data = []
    for line in tokenized_corpus_file:
        try:
            sentence_lang_1, sentence_lang_2 = line.split('\t')
            tensor_lang_1 = create_tensor_from_sentence(sentence_lang_1, token_dict)
            tensor_lang_2 = create_tensor_from_sentence(sentence_lang_2, token_dict)
            data.append((tensor_lang_1, tensor_lang_2))
        except ValueError:
            pass
    return data

def xml2tsv(xml_path):
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    val_tsv_file = open(OUT_DIR + "val_tsv", "w")

    tree = ET.parse(VAL_CORPUS_PATH)
    root = tree.getroot()
    for child in root:
        for src,ref in zip(child, child[1:]):
            for s1, s2 in zip(src.iter('seg'), ref.iter('seg')):
                val_tsv_file.write(f"{s1.text}\t{s2.text}\n")
    return val_tsv_file

def concat_files(filea, fileb, path):
    """
    Write concatenation of two files to the given path
    """
    out_fd = open(path, 'w')
    out_fd.write(filea.read())
    out_fd.write('\n')
    out_fd.write(fileb.read())
    return out_fd

# TODO Get dataset from parsed args
def preprocess_data(train_tsv_file, val_tsv_file):
    # Use cached data if already computed
    if os.path.exists(OUT_DIR + "train_data") and os.path.exists(OUT_DIR + "val_data") \
        and os.path.exists(OUT_DIR + "vocab_size.npy"):
        print("Loading cached train/val data...")
        return [torch.load(OUT_DIR + "train_data"), torch.load(OUT_DIR + "val_data"), np.load(OUT_DIR + "vocab_size.npy")]

    # Ensure file descriptors are open
    train_tsv_file = open(train_tsv_file.name, 'r')
    val_tsv_file = open(val_tsv_file.name, 'r')

    # Remove metadata from corpus
    print("Cleaning data...")
    cleaned_train_set = format_tsv_for_tokenization(train_tsv_file)
    cleaned_val_set = format_tsv_for_tokenization(val_tsv_file)

    # Run BPE tokenization
    print("Generating vocabulary...")
    concatenated_corpora = concat_files(cleaned_train_set, cleaned_val_set, OUT_DIR + 'train_val')
    codes_file = generate_vocab(concatenated_corpora)

    print("Tokenizing data...")
    tokenized_train_set = tokenize_dataset(cleaned_train_set, codes_file, prefix="train")
    tokenized_val_set = tokenize_dataset(cleaned_val_set, codes_file, "val")

    # Map tokens to index in vocabulary
    print("Creating token dictionary...")
    token_dict = create_token_dict(codes_file)

    # Create tensors for each set of parallel sentences
    print("Creating tensors...")
    train_data = create_tensors(tokenized_train_set, token_dict)
    val_data = create_tensors(tokenized_val_set, token_dict)

    # Cache tensors and vocab size
    torch.save(train_data, OUT_DIR + "train_data")
    torch.save(val_data, OUT_DIR + "val_data")
    np.save(OUT_DIR + "vocab_size", len(token_dict))

    return [train_data, val_data, len(token_dict)]

"""
Training helpers (normally these would be in a separate file, but I wasn't sure whether
this was allowed for this assignment)
"""
class TransformerLSTM(torch.nn.Module):
    def __init__(self, vocab_size):
        super(TransformerLSTM, self).__init__()
        self.vocab_size = int(vocab_size)
        self.embedding = torch.nn.Embedding(vocab_size, 256)
        self.transformer1 = torch.nn.TransformerEncoderLayer(
                d_model=256, nhead=8, dim_feedforward=1024, batch_first=True)
        self.transformer2 = torch.nn.TransformerEncoderLayer(
                d_model=256, nhead=8, dim_feedforward=1024, batch_first=True)
        self.attention = torch.nn.MultiheadAttention(
                embed_dim=256, num_heads=1, batch_first=True)
        self.lstm1 = torch.nn.LSTM(
                input_size=256, hidden_size=256, batch_first=True)
        self.lstm2 = torch.nn.LSTM(
                input_size=256, hidden_size=256, batch_first=True)
        self.softmax = torch.nn.Softmax(dim=2)  # Or should it be 0? idk
        self.linear = torch.nn.Linear(256, self.vocab_size)

    def forward(self, src, tgt):
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)
        encdr_hid = self.transformer1(src_embed)
        encdr_out = self.transformer2(encdr_hid)
        attn_out, _ = self.attention(query=tgt_embed, key=encdr_out, value=encdr_out, need_weights=False)
        # TODO Could try running target through another lstm, too, and then concatenating with attn_out
        lstm1_out, _ = self.lstm1(attn_out)
        # TODO What should the lstm hidden size(s) be?
        lstm2_out, _ = self.lstm2(lstm1_out)
        model_outputs = self.linear(lstm2_out)
        return self.softmax(model_outputs)
    
    def train(self, src, tgt, optimizer, criterion):
        # Run teacher-forcing, compute loss, compute gradient

        optimizer.zero_grad()

        tgt_len = tgt.size(1)
        tgt_one_hot = torch.nn.functional.one_hot(tgt, num_classes=self.vocab_size)

        loss = 0

        # Run a series of forward passes
        # TODO Keep lstm hidden state <-- may not train without this
        for di in range(1, tgt_len):
            model_output = self.forward(src, tgt[:,:di])
            x = model_output[:,-1,:]    # Keep only last lstm output
            y = tgt[:,di]
            loss += criterion(x, y)
        loss.backward()

        optimizer.step()

        return loss.item() / tgt_len
    
"""
Main entrypoint
"""
def main():
    # Preprocess data
    with open(TRAIN_CORPUS_PATH, "r") as train_corpus_file:
        with open(VAL_CORPUS_PATH, "r") as val_corpus_file:
            train_data, val_data, vocab_size = preprocess_data(train_corpus_file, xml2tsv(val_corpus_file))
    print("Preprocessed and loaded dataset!")
    
    # TODO Batch data
    # Create model
    model = TransformerLSTM(vocab_size)

    # Train model
    print("Begin training")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    for epoch in range(1, 20):
        for iteration in range(len(train_data)):
            src = train_data[0][0].unsqueeze(0)
            tgt = train_data[0][1].unsqueeze(0)
            loss = model.train(src, tgt, optimizer, criterion)
            print(loss)




if __name__ == "__main__":
    main()
