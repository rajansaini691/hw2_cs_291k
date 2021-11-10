import subword_nmt
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

from io import StringIO
from string import ascii_lowercase, ascii_uppercase
from itertools import chain
import os
import numpy as np
import xml.etree.ElementTree as ET
import time
import random
import pickle
import gc

import torch
from torch.profiler import profile, record_function, ProfilerActivity


OUT_DIR = "./.out/"
TRAIN_CORPUS_PATH = "./opus.ha-en.tsv"
VAL_CORPUS_PATH = "./newsdev2021.en-ha.xml"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

torch.cuda.empty_cache()


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

    # Insert special tokens
    vocab_size = len(token_dict.keys())
    for i, tok in enumerate(['BOS', 'EOS', 'PAD']):
        token_dict[tok] = i + vocab_size

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
            if len(tensor_lang_1) < 30 and len(tensor_lang_2) < 30:
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
        and os.path.exists(OUT_DIR + "token_dict"):
        print("Loading cached train/val data...")
        return [torch.load(OUT_DIR + "train_data"), torch.load(OUT_DIR + "val_data"), pickle.load(open(OUT_DIR + "token_dict", "rb"))]

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
    pickle.dump(token_dict, open(OUT_DIR + "token_dict", "wb"))

    return [train_data, val_data, token_dict]

"""
Training helpers (normally these would be in a separate file, but I wasn't sure whether
this was allowed for this assignment)
"""
class TransformerLSTMCustom(torch.nn.Module):
    def __init__(self, token_dict):
        super(TransformerLSTMCustom, self).__init__()
        self.vocab_size = len(token_dict.keys())
        self.token_dict = token_dict
        self.embedding = torch.nn.Embedding(self.vocab_size, 256)
        self.transformer1 = torch.nn.TransformerEncoderLayer(
                d_model=256, nhead=8, dim_feedforward=1024, batch_first=True)
        self.transformer2 = torch.nn.TransformerEncoderLayer(
                d_model=256, nhead=8, dim_feedforward=1024, batch_first=True)
        self.attention = torch.nn.MultiheadAttention(
                embed_dim=256, num_heads=1, batch_first=True)
        self.lstm = torch.nn.LSTM(
                input_size=256, hidden_size=256, num_layers=2, batch_first=True)
        self.softmax = torch.nn.Softmax(dim=2)  # Or should it be 0? idk
        self.linear = torch.nn.Linear(256, self.vocab_size)
        self.layer_norm = torch.nn.LayerNorm(256)

    def forward_encoder(self, src):
        src_embed = self.embedding(src)
        encdr_hid = self.transformer1(src_embed)
        encdr_hid = self.layer_norm(encdr_hid)
        encdr_out = self.transformer2(encdr_hid)
        return encdr_out

    def forward_decoder(self, lstm_hidden, src, tgt, encdr_out):
        tgt_embed = self.embedding(tgt)
        context, _ = self.attention(query=tgt_embed, key=encdr_out, value=encdr_out, need_weights=False)
        # TODO Could try running target through another lstm, too, and then concatenating with ctx 
        # Only combine with context for first token
        context = self.layer_norm(context)
        lstm_input = torch.cat((context, tgt_embed), 1) if lstm_hidden is None else tgt_embed
        lstm_out, hidden = self.lstm(lstm_input, lstm_hidden)
        # TODO What should the lstm hidden size(s) be?
        skip = self.layer_norm(lstm_out + context + tgt_embed)
        model_outputs = self.linear(skip)
        return model_outputs[:,-1,:], hidden

    def forward(self, src, tgt):
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)
        encdr_hid = self.transformer1(src_embed)
        encdr_out = self.transformer2(encdr_hid)
        context, _ = self.attention(query=tgt_embed, key=encdr_out, value=encdr_out, need_weights=False)
        lstm_out, _ = self.lstm(torch.cat((context, tgt_embed),1))
        # TODO What should the lstm hidden size(s) be?
        model_outputs = self.linear(lstm_out)
        return model_outputs[:,-1,:]
    
    def train(self, src, tgt, optimizer, criterion):
        optimizer.zero_grad()

        tgt_len = tgt.size(1)
        if tgt_len < 2:
            return 0
        loss = 0

        # Teacher forcing
        encdr_out = self.forward_encoder(src)
        hidden = None           # hidden layers for lstm
        for di in range(0, tgt_len):
            tgt_token = tgt[:,di-1].unsqueeze(1)
            x, hidden = self.forward_decoder(hidden, src, tgt_token, encdr_out)
            y = tgt[:,di]
            loss += criterion(x, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
        optimizer.step()
        return loss.item() / tgt_len

class TransformerLSTM(torch.nn.Module):
    def __init__(self, token_dict):
        super(TransformerLSTM, self).__init__()
        self.token_dict = token_dict
        self.vocab_size = len(token_dict.keys())
        self.embedding = torch.nn.Embedding(self.vocab_size, 256)
        self.lstm = torch.nn.LSTM(
                input_size=256, hidden_size=256, num_layers=2, batch_first=True)
        self.transformer = torch.nn.Transformer(d_model=256, nhead=8,
                num_encoder_layers=2, dim_feedforward=1024, num_decoder_layers=1,
                custom_decoder=self.custom_decoder, batch_first=True)
        self.linear = torch.nn.Linear(256, self.vocab_size)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.layer_norm = torch.nn.LayerNorm(256)

    def custom_decoder(self, tgt, memory, **kwargs):
        memory_len = memory.size(1)
        lstm_out, _ = self.lstm(torch.cat((memory, tgt), 1))
        dropout_lstm_out = self.dropout(lstm_out[:,memory_len:,:])
        normalized_lstm_out = self.layer_norm(dropout_lstm_out)
        return self.linear(normalized_lstm_out + tgt)

    def add_bos(self, tgt):
        bos = torch.tensor(self.token_dict['BOS'])
        bos = torch.tile(bos, (tgt.size(0), 1))
        bos = bos.to(device)
        return torch.cat((bos, tgt), 1)

    def add_eos(self, tgt):
        eos = torch.tensor(self.token_dict['EOS'])
        eos = torch.tile(eos, (tgt.size(0), 1))
        eos = eos.to(device)
        return torch.cat((tgt, eos), 1)

    def forward(self, src, tgt):
        tgt = self.add_bos(tgt)
        tgt = self.add_eos(tgt)
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)
        y_pred = self.transformer.forward(src_embed, tgt_embed)
        return y_pred

    def train(self, src, tgt, optimizer, criterion):
        optimizer.zero_grad()

        tgt_len = tgt.size(1)
        if tgt_len < 2:
            return 0
        loss = 0

        tgt = self.add_bos(tgt)
        tgt = self.add_eos(tgt)
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)
        src_embed_dropout = self.dropout(src_embed)
        tgt_embed_dropout = self.dropout(tgt_embed)
        y_pred = self.transformer.forward(src_embed_dropout, tgt_embed_dropout)
        for di in range(1, tgt_len):
            loss += criterion(y_pred[:,di-1,:],tgt[:,di])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
        optimizer.step()
        return loss.item() / tgt_len

"""
Validation
"""
def validate(model, val_data):
    """
    Validate model against unseen data
    """
    criterion = torch.nn.CrossEntropyLoss()
    total_error = 0
    for x, y in val_data:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        x = x.to(device)
        y = y.to(device)
        y_len = y.size(1)
        y_pred = model.forward(x, y)
        error = 0
        for di in range(0, y_len):
            error += criterion(y_pred[:,di-1,:],y[:,di])
        error /= y_len
        total_error += error
    return (total_error / len(val_data)).item()
    
"""
Main entrypoint
"""
def main():
    # Preprocess data
    with open(TRAIN_CORPUS_PATH, "r") as train_corpus_file:
        with open(VAL_CORPUS_PATH, "r") as val_corpus_file:
            train_data, val_data, token_dict = preprocess_data(train_corpus_file, xml2tsv(val_corpus_file))
    print("Preprocessed and loaded dataset!")
    
    # Create model
    model = TransformerLSTM(token_dict)
    model = model.to(device)

    # Train model
    print("Begin training")
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)     # lambda taken from paper on RNMT+
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = 32
    loss = 0

    # Data
    train_history = dict()
    val_history = dict()

    for epoch in range(1):
        random.shuffle(train_data)
        for iteration in range(0, len(train_data)-batch_size, batch_size):
            # Read a batch, pad it, and load it onto a GPU
            src = [train_data[i][0] for i in range(iteration, iteration+batch_size)]
            tgt = [train_data[i][1] for i in range(iteration, iteration+batch_size)]
            src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=token_dict['PAD'])
            tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=token_dict['PAD'])
            src = src.to(device)
            tgt = tgt.to(device)

            # Run one train step (compute loss, gradient update)
            loss = model.train(src, tgt, optimizer, criterion)

            del src
            del tgt
            
            if iteration % 512 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            if iteration % 2048 == 0:
                val_loss = validate(model, val_data)

                print(f"{iteration}/{len(train_data)}")
                print(f"VAL - {val_loss}")
                print(f"TRAIN - {loss}")

                iter_since_beginning = len(train_data) * epoch + iteration
                train_history[iter_since_beginning] = loss
                val_history[iter_since_beginning] = val_loss

        print("Train History")
        print(train_history)
        print("Val History")
        print(val_history)
    torch.save(model, OUT_DIR + "model")


if __name__ == "__main__":
    main()
