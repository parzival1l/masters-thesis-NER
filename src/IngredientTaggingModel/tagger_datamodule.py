from typing import Optional
from typing import Any, Dict, Optional
import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset, disable_caching
import warnings
from omegaconf import DictConfig
from util.defaults import load_from_json, Default
from transformers import (
    AutoTokenizer,
    CanineTokenizer,
    BertTokenizer,
    T5Tokenizer
)
import spacy
warnings.filterwarnings("ignore")
disable_caching()
tqdm.pandas()

class TaggerDataModule() :
    def __init__(self,
                exp_cfg: DictConfig,

            ) -> None:
        """
        """

        self.exp_cfg = exp_cfg
        self.data_path = exp_cfg['data']['datapath']
        self.sample_size =  exp_cfg['data']['sample_size'] if exp_cfg['data']['sample_size'] is not None else 1000
        self.essential_cols = Default.essential_cols
        self.pos2index   = load_from_json(self.exp_cfg['data']['pos2index'])
        self.label2index = load_from_json(self.exp_cfg['data']['label2index'])
        self.index2label = {v:k for k,v in self.label2index.items()}
        self.word_tokenizer = AutoTokenizer.from_pretrained(exp_cfg['word_tokenizer'])
        self.char_tokenizer = CanineTokenizer.from_pretrained(exp_cfg['char_emb_model'])
        self.nlp = spacy.load("en_core_web_sm")
        # fractions = Default.fractions
        # decimals = Default.decimals
        # self.word_tokenizer.add_tokens(fractions)
        # self.word_tokenizer.add_tokens(decimals)

    def _get_tokenizer(self) :
        return self.word_tokenizer

    def _getindex2label(self) :
        return self.index2label

    def load_data(self, data_path : Optional[str] = None):
        """
        Load the data from the path specified in the config file
        """
        if data_path is None :
            self.df = pd.read_csv(self.data_path, nrows=self.sample_size)
        else :
            self.df = pd.read_csv(data_path)

    def prepare_data(self, primary_key : Optional[str] = None) -> Dict[str, Any]:
        """
        """
        if primary_key is not None :
            pass
        elif ['recipeid'] in self.df.columns.values and ['ingredientid'] in self.df.columns.values :
            self.df["primary_key"] = self.df.apply(lambda x : str(x["recipeid"]) + "_" + str(x["ingredientid"]), axis=1)
            primary_key = "primary_key"
        else :
            raise Exception(f"Neither primary_key nor {['recipeid', 'ingredientid']} not found in dataframe. Please check the column names in the dataframe")
        grouped_data = self.df.groupby(primary_key)

        sequences = []
        for sequence_id, group in tqdm(grouped_data):
            sequence = group['IngredientTxt'].tolist()
            sequence_tokens = group['TextToken'].tolist()
            sequence_pos_tags = group['PosStr'].tolist()
            sequence_iob_labels = group['IOBLabel'].tolist()
            pos_tags = [self.pos2index[tag] for tag in sequence_pos_tags]
            try :
                iob_labels = [self.label2index[label] for label in sequence_iob_labels]
            except KeyError :
                print (sequence_tokens, sequence_iob_labels, sequence_id)

            sequences.append({
                'id' : sequence_id,
                "sentence" :sequence[0],
                'pos_tags' : pos_tags,
                'tokens' : group['TextToken'].tolist(),
                'ner_tags' : iob_labels
            })
        return sequences

    def create_dataset_obj(self, sequences : list , split : bool , test_size : Optional[float] = 0.2, shuffle : Optional[bool] = False) :
        data_dict = { key: [entry[key] for entry in sequences] for key in sequences[0]}
        dataset = Dataset.from_dict(data_dict)
        if split :
            dataset = dataset.train_test_split(test_size = test_size, shuffle=shuffle )
        return dataset

    def tokenize_and_align_labels(self, examples : dict , if_label : bool = True):

        tokenized_inputs = self.word_tokenizer(examples["tokens"], padding=self.exp_cfg['tokenizer']['padding'], max_length = self.exp_cfg['tokenizer']['word_max_length'], truncation=True, is_split_into_words=True, add_special_tokens=self.exp_cfg['tokenizer']['add_special_tokens'])
        char_inputs = self.char_tokenizer(examples["sentence"], padding=self.exp_cfg['tokenizer']['padding'], max_length = self.exp_cfg['tokenizer']['char_max_length'], truncation=True, add_special_tokens=self.exp_cfg['tokenizer']['add_special_tokens'])
        keys = list(char_inputs.keys())
        labels = [] #list for storing the re-mapped labels
        pos_tags = []
        token_spans_sent = [] #list for storing the token spans for each sentence
        sentences = []  #list for storing the sentences
        _pos = examples['pos_tags']
        if if_label :
            # if the label is not present in the data, then we will not be able to create the labels (test for custom examples)
            for i, label in enumerate(examples[self.exp_cfg['tokenizer']['label_column']]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                pos_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(self.exp_cfg['tokenizer']['special_token_label'])
                        pos_ids.append(self.exp_cfg['tokenizer']['special_token_label'])
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                        pos_ids.append(_pos[i][word_idx])
                    else:
                        label_ids.append(self.exp_cfg['tokenizer']['special_token_label'])
                        pos_ids.append(self.exp_cfg['tokenizer']['special_token_label'])
                    previous_word_idx = word_idx
                labels.append(label_ids)
                pos_tags.append(pos_ids)
                #Align token spans with the sentences
                sent_ = examples["sentence"][i]
                spans_ = []
                pos_ = 0
                for word in self.word_tokenizer.convert_ids_to_tokens(tokenized_inputs.input_ids[i], skip_special_tokens=True) :
                    word = word.replace("##", "")
                    start = sent_.find(word) ; end  = start + len(word)
                    spans_.append((start + pos_ , end + pos_))
                    sent_ = sent_[end:]
                    pos_ += end

                new_tensor = torch.full((self.exp_cfg['tokenizer']['word_max_length'], 2), -1, dtype=torch.long)
                spans_ = torch.tensor(spans_)
                new_tensor[:spans_.shape[0], :] = spans_
                token_spans_sent.append(new_tensor)

        for key in keys :
            tokenized_inputs["char_" + key] = char_inputs[key]
        tokenized_inputs["token_spans"] = token_spans_sent

        if if_label == True :
           tokenized_inputs["labels"] = labels
           tokenized_inputs["POS"] = pos_tags

        return tokenized_inputs

    def create_dataloader(self, dataset , batch_size : int,  shuffle : bool = True) -> DataLoader :
        """
        Create a dataloader for the dataset
        """
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


    def setup(self, env : str = "train") -> DataLoader:

        kwargs = {
            "split" : True
        }
        if env == "train" :
            self.load_data()
            sequences = self.prepare_data()
        else :
            self.load_data(self.exp_cfg['data']['val_datapath'])
            sequences = self.prepare_data() #primary_key="Index"
            kwargs["split"] = False

        kwargs["sequences"] = sequences
        if env == "train" :
            train_test_split = self.exp_cfg['dataloader']['train_test_split']
            shuffle = self.exp_cfg['dataloader']['train_shuffle']
            kwargs["test_size"] = train_test_split
            kwargs["shuffle"] = shuffle

        dataset_obj = self.create_dataset_obj(**kwargs)
        tokenized_dataset = dataset_obj.map(self.tokenize_and_align_labels, batched=True)
        if env == "train" :
            set_features = set(tokenized_dataset['train'].features.keys())
        else :
            set_features = set(tokenized_dataset.features.keys())
        tokenized_dataset = tokenized_dataset.remove_columns(list(set_features.difference(self.essential_cols)))
        tokenized_dataset.set_format(type='torch')
        if env == "train" :
            train_dataloader = self.create_dataloader(tokenized_dataset['train'], batch_size=self.exp_cfg['dataloader']['batch_size'], shuffle=self.exp_cfg['dataloader']['train_shuffle'])
            test_dataloader  = self.create_dataloader(tokenized_dataset['test'], batch_size=self.exp_cfg['dataloader']['batch_size'], shuffle=self.exp_cfg['dataloader']['train_shuffle'])
            return train_dataloader, test_dataloader
        else :
            val_dataloader = self.create_dataloader(tokenized_dataset, batch_size=self.exp_cfg['dataloader']['batch_size'], shuffle=self.exp_cfg['dataloader']['val_shuffle'])
            return val_dataloader

    def tokenize_sent(self, sent : str ) :
        """
        """
        tokenized_inputs = self.word_tokenizer([sent], padding=self.exp_cfg['tokenizer']['padding'], max_length = self.exp_cfg['tokenizer']['word_max_length'], truncation=True, is_split_into_words=True, add_special_tokens=self.exp_cfg['tokenizer']['add_special_tokens'])
        char_inputs = self.char_tokenizer(sent, padding=self.exp_cfg['tokenizer']['padding'], max_length = self.exp_cfg['tokenizer']['char_max_length'], truncation=True, add_special_tokens=self.exp_cfg['tokenizer']['add_special_tokens'])
        keys = list(char_inputs.keys())
        #Align token spans with the sentences
        spans_ = []
        pos_ = 0
        for word in self.word_tokenizer.convert_ids_to_tokens(tokenized_inputs.input_ids, skip_special_tokens=True) :
            word = word.replace("##", "")
            start = sent.find(word) ; end  = start + len(word)
            spans_.append((start + pos_ , end + pos_))
            sent = sent[end:]
            pos_ += end

        new_tensor = torch.full((self.exp_cfg['tokenizer']['word_max_length'], 2), -1, dtype=torch.long)
        spans_ = torch.tensor(spans_)
        new_tensor[:spans_.shape[0], :] = spans_

        for key in keys :
            tokenized_inputs["char_" + key] = char_inputs[key]
        tokenized_inputs["token_spans"] = new_tensor.tolist()

        for k in list(tokenized_inputs.keys()) :
            tokenized_inputs[k] = torch.tensor(tokenized_inputs[k])

        return tokenized_inputs

    def tokenize_sentences(self, sentences : str ) :
        """
        """
        sequences = []
        for id, sentence in enumerate(sentences) :
            doc = self.nlp(sentence)
            token_ = [] ; pos_ = []
            for tokens in doc :
                token_.append(tokens.text)
                pos_.append(tokens.pos_)
            pos_tags = [self.pos2index[tag] for tag in pos_]

            sequences.append({
                'id' : id,
                "sentence" :sentence,
                'pos_tags' : pos_tags,
                'tokens' : token_,
                'ner_tags' : [1 for i in range(len(token_))]
            })

        kwargs = {
            "split" : False
        }

        kwargs["sequences"] = sequences
        dataset_obj = self.create_dataset_obj(**kwargs)
        tokenized_dataset = dataset_obj.map(self.tokenize_and_align_labels, batched=True)
        set_features = set(tokenized_dataset.features.keys())
        tokenized_dataset = tokenized_dataset.remove_columns(list(set_features.difference(self.essential_cols)))
        tokenized_dataset.set_format(type='torch')
        val_dataloader = self.create_dataloader(tokenized_dataset, batch_size=self.exp_cfg['dataloader']['batch_size'], shuffle=self.exp_cfg['dataloader']['val_shuffle'])
        return val_dataloader
