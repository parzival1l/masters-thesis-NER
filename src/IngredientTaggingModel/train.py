import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torchcrf import CRF
from .lightningtrainModule import trainNERmodel, testNERmodel
    #  IngredientNER
from IngredientTaggingModel.IngredientNER import IngredientLabeller
from IngredientTaggingModel.tagger_datamodule import TaggerDataModule
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    CanineModel,
    BertModel,
    BertConfig,
    AdamW,
    get_scheduler
)
class Trainer():
    def __init__(self, exp_config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_config = exp_config
        self._tdm = TaggerDataModule(exp_cfg = self.exp_config)

    def load_model(self, model_path) :

        loaded_model = IngredientLabeller(tokenizer=self._tdm._get_tokenizer(), exp_config=self.exp_config)
        state_ = torch.load(model_path, map_location=torch.device('cpu'))

        if "l2.char_embeddings.position_ids" in state_.keys():
            state_.pop("l2.char_embeddings.position_ids")

        if "l1.embeddings.position_ids" in state_.keys():
            state_.pop("l1.embeddings.position_ids")
        loaded_model.load_state_dict(state_)
        loaded_model.to(self.device)
        loaded_model.eval()
        return loaded_model

    def validate__(self, model, dataloader) :
        model.to(self.device)
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for _, data in enumerate(dataloader, 0):

                ids = data['input_ids'].to(self.device, dtype = torch.long)
                mask = data['attention_mask'].to(self.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
                targets = data['labels'].to(self.device, dtype = torch.long)
                char_ids = data['char_input_ids'].to(self.device, dtype = torch.long)
                char_mask = data['char_attention_mask'].to(self.device, dtype = torch.long)
                char_token_type_ids = data['char_token_type_ids'].to(self.device, dtype = torch.long)
                token_spans = data['token_spans'].to(self.device, dtype = torch.long)

                tag_space = model(input_ids= ids, attention_mask = mask, token_type_ids = token_type_ids, token_spans = token_spans, char_ids = char_ids, char_attention_mask = char_mask, char_token_type_ids = char_token_type_ids)
                predicted_labels = tag_space.argmax(dim=-1)

                # for index in range(ids.shape[0]) :
                #     print ("word Tokens :" ,word_tokenizer.convert_ids_to_tokens(ids[index]))
                #     print([index2label[t.item()] for t in predicted_labels[index]])
                #     result = []
                #     for t in targets[index]:
                #         if t.item() != -100:
                #             result.append(index2label[t.item()])
                #         else:
                #             result.append(-100)
                #     print(result)

                for index in range(ids.shape[0]) :
                    overall_ = 0
                    counter_ = 0
                    for value1, value2 in zip(predicted_labels[index], targets[index]):
                        if value2.item() != -100:
                            overall_ += 1
                            if value1.item() == value2.item():
                                counter_ += 1

                total_correct += counter_
                total_samples += overall_

        accuracy = total_correct / total_samples
        print(f'Validation Accuracy: {accuracy}')

        return accuracy

    def combine_word_pieces(self, tokens, token_predictions):

        special_tokens = ['[CLS]', '[SEP]', '[PAD]']
        filtered_tokens = [t for t in tokens if t not in special_tokens]
        filtered_ner_tags = [n for t, n in zip(tokens, token_predictions) if t not in special_tokens]

        combined_tokens = []
        combined_ner_tags = []
        current_token = ""
        current_ner_tag = None

        for token, ner_tag in zip(filtered_tokens, filtered_ner_tags):
            if not token.startswith("##"):
                if current_token:
                    combined_tokens.append(current_token)
                    combined_ner_tags.append(current_ner_tag)
                current_token = token
                current_ner_tag = ner_tag
            else:
                current_token += token[2:]
        if current_token:
            combined_tokens.append(current_token)
            combined_ner_tags.append(current_ner_tag)

        return combined_tokens, combined_ner_tags

    def train(self) :
        train_dataloader, test_dataloader = self._tdm.setup()
        val_dataloader = self._tdm.setup(env="val")
        dataloaders = {
                    "train_dataloader" : train_dataloader,
                    "test_dataloader" : test_dataloader,
                    "val_dataloader" : val_dataloader }

        trainNERmodel(dataloaders , self.exp_config, self._tdm._get_tokenizer())

    def validate(self) :

        val_dataloader = self._tdm.setup(env="val")
        testNERmodel(val_dataloader , self.exp_config, self._tdm._get_tokenizer(), self._tdm.index2label)

    def test_multiple(self, sentences, exp_config) :
        _tdm = TaggerDataModule(exp_cfg = exp_config)
        loaded_model = self.load_model(exp_config['model']['load_path'])
        self.get_sentence_predictions(sentences, _tdm, loaded_model)

    def test_single(self, sentence, exp_config) :
        _tdm = TaggerDataModule(exp_cfg = exp_config)
        loaded_model = self.load_model(exp_config['model']['load_path'])
        ner_ , tokens_ = self.test_sentence(sentence, loaded_model, _tdm)

    def get_sentence_predictions(self, sentences, _tdm, loaded_model) :
        dataloader = _tdm.tokenize_sentences(sentences)
        index2label = _tdm._getindex2label()
        with torch.no_grad():
            for _, inputs in enumerate(dataloader, 0):
                ids = inputs['input_ids'].to(self.device, dtype = torch.long)
                mask = inputs['attention_mask'].to(self.device, dtype = torch.long)
                token_type_ids = inputs['token_type_ids'].to(self.device, dtype = torch.long)
                char_ids = inputs['char_input_ids'].to(self.device, dtype = torch.long)
                char_mask = inputs['char_attention_mask'].to(self.device, dtype = torch.long)
                char_token_type_ids = inputs['char_token_type_ids'].to(self.device, dtype = torch.long)
                token_spans = inputs['token_spans'].to(self.device, dtype = torch.long)

                tag_space = loaded_model.decode(input_ids= ids, attention_mask = mask, token_type_ids = token_type_ids, token_spans = token_spans, char_input_ids = char_ids, char_attention_mask = char_mask, char_token_type_ids = char_token_type_ids)
                for id in range(ids.shape[0]):
                    print("Sentence : ", sentences[_+id])
                    tokens = _tdm.word_tokenizer.convert_ids_to_tokens(ids[id].squeeze().tolist())
                    token_predictions = [index2label[i] for i in tag_space[0][id].cpu().numpy()]
                    combined_tokens, combined_ner_tags = self.combine_word_pieces(tokens, token_predictions)
                    for token, ner_tag in zip(combined_tokens, combined_ner_tags):
                        print((token, ner_tag), end= ",")
                    print("\n")

    def test_sentence(self, sentence, loaded_model, _tdm) :
        inputs = _tdm.tokenize_sent(sentence)
        index2label = _tdm._getindex2label()
        with torch.no_grad():
            ids = inputs['input_ids'].unsqueeze(0).to(self.device, dtype = torch.long)
            mask = inputs['attention_mask'].unsqueeze(0).to(self.device, dtype = torch.long)
            token_type_ids = inputs['token_type_ids'].unsqueeze(0).to(self.device, dtype = torch.long)
            char_ids = inputs['char_input_ids'].unsqueeze(0).to(self.device, dtype = torch.long)
            char_mask = inputs['char_attention_mask'].unsqueeze(0).to(self.device, dtype = torch.long)
            char_token_type_ids = inputs['char_token_type_ids'].unsqueeze(0).to(self.device, dtype = torch.long)
            token_spans = inputs['token_spans'].unsqueeze(0).to(self.device, dtype = torch.long)

            tag_space = loaded_model.decode(input_ids= ids, attention_mask = mask, token_type_ids = token_type_ids, token_spans = token_spans, char_input_ids = char_ids, char_attention_mask = char_mask, char_token_type_ids = char_token_type_ids)
            print("Sentence : ", sentence)
            tokens = _tdm.word_tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
            token_predictions = [index2label[i] for i in tag_space[0][0].cpu().numpy()]
            combined_tokens, combined_ner_tags = self.combine_word_pieces(tokens, token_predictions)
            for token, ner_tag in zip(combined_tokens, combined_ner_tags):
                print((token, ner_tag), end= ",")
        return combined_tokens, combined_ner_tags
