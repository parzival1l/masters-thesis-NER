import torch
import torch.nn as nn
# from .crf.crf import crf
from torchcrf import CRF
import torch.optim as optim
from transformers import (
    AutoModel,
    CanineModel,
    BertModel,
    BertConfig,
    T5ForConditionalGeneration,
    AdamW,
    get_scheduler
)
from datasets import disable_caching
import warnings
import torch
import torch.nn.functional as F


warnings.filterwarnings("ignore")
disable_caching()

class IngredientLabeller(torch.nn.Module):

    def __init__(self, tokenizer, exp_config):
        super(IngredientLabeller, self).__init__()
        self.exp_config = exp_config
        self.l1 = BertModel.from_pretrained(exp_config['word_emb_model'])
        # self.l1.load_state_dict(torch.load(exp_config["word_emb_state_dict"]),strict=False)
        self.l2 = CanineModel.from_pretrained(exp_config['char_emb_model'])
        self.num_tags = exp_config['num_tags']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Freeze word and character embedding layers
        if exp_config['freeze_word_emb'] :
            for param in self.l1.parameters():
                param.requires_grad = False
        if exp_config['freeze_char_emb'] :
            for param in self.l2.parameters():
                param.requires_grad = False

        hidden_dim = 512
        self.POS_ON = self.exp_config['POS_ON']
        extra_layer = 1 if self.POS_ON is True else 0
        self.bilstm = nn.LSTM(exp_config['final_emb_size'] + extra_layer , hidden_dim // 2, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.num_tags)
        self.crf = CRF(self.num_tags, batch_first=True)
        self.CRF_ON = False

    def forward(self, input_ids, attention_mask, token_type_ids, token_spans , char_input_ids, char_attention_mask, char_token_type_ids, labels, POS):

        emissions = self.tag_outputs(input_ids, attention_mask, token_type_ids, token_spans , char_input_ids, char_attention_mask, char_token_type_ids, labels, POS)
        if self.CRF_ON :
            log_likelihood = self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            loss = -log_likelihood  # Negative log-likelihood
        else :
            loss = F.cross_entropy(emissions.view(-1, self.num_tags), labels.view(-1), ignore_index=-100)
        return loss

    def tag_outputs(self, input_ids, attention_mask, token_type_ids, token_spans, char_input_ids, char_attention_mask, char_token_type_ids, labels, POS = None) :

        batch_size = input_ids.shape[0]
        # word_embs = self.l1(input_ids= input_ids, attention_mask= attention_mask, labels = labels).encoder_last_hidden_state
        word_embs = self.l1(input_ids= input_ids, attention_mask= attention_mask, token_type_ids = token_type_ids).last_hidden_state
        _char_embs = self.l2(input_ids = char_input_ids, attention_mask = char_attention_mask, token_type_ids = char_token_type_ids).last_hidden_state

        mean_chars = torch.zeros((word_embs.shape[0], word_embs.shape[1], 2048)).to(self.device)
        for i in range(word_embs.shape[0]):
            for j in range(word_embs.shape[1]):
                start, end = token_spans[i, j]
                if start != -1:
                    mean_chars[i, j] = _char_embs[i, start:end+1].mean(dim=0).mean(dim=0) #+1 for CLS token, +2 for CLS and SEP token

        combined_embs = torch.cat((word_embs, mean_chars), dim=2)
        if self.POS_ON :
            combined_embs = torch.cat((combined_embs, POS.unsqueeze(2)), dim=2)
        lstm_out, _ = self.bilstm(combined_embs)
        emissions = self.hidden2tag(lstm_out)

        return emissions

    def decode(self, input_ids, attention_mask, token_type_ids, token_spans, char_input_ids, char_attention_mask, char_token_type_ids, POS=None, labels = None) :

        emissions = self.tag_outputs(input_ids, attention_mask, token_type_ids, token_spans , char_input_ids, char_attention_mask, char_token_type_ids, labels=labels)
        if self.CRF_ON :
            predictions =  self.crf.decode(emissions, attention_mask.byte())
            desired_shape = (self.exp_config['dataloader']['batch_size'], self.exp_config['tokenizer']['word_max_length'])
            predictions = self.convert_to_tensor(predictions, desired_shape)
        else :
            predictions = emissions.argmax(dim=-1)
            # active_logits = emissions.view(-1, self.num_tags) # shape (batch_size * seq_len, num_labels)
            # flattened_predictions = torch.argmax(active_logits, axis=1)
        return predictions , labels

    def convert_to_tensor(self, input_list, desired_shape):
        output_tensor = torch.zeros(desired_shape)
        for i, inner_list in enumerate(input_list):
            output_tensor[i, :len(inner_list)] = torch.tensor(inner_list)
        return output_tensor
