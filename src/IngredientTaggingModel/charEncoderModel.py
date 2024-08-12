import sys
from typing import Any
from lightning.pytorch import LightningModule
from .IngredientNER import IngredientLabeller
from torch import (nn, optim)
import torch.nn.functional as F
import torch
from util.defaults import load_from_json

class charEncoderModel(LightningModule):

       def __init__(self,
                     tokenizer,
                     exp_config,
                     **kwargs):
              super().__init__()
              self.adam_beta1=0.90
              self.adam_beta2=0.999
              self.adam_epsilon=1e-8
              self.exp_config = exp_config
              self.learning_rate = exp_config['learning_rate']
              self.label2index = load_from_json(self.exp_config['data']['label2index'])
              self.index2label = {v:k for k,v in self.label2index.items()}
              self.encoderModel = IngredientLabeller(tokenizer=tokenizer, exp_config=exp_config)
              self.save_hyperparameters()
              self.predictions = []

       def forward(self, x):
              pass

       def training_step(self, batch, batch_idx):
              batch = {k: v.to(self.device, dtype = torch.long) for k, v in batch.items()}
              loss = self.encoderModel(**batch)
              self.log('train_loss', loss, on_step=True, on_epoch=True)
              return loss

       def validation_step(self, batch, batch_idx):
              batch = {k: v.to(self.device, dtype = torch.long) for k, v in batch.items()}
              predictions, labels = self.encoderModel.decode(**batch)
              pad = (labels == -100)
              mask = ~pad
              predictions = predictions.to(self.device, dtype = torch.long)
              matches = predictions[mask] == labels[mask]
              accuracy = torch.mean(matches.float())
              self.log('val_acc', round(accuracy.item() , 3), on_step=False, on_epoch=True)
              for key, value in self.index2label.items():
                     if value not in ["O", "--PADDING--"] :
                            precision, recall = self._return_confusionmatrix(predictions, labels, key)
                            self.log(f'precision_{value}', round(precision.item() , 3), on_step=False, on_epoch=True)
                            self.log(f'recall_{value}', round(recall.item() , 3), on_step=False, on_epoch=True)

              return round(accuracy.item() , 3)

       def predict_step(self, batch, batch_idx) -> Any:
              batch = {k: v.to(self.device, dtype = torch.long) for k, v in batch.items()}
              predictions, labels = self.encoderModel.decode(**batch)
              self.predictions.append((predictions, labels))
              return predictions, labels

       def _return_confusionmatrix(self, predictions, labels, target_label):
              pad = (labels == 0)
              mask = ~pad
              tp = (predictions == target_label) & (labels == target_label) & mask
              fp = (predictions == target_label) & (labels != target_label) & mask
              fn = (predictions != target_label) & (labels == target_label) & mask
              tp = tp.sum()
              fp = fp.sum()
              fn = fn.sum()
              precision = tp / (tp + fp)
              recall = tp / (tp + fn)

              return precision, recall

       def configure_optimizers(self):
              optimizer = optim.Adam(self.parameters(),
                                          lr= self.learning_rate,
                                          eps=self.adam_epsilon,
                                          betas=(self.adam_beta1,self.adam_beta2),
                                          )
              return optimizer