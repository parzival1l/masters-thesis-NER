import os
import sys
import pandas as pd
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from .charEncoderModel import charEncoderModel
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner
import mlflow
import torch
import util
import yaml
log = util.get_pylogger(__name__)

def trainNERmodel(data_loaders, exp_config, tokenizer)-> None:
    # Model
    log.info("Instantiating NERModel...")
    charModel = charEncoderModel(
        tokenizer = tokenizer,
        exp_config= exp_config
    )

    # Callbacks
    mlflow.pytorch.autolog()
    log.info("Instantiating callbacks...")
    early_stop_callback = EarlyStopping(monitor="train_loss",
                                        min_delta=0.000,
                                        patience=2,
                                        verbose=True,
                                        mode="min",
                                        )

    lr_logger = LearningRateMonitor(logging_interval='step',log_momentum=True)

    checkpoint_callback = ModelCheckpoint(filename=None,
                                          dirpath=os.path.join(exp_config['model']['save_path'],'checkpoints'),
                                          monitor='train_loss',
                                          verbose=True,
                                          save_top_k=-1,
                                          save_weights_only=False,
                                          mode='min',
                                          auto_insert_metric_name=True,
                                          save_on_train_epoch_end=True,
                                          every_n_epochs=1,
                                          every_n_train_steps=None,
                                          train_time_interval=None,
                                          save_last=None
                                          )

    richProgressBar_callback = RichProgressBar()
    # training
    log.info("Instantiating trainer...")
    trainer = Trainer(devices= exp_config['trainer']['devices'],
                      accelerator=exp_config['trainer']['accelerator'],
                      max_epochs=exp_config['trainer']['max_epochs'],
                      callbacks=[early_stop_callback,lr_logger,checkpoint_callback,richProgressBar_callback],
                    #   accumulate_grad_batches=4
                      )

    # Auto-scale batch size by growing it exponentially (default)
    log.info("Train the model...")
    trainer.fit(model=charModel,
                train_dataloaders= data_loaders["train_dataloader"], val_dataloaders= data_loaders["val_dataloader"])

    mlflow.log_params({
        "max_epochs": exp_config['trainer']['max_epochs'],
        "max_data": 1000,
        "batch_size": exp_config['dataloader']['batch_size']
    })

    state_dict = charModel.encoderModel.state_dict()
    _path=os.path.join(exp_config['model']['save_path'], 'state_dict.pth')
    torch.save(state_dict,_path)

def testNERmodel(test_dataloader, exp_config, tokenizer, index2label)-> None:
    # Model
    log.info("Instantiating NERModel...")
    charModel = charEncoderModel(
        tokenizer = tokenizer,
        exp_config= exp_config
    )

    # Callbacks
    mlflow.pytorch.autolog()
    modelLoadPath = exp_config['model']['load_path']
    if os.path.exists(modelLoadPath):
        log.info("Loading model from checkpoint...")
        if exp_config['model']['load_path'].endswith(".pth") :
            charModel.encoderModel.load_state_dict(torch.load(modelLoadPath))
        elif exp_config['model']['load_path'].endswith(".ckpt") :
            charModel.load_state_dict(torch.load(modelLoadPath)['state_dict'])
        else :
            modelLoadPath = exp_config['model']['load_path'] + "/state_dict.pth"
            charModel.encoderModel.load_state_dict(torch.load(modelLoadPath))


    log.info("Instantiating trainer...")
    trainer = Trainer(devices= exp_config['trainer']['devices'],
                      check_val_every_n_epoch = 1,
                      accelerator=exp_config['trainer']['accelerator'],
                      max_epochs=exp_config['trainer']['max_epochs'],
                    #   accumulate_grad_batches=4
                      )
    log.info("Test the model...")
    results = trainer.predict(model=charModel, dataloaders=test_dataloader, return_predictions=True)

    predictions = torch.tensor(results[0][0])
    labels  = torch.tensor(results[0][1])
    for index in range(1,len(results)) :
        predictions = torch.cat((predictions, results[index][0]), dim =0)
        labels = torch.cat((labels, results[index][1]), dim =0)
    df_results_new = return_predictions(predictions, labels, tokenizer, index2label, test_dataloader)
    df_results_new.to_csv(exp_config['model']['save_path'] + "/results.csv")
    pad = (labels == exp_config['tokenizer']['special_token_label'])
    mask = ~pad
    device = torch.device("cpu")
    predictions = predictions.to(device, dtype = torch.long)
    matches = predictions[mask] == labels[mask]
    accuracy = torch.mean(matches.float())
    print('val_acc :', round(accuracy.item() , 3))
    mlflow.log_metric("val_acc", round(accuracy.item() , 3))
    for key, value in index2label.items():
           if value not in ["O", "--PADDING--"] :
                  precision, recall = _return_confusionmatrix(predictions, labels, key)
                  print(f'precision_{value}', round(precision.item() , 3))
                  mlflow.log_metric(f'precision_{value}', round(precision.item() , 3))
                  print(f'recall_{value}', round(recall.item() , 3))
                  mlflow.log_metric(f'recall_{value}', round(recall.item() , 3))

    mlflow.log_artifact(exp_config['model']['save_path'] + "results.csv", artifact_path = mlflow.get_artifact_uri())

def _return_confusionmatrix(predictions, labels, target_label):
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

def return_predictions(predictions, labels, tokenizer, index2label, test_dataloader) :
    allTokens = []
    allTags = []
    allTrueLabels = []
    for index in range(0, predictions.shape[0]) :
        tokens = tokenizer.convert_ids_to_tokens(test_dataloader.dataset['input_ids'][index].squeeze().tolist())
        predicted_labels = [index2label[i] for i in predictions[index].cpu().numpy()]
        true_labels  = [index2label[i] for i in labels[index].cpu().numpy()]
        combined_tokens, combined_true_labels = combine_word_pieces(tokens, true_labels)
        combined_tokens, combined_ner_tags = combine_word_pieces(tokens, predicted_labels)
        allTokens.append(combined_tokens)
        allTags.append(combined_ner_tags)
        allTrueLabels.append(combined_true_labels)

    df_results = pd.DataFrame({'Tokens': allTokens, 'Tags': allTags, 'True_Tags': allTrueLabels})
    df_results = df_results.reset_index()
    df_results_new = pd.DataFrame({
        'index': df_results['index'].repeat(df_results['Tokens'].apply(len)),
        'Token': [item for sublist in df_results['Tokens'] for item in sublist],
        'Tags': [item for sublist in df_results['Tags'] for item in sublist],
        'True_Tags': [item for sublist in df_results['True_Tags'] for item in sublist]
            })

    return df_results_new
def combine_word_pieces(tokens, token_predictions):

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

# from argparse import ArgumentParser
# def main(hparams):
#     model = LightningModule()
#     trainer = Trainer(accelerator=hparams.accelerator, devices=hparams.devices)
#     trainer.fit(model)


# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--accelerator", default=None)
#     parser.add_argument("--devices", default=None)
#     args = parser.parse_args()

#     main(args)