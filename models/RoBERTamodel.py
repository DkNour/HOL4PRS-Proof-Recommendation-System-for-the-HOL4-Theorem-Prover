from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Optional
import json

class BERTDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_token_len: int = 120,
        label_column: str = "label",
        text_column: str = "text",
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_column = text_column
        self.max_token_len = max_token_len
        self.label_column = label_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        text = data_row[self.text_column]
        label = data_row[self.label_column]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.tensor(label, dtype=torch.int32),
        )
    
class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        test_path: str,
        val_path: str,
        bert_model: str,
        text_column: str = "text",
        data_dir: str = "data/",
        label_column: str = "label",
        train_batch_size: int = 32,
        max_len: int = 120,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path

        self.train_data = pd.read_csv(
            os.path.join(data_dir, train_path), encoding="latin1"
        )
        self.test_data = pd.read_csv(
            os.path.join(data_dir, test_path), encoding="latin1"
        )
        self.val_data = pd.read_csv(os.path.join(data_dir, val_path), encoding="latin1")

        self.train_batch_size = train_batch_size

        self.bert_model = bert_model
        self.max_len = max_len
        self.label_column = label_column
        self.text_column = text_column

    def prepare_data(self):
        self.labelencoder = LabelEncoder()
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_data["label"] = self.labelencoder.fit_transform(
                self.train_data[self.label_column].values
            )

            self.train_dataset = BERTDataset(
                data=self.train_data,
                tokenizer=self.tokenizer,
                max_token_len=self.max_len,
                text_column=self.text_column,
                label_column="label",
            )

            self.val_data["label"] = self.labelencoder.transform(
                self.val_data[self.label_column].values
            )
            self.val_dataset = BERTDataset(
                data=self.val_data,
                tokenizer=self.tokenizer,
                max_token_len=self.max_len,
                text_column=self.text_column,
                label_column="label",
            )

            encodings = dict(
                zip(self.labelencoder.classes_, range(len(self.labelencoder.classes_)))
            )

            with open("labelencoder.json", "w") as le:
                le.write(json.dumps(encodings))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_data["label"] = self.labelencoder.transform(
                self.test_data[self.label_column].values
            )
            self.test_dataset = BERTDataset(
                data=self.test_data,
                tokenizer=self.tokenizer,
                max_token_len=self.max_len,
                text_column=self.text_column,
                label_column="label",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)   
    
class BertTextClassifier(pl.LightningModule):
    def __init__(
        self,
        bert_model: str,
        n_classes: int,
        lr: float = 2e-5,
        label_column: str = "label",
        n_training_steps=None,
        outputdir: str = "outputs",
        
    ):
        """Bert Classifier Model

        Args:
            bert_model (str): huggingface bert model
            n_classes (int): number of output classes
            lr (float, optional): learning rate value. Defaults to 2e-5.
            label_column (str, optional): the name of the label column in the dataframe. Defaults to "label".
            n_training_steps ([type], optional): optimizer parameter. Defaults to None.
        """

        super().__init__()
        self.bert_model = bert_model
        self.label_column = label_column
        self.bert = AutoModel.from_pretrained(bert_model, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_classes = n_classes
        self.n_training_steps = n_training_steps
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.average_training_loss = None
        self.average_validation_loss = None
        self.outputdir = outputdir
        #self.f1 = MultilabelF1Score(num_labels=n_classes, average="macro")
        #self.accuracy = MulticlassAccuracy(num_classes=n_classes, average="macro")

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels.long())
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        outputs = torch.argmax(outputs, dim=1)
        #accuracy = self.accuracy(outputs, labels)
        #f1 = self.f1(outputs, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
        #self.log("train_accuracy",accuracy,prog_bar=True,logger=True,batch_size=len(batch),)
        #self.log("train_f1", f1, prog_bar=True, logger=True, batch_size=len(batch))
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        outputs = torch.argmax(outputs, dim=1)
        #accuracy = self.accuracy(outputs, labels)
        #f1 = self.f1(outputs, labels)

        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
        #self.log("val_accuracy", accuracy, prog_bar=True, logger=True, batch_size=len(batch))
        #self.log("val_f1", f1, prog_bar=True, logger=True, batch_size=len(batch))

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        outputs = torch.argmax(outputs, dim=1)
        #accuracy = self.accuracy(outputs, labels)
        #f1 = self.f1(outputs, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
        #self.log("test_accuracy", accuracy, prog_bar=True, logger=True, batch_size=len(batch))
        #self.log("test_f1", f1, prog_bar=True, logger=True, batch_size=len(batch))
        return loss

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
      return [optimizer]
    
'''    def training_epoch_end(self, training_step_outputs):
        """ save tokenizer and model on epoch end """
        self.average_training_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )
        path = f"{self.outputdir}/simplet5-epoch-{self.current_epoch}-train-loss-{str(self.average_training_loss)}-val-loss-{str(self.average_validation_loss)}"
        self.tokenizer.save_pretrained(path)
        self.classifier.save(path)

    def validation_epoch_end(self, validation_step_outputs):
        _loss = [x.cpu() for x in validation_step_outputs]
        self.average_validation_loss = np.round(
            torch.mean(torch.stack(_loss)).item(),
            4,
        )'''

class BERTmodel:

    def __init__(self) -> None:
        print("BERTmode created")

    def from_pretrained(self, model_name="roberta-base", tokenizer=None) -> None:
        if tokenizer is not None:
            self.tokenizer = tokenizer  
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
            self.model = AutoModel.from_pretrained(
                f"{model_name}", return_dict=True
            )


    def train(
        self,
        data_dir = "/content/",
        model_name = "roberta-base",
        max_length = 120,
        train_path = "train.csv",
        test_path = "test.csv",
        val_path = "test.csv",
        batch_size = 32,
        lr = 2e-5,
        num_classes = 84,
        n_training_steps = 320,
        deterministic = True,
        max_epochs = 1,
        num_gpus = 1,
        outputdir: str = "outputs",
        ):

        self.text_datamodule = TextDataModule( data_dir=data_dir,
                                        bert_model=model_name,
                                        text_column="context",
                                        label_column="class",
                                        max_len=max_length,
                                        train_path=train_path,
                                        test_path=test_path,
                                        val_path=val_path,
                                        train_batch_size=batch_size,
                                        )

        self.model = BertTextClassifier(
            bert_model=model_name,
            label_column="class",
            lr=lr,
            n_classes=num_classes,
            n_training_steps=n_training_steps,
        )

        checkpoint_callback = ModelCheckpoint( dirpath=outputdir, filename="best-checkpoint", save_top_k=1, verbose=True, monitor="val_loss", mode="min" )

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            deterministic=deterministic,
            max_epochs=max_epochs,
            gpus=num_gpus,
            default_root_dir=outputdir,
            )

        trainer.fit(model=self.model, datamodule=self.text_datamodule)

    def predict(text, model, tokenizer, max_length=120, top_k=7):

        encoding = tokenizer.encode_plus(
            text,
            max_length=max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        encoding["input_ids"], encoding["attention_mask"] = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
        _, test_prediction = model(encoding["input_ids"], encoding["attention_mask"])
        top_k_values, top_k_indices = torch.topk(test_prediction, k=top_k, dim=-1)
        with open("labelencoder.json", 'r') as file:
            data = json.load(file)
        result = {}
        preds = top_k_indices.tolist()[0]
        for key, value in data.items():
            if value in preds:
                result[value] = key

        preds = list(result.values())

        return preds