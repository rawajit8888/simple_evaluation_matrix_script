import os
import pathlib
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import label_studio_sdk
import logging

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

import torch
import torch.optim as optim
import torch.nn as nn


from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler

from keras.utils import pad_sequences
from transformers import pipeline, Pipeline
from transformers import AdamW
from itertools import groupby
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer,AdamW
from transformers import DataCollatorForTokenClassification
from datasets import Dataset, ClassLabel, Value, Sequence, Features 
from functools import partial
from dataprocessing.processlabels import ProcessLabels
from multitask_nn_model import MultiTaskNNModel


def simple_chunk_text(text, max_chars=800):
    """
    Simple and safe character-based chunking.
    - No tokenizer dependency
    - No overlap
    - Deterministic
    """
    if not isinstance(text, str):
        return []

    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end

    return chunks


class MultiTaskBertModel:

    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', '')
    START_TRAINING_EACH_N_UPDATES = int(os.getenv('START_TRAINING_EACH_N_UPDATES', 10))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-3))
    NUM_TRAIN_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 3))
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', 0.01))
     

    def __init__(self,config,logger,label_interface=None,parsed_label_config=None):
        self.config = config
        self.logger = logger
        self.label_interface = label_interface
        self.parsed_label_config = parsed_label_config
        self.processed_label_encoders = ProcessLabels(self.parsed_label_config,pathlib.Path(self.config['MODEL_DIR']), training_mode=False) #5
        # self.reload_model()

    def reload_model(self):
        self.model = None
        classification_label_length = len(self.processed_label_encoders.labels["classification"]["encoder"].classes_) if  self.processed_label_encoders.labels["classification"]["encoder"] is not None else 1000
        print(classification_label_length)
        sentiment_label_length = len(self.processed_label_encoders.labels["sentiment"]["encoder"].classes_) if  self.processed_label_encoders.labels["sentiment"]["encoder"] is not None else 3

        try:
            self.chk_path = str(pathlib.Path(self.config['MODEL_DIR'])/self.config['FINETUNED_MODEL_NAME'])
            self.finedtunnedmodelpath = f'.\\{self.chk_path}'
            


            self.logger.info(f"Loading finetuned model from {self.chk_path}")
            self.model = MultiTaskNNModel(self.finedtunnedmodelpath,classification_label_length,sentiment_label_length)
            self.model.LoadModel(self.finedtunnedmodelpath)
            # self.model.load_state_dict(torch.load(self.finedtunnedmodelpath))
            # self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.finedtunnedmodelpath)
            
        except Exception as e:
            #self.chk_path = self.config['BASELINE_MODEL_NAME']
            self.chk_path = str(pathlib.Path(self.config['MODEL_DIR'])/self.config['BASELINE_MODEL_NAME']) #6 
            self.finedtunnedmodelpath = f'.\\{self.chk_path}'
            # if finetuned model is not available, use the baseline model with the original labels
            self.logger.info(f"Error {e}")
            self.logger.info(f"Loading baseline model {self.chk_path}")
            
            self.model = MultiTaskNNModel(self.chk_path,classification_label_length,sentiment_label_length)
            self.tokenizer = AutoTokenizer.from_pretrained(self.chk_path)
            self.chk_path = str(pathlib.Path(self.config['MODEL_DIR'])/self.config['FINETUNED_MODEL_NAME']) #7
            self.finedtunnedmodelpath = f'.\\{self.chk_path}'
            
    def fit(self, event, data,tasks, **kwargs):
        """Download dataset from Label Studio and prepare data for training in BERT
        """
        self.logger.info("start training entered")
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            self.logger.info(f"Skip training: event {event} is not supported")
            return

        # project_id = data['project']['id']
        # tasks = self._get_tasks(project_id)

        ds_raw = []
        def getClassificationAttrName(attrs):
            return attrs == 'classification'
        
        def getSentimentAttrName(attrs):
            return attrs == 'sentiment'

        from_name_classification, to_name_classification, value_classification = self.label_interface.get_first_tag_occurence('Taxonomy', 'HyperText',getClassificationAttrName)
        from_name_sentiment, to_name_sentiment, value_sentiment = self.label_interface.get_first_tag_occurence('Taxonomy', 'HyperText',getSentimentAttrName)
        self.logger.info("before tokenizer")
        tokenizer = self.tokenizer

        for task in tasks:
            for annotation in task['annotations']:
                if not annotation.get('result'):
                    continue

                sentiment_label = [r['value']['taxonomy'][0][0] for r in annotation['result'] if r["from_name"]=="sentiment"][0]
                classification_label = " > ".join([r['value']['taxonomy'][0] for r in annotation['result'] if r["from_name"]=="classification"][0])
                self.logger.info("after classification label")
                match = re.search(r"pre[^>]*>\s*(.*?)\s*</pre>", value_classification, re.DOTALL)
                value_classification = match.group(1)[1:] if match else value_classification

                text = self.preload_task_data(task, task['data'][value_classification])
                self.logger.info(classification_label)
                sentiment = self.processed_label_encoders['sentiment'].transform([sentiment_label])[0]
                classification = self.processed_label_encoders['classification'].transform([classification_label])[0]
                

                chunks = simple_chunk_text(text, max_chars=800)

                for chunk in chunks:
                    ds_raw.append([
                        chunk,
                        classification_label,
                        sentiment_label,
                        classification,
                        sentiment
                    ])


        self.logger.debug(f"Dataset: {ds_raw}")

        df = pd.DataFrame(np.array(ds_raw), columns =["text","classification_label","sentiment_label","classification","sentiment"] )
        # df.to_csv('output.tsv', sep='\t', index=False)

        MAX_LEN = 256 # Define the maximum length of tokenized texts
        batch_size = 16
        tokenizer = self.tokenizer
        tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in df['text']]

        # Create attention masks
        train_masks = [[int(token_id > 0) for token_id in input_id] for input_id in tokenized_texts]
        
        train_inputs = pad_sequences(tokenized_texts, maxlen=MAX_LEN, dtype='long', value=0, truncating='post', padding='post')
        train_masks = pad_sequences(train_masks, maxlen=MAX_LEN, dtype='long', value=0, truncating='post', padding='post')
        df_classification = df["classification"].astype(np.int64)
        df_sentiments = df['sentiment'].astype(np.int64)
        train_data = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_masks),
                                torch.tensor(df_classification.values), torch.tensor(df_sentiments.values))
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        model = self.model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        model.to(device)

        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=2e-5)

        
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        criterion = nn.CrossEntropyLoss()
        with tqdm(total=self.NUM_TRAIN_EPOCHS, desc="Epochs") as pbar:
            for epoch in range(self.NUM_TRAIN_EPOCHS):
            
                    for step, batch in enumerate(train_dataloader):
                        model.train()
                        input_ids = batch[0].to(device)
                        attention_mask = batch[1].to(device)
                        classification_labels = batch[2].to(device)
                        sentiment_labels = batch[3].to(device)
                        

                        optimizer.zero_grad()

                        classification_logits, sentiment_logits, classification_probs , sentiment_probs = model(input_ids, attention_mask)

                        classification_loss = criterion(classification_logits, classification_labels)
                        sentiment_loss = criterion(sentiment_logits, sentiment_labels)

                        loss = classification_loss +  sentiment_loss

                        loss.backward()
                        optimizer.step()

                        # self.logger.info(f"MultiTask Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")       
                        pbar.set_postfix(loss=loss.item())  # Update the progress bar with current loss
                        pbar.update(1)  # Increment the progress bar

        
        directory =  os.path.dirname(self.finedtunnedmodelpath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.logger.info(f"Model is saving at:{self.finedtunnedmodelpath}")
        # torch.save(model.state_dict(), self.finedtunnedmodelpath )
        model.SaveModel(self.finedtunnedmodelpath)
        

        # Save the tokenizer
        tokenizer.save_pretrained(self.finedtunnedmodelpath)
        self.logger.info(f"Model is saved at:{self.finedtunnedmodelpath}")

        # self.reload_model()

    def fit_external(self, event, data,tasks, **kwargs):
        """Download dataset from Label Studio and prepare data for training in BERT
        """
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            self.logger.info(f"Skip training: event {event} is not supported")
            return

        ds_raw = tasks

        df = pd.DataFrame.from_dict(ds_raw)

        chunked_rows = []

        for _, row in df.iterrows():
            chunks = simple_chunk_text(row["text"], max_chars=800)
            for chunk in chunks:
                new_row = row.copy()
                new_row["text"] = chunk
                chunked_rows.append(new_row)

        df = pd.DataFrame(chunked_rows)


        df = df.rename(columns={"sentiment":"sentiment_label"})

        df["sentiment"] = self.processed_label_encoders['sentiment'].transform(df["sentiment_label"])
        df["classification"] = self.processed_label_encoders['classification'].transform(df["label"])

        MAX_LEN = 256 # Define the maximum length of tokenized texts
        batch_size = 16
        tokenizer = self.tokenizer
        tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in df['text']]

        # Create attention masks
        train_masks = [[int(token_id > 0) for token_id in input_id] for input_id in tokenized_texts]
        
        train_inputs = pad_sequences(tokenized_texts, maxlen=MAX_LEN, dtype='long', value=0, truncating='post', padding='post')
        train_masks = pad_sequences(train_masks, maxlen=MAX_LEN, dtype='long', value=0, truncating='post', padding='post')
        df_classification = df["classification"].astype(np.int64)
        df_sentiments = df['sentiment'].astype(np.int64)
        train_data = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_masks),
                                torch.tensor(df_classification.values), torch.tensor(df_sentiments.values))
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        model = self.model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        model.to(device)

        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=2e-5)
    
            
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        criterion = nn.CrossEntropyLoss()
        with tqdm(total=self.NUM_TRAIN_EPOCHS, desc="Epochs") as pbar:
            for epoch in range(self.NUM_TRAIN_EPOCHS):
            
                    for step, batch in enumerate(train_dataloader):
                        model.train()
                        input_ids = batch[0].to(device)
                        attention_mask = batch[1].to(device)
                        classification_labels = batch[2].to(device)
                        sentiment_labels = batch[3].to(device)
                        

                        optimizer.zero_grad()

                        classification_logits, sentiment_logits, classification_probs , sentiment_probs = model(input_ids, attention_mask)

                        classification_loss = criterion(classification_logits, classification_labels)
                        sentiment_loss = criterion(sentiment_logits, sentiment_labels)

                        loss = classification_loss +  sentiment_loss

                        loss.backward()
                        optimizer.step()

                        # self.logger.info(f"MultiTask Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")       
                        pbar.set_postfix(loss=loss.item())  # Update the progress bar with current loss
                        pbar.update(1)  # Increment the progress bar


        directory =  os.path.dirname(self.finedtunnedmodelpath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # torch.save(model.state_dict(), self.finedtunnedmodelpath )
        model.SaveModel(self.finedtunnedmodelpath)
        

        # Save the tokenizer
        tokenizer.save_pretrained(self.finedtunnedmodelpath)

        # self.reload_model()


    def predict(self, tasks: List[Dict],texts:str, context: Optional[Dict] = None, **kwargs):
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        def getClassificationAttrName(attrs):
            return attrs == 'classification'
        
        def getSentimentAttrName(attrs):
            return attrs == 'sentiment'

        li = self.label_interface
        from_name_classification, to_name_classification, value_classification = self.label_interface.get_first_tag_occurence('Taxonomy', 'HyperText',getClassificationAttrName)
        from_name_sentiment, to_name_sentiment, value_sentiment = self.label_interface.get_first_tag_occurence('Taxonomy', 'HyperText',getSentimentAttrName)
        
        # texts = [self.preload_task_data(task, task['data'][value]) for task in tasks]

        tokenizer = self.tokenizer
        tokenized_texts = [tokenizer.encode(text['text'] if not isinstance(text, str) else text, add_special_tokens=True) for text in texts]

        # Create attention masks
        attention_mask = [[int(token_id > 0) for token_id in input_id] for input_id in tokenized_texts]
        MAX_LEN = 256
        batch_size = 16
        _inputs = pad_sequences(tokenized_texts, maxlen=MAX_LEN, dtype='long', value=0, truncating='post', padding='post')
        _masks = pad_sequences(attention_mask, maxlen=MAX_LEN, dtype='long', value=0, truncating='post', padding='post')

        train_data = TensorDataset(torch.tensor(_inputs), torch.tensor(_masks))
        train_sampler = SequentialSampler(train_data)
        dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                
                input_ids, attention_mask = tuple(t.to(device) for t in batch)

                classification_logits,  sentiment_logits, classification_probs1 ,  sentiment_probs1= self.model(input_ids, attention_mask)

                classification_probs = nn.Softmax(dim=1)(classification_logits)
                sentiment_probs = nn.Softmax(dim=1)(sentiment_logits)

                # Get the predicted labels (indices of the max probabilities)
                classification_preds = torch.argmax(classification_probs, dim=1)
                sentiment_preds = torch.argmax(sentiment_probs, dim=1)

                decoded_classification_preds = self.processed_label_encoders['classification'].inverse_transform(classification_preds.cpu().numpy())
                decoded_sentiment_preds = self.processed_label_encoders['sentiment'].inverse_transform(sentiment_preds.cpu().numpy())


                for i in range(len(classification_probs)):
                    # predictions.append({
                    #     'text': tokenizer.decode(input_ids[i]),
                    #     'classification_score': classification_probs[i].tolist(),
                    #     'classification_pred': decoded_classification_preds[i],
                    #     'sentiment_score': sentiment_probs[i].tolist(),
                    #     'sentiment_pred': decoded_sentiment_preds[i]
                    # })

                    predictions.append({
                        'from_name': from_name_classification,
                        'to_name':to_name_classification,
                        'type':'taxonomy',
                        'value':{
                                'taxonomy': [decoded_classification_preds[i].split(' > ')],
                                'score': classification_probs.gather(1, classification_preds.unsqueeze(1)).squeeze(1)[0].item()
                            },
                        
                    })
                    predictions.append({
                        'from_name': from_name_sentiment,
                        'to_name':to_name_sentiment,
                        'type':'taxonomy',
                        'value':{
                                'taxonomy': [[decoded_sentiment_preds[i]]],
                                'score': sentiment_probs.gather(1, sentiment_preds.unsqueeze(1)).squeeze(1)[0].item()
                            },
                        # 'score': sentiment_probs[i].tolist(),
                    })
        return predictions
