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
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer, AdamW
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
    """
    3-Level Hierarchical BERT Email Classification Model
    
    Training Strategy:
    - Level 1: Email → Master Department (Internet Banking, Credit Cards, etc.)
    - Level 2: Email → Sub-Category (Internet Banking > Account Access, etc.)
    - Level 3: Email → Specific Issue (Internet Banking > Account Access > Unblock, etc.)
    - Sentiment: Parallel sentiment classification
    
    All levels are trained simultaneously with multi-task loss.
    """

    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', '')
    START_TRAINING_EACH_N_UPDATES = int(os.getenv('START_TRAINING_EACH_N_UPDATES', 10))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-5))
    NUM_TRAIN_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 4))
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', 0.01))

    def __init__(self, config, logger, label_interface=None, parsed_label_config=None):
        self.config = config
        self.logger = logger
        self.label_interface = label_interface
        self.parsed_label_config = parsed_label_config
        self.processed_label_encoders = ProcessLabels(
            self.parsed_label_config,
            pathlib.Path(self.config['MODEL_DIR'])
        )

    def reload_model(self):
        """Load or initialize 3-level hierarchical model"""
        self.model = None

        # Get encoder sizes
        l1 = len(self.processed_label_encoders.labels["lvl1"]["encoder"].classes_) \
            if self.processed_label_encoders.labels["lvl1"]["encoder"] is not None else 50

        l2 = len(self.processed_label_encoders.labels["lvl2"]["encoder"].classes_) \
            if self.processed_label_encoders.labels["lvl2"]["encoder"] is not None else 100

        l3 = len(self.processed_label_encoders.labels["lvl3"]["encoder"].classes_) \
            if self.processed_label_encoders.labels["lvl3"]["encoder"] is not None else 150

        sentiment_label_length = len(self.processed_label_encoders.labels["sentiment"]["encoder"].classes_) \
            if self.processed_label_encoders.labels["sentiment"]["encoder"] is not None else 3

        try:
            self.chk_path = str(pathlib.Path(self.config['MODEL_DIR']) / self.config['FINETUNED_MODEL_NAME'])
            self.finedtunnedmodelpath = f'.\\{self.chk_path}'

            self.logger.info(f"📂 Loading finetuned 3-level model from {self.chk_path}")

            self.model = MultiTaskNNModel(
                self.finedtunnedmodelpath,
                l1, l2, l3,
                sentiment_label_length
            )

            self.model.LoadModel(self.finedtunnedmodelpath)
            self.tokenizer = AutoTokenizer.from_pretrained(self.finedtunnedmodelpath)
            self.logger.info("✅ 3-Level model loaded successfully")

        except Exception as e:
            self.chk_path = str(pathlib.Path(self.config['MODEL_DIR']) / self.config['BASELINE_MODEL_NAME'])
            self.finedtunnedmodelpath = f'.\\{self.chk_path}'

            self.logger.info(f"⚠️  Error: {e}")
            self.logger.info(f"📂 Loading baseline model from {self.chk_path}")

            self.model = MultiTaskNNModel(
                self.chk_path,
                l1, l2, l3,
                sentiment_label_length
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.chk_path)
            self.chk_path = str(pathlib.Path(self.config['MODEL_DIR']) / self.config['FINETUNED_MODEL_NAME'])
            self.finedtunnedmodelpath = f'.\\{self.chk_path}'

    def fit(self, event, data, tasks, **kwargs):
        """
        Train 3-level hierarchical classification model.
        
        Training data format:
        - Each email has taxonomy path: ["Internet Banking", "Account Access", "Unblock"]
        - We split this into:
          - lvl1: "Internet Banking"
          - lvl2: "Internet Banking > Account Access"
          - lvl3: "Internet Banking > Account Access > Unblock"
        """
        self.logger.info("🚀 Starting 3-level hierarchical training")

        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            self.logger.info(f"Skip training: event {event} is not supported")
            return

        ds_raw = []

        def getClassificationAttrName(attrs):
            return attrs == 'classification'

        def getSentimentAttrName(attrs):
            return attrs == 'sentiment'

        from_name_classification, to_name_classification, value_classification = \
            self.label_interface.get_first_tag_occurence('Taxonomy', 'HyperText', getClassificationAttrName)
        from_name_sentiment, to_name_sentiment, value_sentiment = \
            self.label_interface.get_first_tag_occurence('Taxonomy', 'HyperText', getSentimentAttrName)

        self.logger.info("📊 Processing annotation data...")
        tokenizer = self.tokenizer

        for task in tasks:
            for annotation in task['annotations']:
                if not annotation.get('result'):
                    continue

                # Get sentiment label
                sentiment_label = [r['value']['taxonomy'][0][0] for r in annotation['result']
                                   if r["from_name"] == "sentiment"][0]

                # Get full taxonomy path
                classification_path = [r['value']['taxonomy'][0] for r in annotation['result']
                                       if r["from_name"] == "classification"][0]

                # ===== CRITICAL: Split into 3 levels =====
                # Level 1: First element only
                l1_label = classification_path[0]

                # Level 2: First + Second joined with " > "
                l2_label = " > ".join(classification_path[:2])

                # Level 3: All three joined with " > "
                l3_label = " > ".join(classification_path[:3])

                self.logger.debug(f"Levels: L1='{l1_label}' | L2='{l2_label}' | L3='{l3_label}'")

                # Get email text
                match = re.search(r"pre[^>]*>\s*(.*?)\s*</pre>", value_classification, re.DOTALL)
                value_classification = match.group(1)[1:] if match else value_classification
                text = self.preload_task_data(task, task['data'][value_classification])

                # Encode labels to integer IDs
                sentiment = self.processed_label_encoders['sentiment'].transform([sentiment_label])[0]
                l1_id = self.processed_label_encoders['lvl1'].transform([l1_label])[0]
                l2_id = self.processed_label_encoders['lvl2'].transform([l2_label])[0]
                l3_id = self.processed_label_encoders['lvl3'].transform([l3_label])[0]

                # Chunk text if too long
                chunks = simple_chunk_text(text, max_chars=800)

                for chunk in chunks:
                    ds_raw.append([
                        chunk,
                        l1_label, l2_label, l3_label,  # String labels
                        sentiment_label,
                        l1_id, l2_id, l3_id,  # Integer IDs
                        sentiment
                    ])

        self.logger.debug(f"Dataset preview: {ds_raw[:2]}")

        df = pd.DataFrame(
            ds_raw,
            columns=["text", "l1_label", "l2_label", "l3_label", "sentiment_label",
                     "l1_id", "l2_id", "l3_id", "sentiment"]
        )

        self.logger.info(f"✅ Processed {len(df)} training samples")

        MAX_LEN = 256
        batch_size = 16
        tokenizer = self.tokenizer

        # Tokenize texts
        tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in df['text']]

        # Create attention masks
        train_masks = [[int(token_id > 0) for token_id in input_id] for input_id in tokenized_texts]

        train_inputs = pad_sequences(tokenized_texts, maxlen=MAX_LEN, dtype='long', value=0,
                                     truncating='post', padding='post')
        train_masks = pad_sequences(train_masks, maxlen=MAX_LEN, dtype='long', value=0,
                                    truncating='post', padding='post')

        # Convert labels to int64
        df_l1 = df["l1_id"].astype(np.int64)
        df_l2 = df["l2_id"].astype(np.int64)
        df_l3 = df["l3_id"].astype(np.int64)
        df_sentiments = df['sentiment'].astype(np.int64)

        # Create training dataset
        train_data = TensorDataset(
            torch.tensor(train_inputs),
            torch.tensor(train_masks),
            torch.tensor(df_l1.values),
            torch.tensor(df_l2.values),
            torch.tensor(df_l3.values),
            torch.tensor(df_sentiments.values)
        )

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        model = self.model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"🖥️  Using device: {device}")
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        criterion = nn.CrossEntropyLoss()

        self.logger.info("=" * 80)
        self.logger.info("🎯 TRAINING 3-LEVEL HIERARCHICAL MODEL")
        self.logger.info("=" * 80)

        with tqdm(total=self.NUM_TRAIN_EPOCHS, desc="Epochs") as pbar:
            for epoch in range(self.NUM_TRAIN_EPOCHS):
                epoch_total_loss = 0.0

                for step, batch in enumerate(train_dataloader):
                    model.train()

                    # Get inputs and labels
                    input_ids = batch[0].to(device)
                    attention_mask = batch[1].to(device)
                    l1_labels = batch[2].to(device)
                    l2_labels = batch[3].to(device)
                    l3_labels = batch[4].to(device)
                    sentiment_labels = batch[5].to(device)

                    optimizer.zero_grad()

                    # ===== FIXED: Get model outputs properly =====
                    outputs = model(input_ids, attention_mask)
                    
                    # Unpack outputs (8 values)
                    l1_logits = outputs[0]
                    l2_logits = outputs[1]
                    l3_logits = outputs[2]
                    sentiment_logits = outputs[3]
                    # outputs[4:8] are probabilities (we don't need them for training)

                    # Calculate hierarchical multi-task loss
                    l1_loss = criterion(l1_logits, l1_labels)
                    l2_loss = criterion(l2_logits, l2_labels)
                    l3_loss = criterion(l3_logits, l3_labels)
                    sentiment_loss = criterion(sentiment_logits, sentiment_labels)

                    # Total loss (sum of all levels)
                    loss = l1_loss + l2_loss + l3_loss + sentiment_loss

                    loss.backward()
                    optimizer.step()

                    epoch_total_loss += loss.item()

                    # Log every 10 batches
                    if step % 10 == 0:
                        self.logger.info(
                            f"Epoch {epoch + 1}/{self.NUM_TRAIN_EPOCHS}, "
                            f"Batch {step}/{len(train_dataloader)}, "
                            f"Loss: {loss.item():.4f} "
                            f"(L1: {l1_loss.item():.4f}, L2: {l2_loss.item():.4f}, "
                            f"L3: {l3_loss.item():.4f}, Sent: {sentiment_loss.item():.4f})"
                        )

                avg_epoch_loss = epoch_total_loss / len(train_dataloader)
                pbar.set_postfix(loss=avg_epoch_loss)
                pbar.update(1)

                self.logger.info(f"✅ Epoch {epoch + 1} complete. Avg Loss: {avg_epoch_loss:.4f}")

        self.logger.info("=" * 80)
        self.logger.info("🎉 TRAINING COMPLETED")
        self.logger.info("=" * 80)

        # Save model
        directory = os.path.dirname(self.finedtunnedmodelpath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.logger.info(f"💾 Saving model to: {self.finedtunnedmodelpath}")

        # Attach encoders before saving
        model.set_encoders(
            self.processed_label_encoders['lvl1'],
            self.processed_label_encoders['lvl2'],
            self.processed_label_encoders['lvl3'],
            self.processed_label_encoders['sentiment']
        )

        model.SaveModel(self.finedtunnedmodelpath)
        tokenizer.save_pretrained(self.finedtunnedmodelpath)

        self.logger.info("✅ Model saved successfully with bundled encoders!")

    def predict(self, tasks: List[Dict], texts: str, context: Optional[Dict] = None, **kwargs):
        """
        Prediction with 3-level hierarchical output
        
        Returns predictions for:
        - Level 1: Master Department
        - Level 2: Sub-Category  
        - Level 3: Specific Issue
        - Sentiment
        """
        self.logger.info(f"🔮 Running inference on {len(texts)} samples")

        def getClassificationAttrName(attrs):
            return attrs == 'classification'

        def getSentimentAttrName(attrs):
            return attrs == 'sentiment'

        from_name_classification, to_name_classification, value_classification = \
            self.label_interface.get_first_tag_occurence('Taxonomy', 'HyperText', getClassificationAttrName)
        from_name_sentiment, to_name_sentiment, value_sentiment = \
            self.label_interface.get_first_tag_occurence('Taxonomy', 'HyperText', getSentimentAttrName)

        tokenizer = self.tokenizer
        tokenized_texts = [tokenizer.encode(text['text'] if not isinstance(text, str) else text, add_special_tokens=True)
                           for text in texts]

        # Create attention masks
        attention_mask = [[int(token_id > 0) for token_id in input_id] for input_id in tokenized_texts]

        MAX_LEN = 256
        batch_size = 16

        _inputs = pad_sequences(tokenized_texts, maxlen=MAX_LEN, dtype='long', value=0,
                                truncating='post', padding='post')
        _masks = pad_sequences(attention_mask, maxlen=MAX_LEN, dtype='long', value=0,
                               truncating='post', padding='post')

        test_data = TensorDataset(torch.tensor(_inputs), torch.tensor(_masks))
        test_sampler = SequentialSampler(test_data)
        dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = tuple(t.to(device) for t in batch)

                # ===== FIXED: Get model outputs properly =====
                outputs = self.model(input_ids, attention_mask)
                
                # Unpack outputs
                l1_logits = outputs[0]
                l2_logits = outputs[1]
                l3_logits = outputs[2]
                sentiment_logits = outputs[3]
                # outputs[4:8] are already softmaxed probabilities

                # Apply softmax to logits
                l1_probs = nn.Softmax(dim=1)(l1_logits)
                l2_probs = nn.Softmax(dim=1)(l2_logits)
                l3_probs = nn.Softmax(dim=1)(l3_logits)
                sentiment_probs = nn.Softmax(dim=1)(sentiment_logits)

                # Get predictions
                l1_preds = torch.argmax(l1_probs, dim=1)
                l2_preds = torch.argmax(l2_probs, dim=1)
                l3_preds = torch.argmax(l3_probs, dim=1)
                sentiment_preds = torch.argmax(sentiment_probs, dim=1)

                # ===== FIXED: Use correct encoder names =====
                decoded_l1 = self.processed_label_encoders['lvl1'].inverse_transform(l1_preds.cpu().numpy())
                decoded_l2 = self.processed_label_encoders['lvl2'].inverse_transform(l2_preds.cpu().numpy())
                decoded_l3 = self.processed_label_encoders['lvl3'].inverse_transform(l3_preds.cpu().numpy())
                decoded_sentiment = self.processed_label_encoders['sentiment'].inverse_transform(sentiment_preds.cpu().numpy())

                for i in range(len(l1_probs)):
                    # ===== Classification: Return full 3-level path =====
                    # Split lvl3 back into array format for Label Studio
                    l3_path = decoded_l3[i].split(' > ')

                    predictions.append({
                        'from_name': from_name_classification,
                        'to_name': to_name_classification,
                        'type': 'taxonomy',
                        'value': {
                            'taxonomy': [l3_path],  # Full path as array
                            'score': l3_probs[i][l3_preds[i]].item()  # Use L3 confidence
                        },
                    })

                    # ===== Sentiment =====
                    predictions.append({
                        'from_name': from_name_sentiment,
                        'to_name': to_name_sentiment,
                        'type': 'taxonomy',
                        'value': {
                            'taxonomy': [[decoded_sentiment[i]]],
                            'score': sentiment_probs[i][sentiment_preds[i]].item()
                        },
                    })

        self.logger.info(f"✅ Generated {len(predictions)} predictions")
        return predictions
