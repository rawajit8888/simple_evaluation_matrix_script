import os
import pathlib
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from collections import Counter
from torch.utils.data import WeightedRandomSampler

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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


class MultiTaskBertModel:

    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8081')
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', '293da9e6a2f671bcbd79075b6aba64639904a023')
    START_TRAINING_EACH_N_UPDATES = int(os.getenv('START_TRAINING_EACH_N_UPDATES', 10))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-3))
    NUM_TRAIN_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 100))
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', 0.01))

    def save_metrics_to_sqlite(
        self,
        class_accuracy,
        class_f1_weighted,
        sent_accuracy,
        sent_f1_weighted,
        train_size,
        test_size
    ):
        import sqlite3
        from datetime import datetime

        base_dir = os.path.dirname(self.finedtunnedmodelpath)
        eval_dir = os.path.join(base_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        db_path = os.path.join(eval_dir, "evaluation.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS run_info (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            train_size INTEGER,
            test_size INTEGER
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            task TEXT,
            metric TEXT,
            value REAL
        )
        """)

        cur.execute("""
        INSERT INTO run_info (created_at, train_size, test_size)
        VALUES (?, ?, ?)
        """, (datetime.now().isoformat(), train_size, test_size))

        run_id = cur.lastrowid

        metrics = [
            ("classification", "accuracy", class_accuracy),
            ("classification", "f1_weighted", class_f1_weighted),
            ("sentiment", "accuracy", sent_accuracy),
            ("sentiment", "f1_weighted", sent_f1_weighted),
        ]

        cur.executemany("""
        INSERT INTO evaluation_metrics (run_id, task, metric, value)
        VALUES (?, ?, ?, ?)
        """, [(run_id, *m) for m in metrics])

        conn.commit()
        conn.close()

    def save_full_classification_report_to_sqlite(
        self,
        run_id,
        task_name,
        y_true,
        y_pred
    ):
        import sqlite3, re
        from sklearn.metrics import classification_report

        def clean(name):
            return re.sub(r'[^a-zA-Z0-9_]+', '_', name).lower()

        base_dir = os.path.dirname(self.finedtunnedmodelpath)
        db_path = os.path.join(base_dir, "evaluation", "evaluation.db")

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO evaluation_metrics (run_id, task, metric, value)
        VALUES (?, ?, ?, ?)
        """, (run_id, task_name, "accuracy_overall", float(report["accuracy"])))

        for avg in ["macro avg", "weighted avg"]:
            for m in ["precision", "recall", "f1-score"]:
                cur.execute("""
                INSERT INTO evaluation_metrics (run_id, task, metric, value)
                VALUES (?, ?, ?, ?)
                """, (run_id, task_name, clean(f"{avg}_{m}"), float(report[avg][m])))

        for label, values in report.items():
            if label in ["accuracy", "macro avg", "weighted avg"]:
                continue
            for m in ["precision", "recall", "f1-score", "support"]:
                cur.execute("""
                INSERT INTO evaluation_metrics (run_id, task, metric, value)
                VALUES (?, ?, ?, ?)
                """, (run_id, task_name, clean(f"{label}_{m}"), float(values.get(m, 0))))

        conn.commit()
        conn.close()

    def get_latest_metrics_from_db(self):
        import sqlite3

        base_dir = os.path.dirname(self.finedtunnedmodelpath)
        db_path = os.path.join(base_dir, "evaluation", "evaluation.db")

        if not os.path.exists(db_path):
            return {"error": "evaluation.db not found"}

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT * FROM run_info ORDER BY run_id DESC LIMIT 1")
        run = cur.fetchone()

        cur.execute("""
        SELECT task, metric, value
        FROM evaluation_metrics
        WHERE run_id = ?
        """, (run["run_id"],))

        metrics = [dict(r) for r in cur.fetchall()]
        conn.close()

        return {"run_info": dict(run), "metrics": metrics}
    

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
        self.model = None
        classification_label_length = len(
            self.processed_label_encoders.labels["classification"]["encoder"].classes_
        ) if self.processed_label_encoders.labels["classification"]["encoder"] is not None else 1000
        sentiment_label_length = len(
            self.processed_label_encoders.labels["sentiment"]["encoder"].classes_
        ) if self.processed_label_encoders.labels["sentiment"]["encoder"] is not None else 3

        try:
            self.chk_path = str(pathlib.Path(self.config['MODEL_DIR']) / self.config['FINETUNED_MODEL_NAME'])
            self.finedtunnedmodelpath = f'.\\{self.chk_path}'

            self.logger.info(f"Loading finetuned model from {self.chk_path}")
            self.model = MultiTaskNNModel(self.finedtunnedmodelpath, classification_label_length, sentiment_label_length)
            self.model.LoadModel(self.finedtunnedmodelpath)
            self.tokenizer = AutoTokenizer.from_pretrained(self.finedtunnedmodelpath)

        except Exception as e:
            self.logger.info(f"Error Loading finetunned model: {e}")
            self.chk_path = str(pathlib.Path(self.config['MODEL_DIR']) / self.config['BASELINE_MODEL_NAME'])
            self.finedtunnedmodelpath = f'.\\{self.chk_path}'

            self.logger.info(f"Loading baseline model {self.chk_path}")
            self.model = MultiTaskNNModel(self.chk_path, classification_label_length, sentiment_label_length)
            self.tokenizer = AutoTokenizer.from_pretrained(self.chk_path)
            self.finedtunnedmodelpath = str(pathlib.Path(self.config['MODEL_DIR']) / self.config['FINETUNED_MODEL_NAME'])

    def fit(self, event, data, tasks, **kwargs):
        """Download dataset from Label Studio and prepare data for training in BERT"""
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

        tokenizer = self.tokenizer

        for task in tasks:
            # SubQueryType comes from Label Studio data
            subquery_type = task['data'].get('SubQueryType', 'Unknown')
            for annotation in task['annotations']:
                if not annotation.get('result'):
                    continue

                sentiment_label = [r['value']['taxonomy'][0][0] for r in annotation['result']
                                   if r["from_name"] == "sentiment"][0]
                classification_label = " > ".join(
                    [r['value']['taxonomy'][0] for r in annotation['result'] if r["from_name"] == "classification"][0]
                )

                match = re.search(r"pre[^>]*>\s*(.*?)\s*</pre>", value_classification, re.DOTALL)
                value_classification_clean = match.group(1)[1:] if match else value_classification

                text = self.preload_task_data(task, task['data'][value_classification_clean])

                sentiment = self.processed_label_encoders['sentiment'].transform([sentiment_label])[0]
                classification = self.processed_label_encoders['classification'].transform([classification_label])[0]

                # include SubQueryType
                ds_raw.append([
                    text,
                    classification_label,
                    sentiment_label,
                    classification,
                    sentiment,
                    subquery_type
                ])

        self.logger.debug(f"Dataset: {ds_raw}")

        # Build dataframe
        df = pd.DataFrame(
            ds_raw,
            columns=["text", "classification_label", "sentiment_label",
                     "classification", "sentiment", "SubQueryType"]
        )

        # ---- Stratified 80/20 train/test split by SubQueryType ----
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df["SubQueryType"] = df["SubQueryType"].fillna("Unknown")

        train_indices = []
        test_indices = []

        rng = np.random.default_rng(42)

        for subq, group in df.groupby("SubQueryType"):
            idx = group.index.to_list()
            rng.shuffle(idx)
            n = len(idx)

            if n < 2:
                # If only 1 sample of this SubQueryType, keep it in train
                train_indices.extend(idx)
                continue

            n_train = int(np.floor(0.8 * n))
            if n_train == 0:
                n_train = 1

            train_indices.extend(idx[:n_train])
            test_indices.extend(idx[n_train:])

        # Fallback in case no test samples got created
        if len(test_indices) == 0 and len(df) > 1:
            all_idx = list(df.index)
            rng.shuffle(all_idx)
            split_idx = int(len(all_idx) * 0.8)
            train_indices = all_idx[:split_idx]
            test_indices = all_idx[split_idx:]

        train_df = df.loc[train_indices].reset_index(drop=True)
        test_df = df.loc[test_indices].reset_index(drop=True)

        self.logger.info(
            f"Total samples: {len(df)}, Train: {len(train_df)}, Test: {len(test_df)} (stratified by SubQueryType)"
        )

        # ---------- SAVE RAW 20% TEST SPLIT BEFORE TRAINING ----------
        # This saves the raw test data (text + labels + SubQueryType) before fine-tuning starts
        directory = os.path.dirname(self.finedtunnedmodelpath)
        eval_dir = os.path.join(directory, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        raw_test_path = os.path.join(eval_dir, "test_raw_before_training.csv")
        # keep only human-readable columns for raw view
        raw_cols = ["text", "SubQueryType", "classification_label", "sentiment_label"]
        test_df[raw_cols].to_csv(raw_test_path, index=False)
        self.logger.info(f"Saved raw 20% test split (before training) to {raw_test_path}")

        MAX_LEN = 256  # Define the maximum length of tokenized texts
        batch_size = 16
        tokenizer = self.tokenizer

        # ---------- TRAIN DATA (80%) ----------
        tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in train_df['text']]

        # Create attention masks
        train_masks = [[int(token_id > 0) for token_id in input_id] for input_id in tokenized_texts]

        train_inputs = pad_sequences(
            tokenized_texts, maxlen=MAX_LEN, dtype='long', value=0,
            truncating='post', padding='post'
        )
        train_masks = pad_sequences(
            train_masks, maxlen=MAX_LEN, dtype='long', value=0,
            truncating='post', padding='post'
        )
        df_classification = train_df["classification"].astype(np.int64)
        df_sentiments = train_df['sentiment'].astype(np.int64)


        # -------- WEIGHTED SAMPLER SETUP (CLASSIFICATION) --------
        class_labels = df_classification.values.tolist()
        class_counts = Counter(class_labels)

        self.logger.info(f"Classification class distribution: {class_counts}")


        MAX_WEIGHT = 5.0     # prevents 1–2 sample classes from dominating
        MIN_COUNT = 3        # classes with <3 samples are heavily downweighted

        sample_weights = []

        for label in class_labels:
            count = class_counts[label]

            if count < MIN_COUNT:
                weight = 0.1     # almost ignored (you already manually increased <10)
            else:
                weight = min(1.0 / count, MAX_WEIGHT)

            sample_weights.append(weight)


        train_data = TensorDataset(
            torch.tensor(train_inputs),
            torch.tensor(train_masks),
            torch.tensor(df_classification.values),
            torch.tensor(df_sentiments.values)
        )
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
            )
        
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=batch_size
            )
        
        for batch in train_dataloader:
            batch_labels = batch[2].tolist()
            self.logger.info(f"Batch label distribution: {Counter(batch_labels)}")
            break


        model = self.model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        criterion = nn.CrossEntropyLoss()

        # ---- PRINT WHEN FINE-TUNING STARTS ----
        print("fine-tuning in process")

        with tqdm(total=self.NUM_TRAIN_EPOCHS, desc="Epochs") as pbar:
            try:
                for epoch in range(self.NUM_TRAIN_EPOCHS):
                    for step, batch in enumerate(train_dataloader):
                        model.train()
                        input_ids = batch[0].to(device)
                        attention_mask = batch[1].to(device)
                        classification_labels = batch[2].to(device)
                        sentiment_labels = batch[3].to(device)

                        optimizer.zero_grad()

                        classification_logits, sentiment_logits, classification_probs, sentiment_probs = \
                            model(input_ids, attention_mask)

                        classification_loss = criterion(classification_logits, classification_labels)
                        sentiment_loss = criterion(sentiment_logits, sentiment_labels)

                        loss = classification_loss + sentiment_loss

                        loss.backward()
                        optimizer.step()

                        pbar.set_postfix(loss=loss.item())
                        pbar.update(1)
            except Exception as e:
                self.logger.error(str(e), exc_info=True)
                raise

        # directory already used above, but keep this safe check
        directory = os.path.dirname(self.finedtunnedmodelpath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model.SaveModel(self.finedtunnedmodelpath)
        tokenizer.save_pretrained(self.finedtunnedmodelpath)

        # ---------- EVALUATION ON 20% TEST SPLIT ----------
        if len(test_df) > 0:
            self.logger.info("Running evaluation on 20% test split...")

            model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            test_tokenized = [tokenizer.encode(text, add_special_tokens=True) for text in test_df["text"]]
            test_masks = [[int(token_id > 0) for token_id in input_id] for input_id in test_tokenized]

            test_inputs = pad_sequences(
                test_tokenized, maxlen=MAX_LEN, dtype='long', value=0,
                truncating='post', padding='post'
            )
            test_masks = pad_sequences(
                test_masks, maxlen=MAX_LEN, dtype='long', value=0,
                truncating='post', padding='post'
            )

            test_dataset = TensorDataset(
                torch.tensor(test_inputs),
                torch.tensor(test_masks),
            )
            test_sampler = SequentialSampler(test_dataset)
            test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

            all_class_true_ids = test_df["classification"].astype(np.int64).tolist()
            all_sent_true_ids = test_df["sentiment"].astype(np.int64).tolist()

            all_class_pred_ids = []
            all_sent_pred_ids = []
            all_class_score = []
            all_sent_score = []

            with torch.no_grad():
                for batch in test_loader:
                    input_ids, attention_mask = [t.to(device) for t in batch]

                    classification_logits, sentiment_logits, _, _ = model(input_ids, attention_mask)

                    class_probs = nn.Softmax(dim=1)(classification_logits)
                    sent_probs = nn.Softmax(dim=1)(sentiment_logits)

                    class_preds = torch.argmax(class_probs, dim=1)
                    sent_preds = torch.argmax(sent_probs, dim=1)

                    all_class_pred_ids.extend(class_preds.cpu().numpy().tolist())
                    all_sent_pred_ids.extend(sent_preds.cpu().numpy().tolist())

                    all_class_score.extend(
                        class_probs.gather(1, class_preds.unsqueeze(1)).squeeze(1).cpu().numpy().tolist()
                    )
                    all_sent_score.extend(
                        sent_probs.gather(1, sent_preds.unsqueeze(1)).squeeze(1).cpu().numpy().tolist()
                    )

            # Decode ids back to label strings
            class_encoder = self.processed_label_encoders['classification']
            sent_encoder = self.processed_label_encoders['sentiment']

            decoded_class_true = class_encoder.inverse_transform(np.array(all_class_true_ids))
            decoded_class_pred = class_encoder.inverse_transform(np.array(all_class_pred_ids))

            decoded_sent_true = sent_encoder.inverse_transform(np.array(all_sent_true_ids))
            decoded_sent_pred = sent_encoder.inverse_transform(np.array(all_sent_pred_ids))

            # Build evaluation dataframe
            eval_df = pd.DataFrame({
                "text": test_df["text"],
                "SubQueryType": test_df["SubQueryType"],
                "classification_true": decoded_class_true,
                "classification_pred": decoded_class_pred,
                "classification_score": all_class_score,
                "sentiment_true": decoded_sent_true,
                "sentiment_pred": decoded_sent_pred,
                "sentiment_score": all_sent_score,
            })

            eval_dir = os.path.join(directory, "evaluation")
            os.makedirs(eval_dir, exist_ok=True)
            eval_path = os.path.join(eval_dir, "test_predictions.csv")

            eval_df.to_csv(eval_path, index=False)
            self.logger.info(f"Saved test predictions to {eval_path}")

            # ---------- METRICS (ACCURACY, F1, CLASSIFICATION REPORT) ----------
            self.logger.info("Calculating evaluation metrics for classification and sentiment...")

            # Classification metrics (string labels)
            class_accuracy = accuracy_score(decoded_class_true, decoded_class_pred)
            class_f1_weighted = f1_score(decoded_class_true, decoded_class_pred, average='weighted')
            class_report = classification_report(decoded_class_true, decoded_class_pred)

            # Sentiment metrics (string labels)
            sent_accuracy = accuracy_score(decoded_sent_true, decoded_sent_pred)
            sent_f1_weighted = f1_score(decoded_sent_true, decoded_sent_pred, average='weighted')
            sent_report = classification_report(decoded_sent_true, decoded_sent_pred)

            # Log to console/logs
            self.logger.info(f"Classification Accuracy: {class_accuracy:.4f}")
            self.logger.info(f"Classification F1 (weighted): {class_f1_weighted:.4f}")
            self.logger.info(f"Classification report:\n{class_report}")

            self.logger.info(f"Sentiment Accuracy: {sent_accuracy:.4f}")
            self.logger.info(f"Sentiment F1 (weighted): {sent_f1_weighted:.4f}")
            self.logger.info(f"Sentiment report:\n{sent_report}")

            # Save metrics to text file
            metrics_path = os.path.join(eval_dir, "metrics.txt")
            with open(metrics_path, "w", encoding="utf-8") as f:
                f.write("=== Classification Metrics ===\n")
                f.write(f"Accuracy: {class_accuracy:.4f}\n")
                f.write(f"F1 (weighted): {class_f1_weighted:.4f}\n\n")
                f.write("Classification report:\n")
                f.write(class_report)
                f.write("\n\n=== Sentiment Metrics ===\n")
                f.write(f"Accuracy: {sent_accuracy:.4f}\n")
                f.write(f"F1 (weighted): {sent_f1_weighted:.4f}\n\n")
                f.write("Sentiment report:\n")
                f.write(sent_report)

            self.logger.info(f"Saved evaluation metrics to {metrics_path}")

            self.save_metrics_to_sqlite(
            class_accuracy=class_accuracy,
            class_f1_weighted=class_f1_weighted,
            sent_accuracy=sent_accuracy,
            sent_f1_weighted=sent_f1_weighted,
            train_size=len(train_df),
            test_size=len(test_df)
            )

# ================== ADD BELOW THIS LINE ==================

# ---- Fetch latest run_id ----
            import sqlite3

            base_dir = os.path.dirname(self.finedtunnedmodelpath)
            db_path = os.path.join(base_dir, "evaluation", "evaluation.db")

            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT MAX(run_id) FROM run_info")
            run_id = cur.fetchone()[0]
            conn.close()

# ---- Save FULL metrics ----
            self.save_full_classification_report_to_sqlite(
            run_id=run_id,
            task_name="classification",
            y_true=decoded_class_true,
            y_pred=decoded_class_pred
            )

            self.save_full_classification_report_to_sqlite(
            run_id=run_id,
            task_name="sentiment",
            y_true=decoded_sent_true,
            y_pred=decoded_sent_pred
            )

# ================== ADD ABOVE THIS LINE ==================

        
            

    def fit_external(self, event, data, tasks, **kwargs):
        """Download dataset from external source and prepare data for training in BERT"""
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            self.logger.info(f"Skip training: event {event} is not supported")
            return

        ds_raw = tasks

        df = pd.DataFrame.from_dict(ds_raw)
        df = df.rename(columns={"sentiment": "sentiment_label"})

        df["sentiment"] = self.processed_label_encoders['sentiment'].transform(df["sentiment_label"])
        df["classification"] = self.processed_label_encoders['classification'].transform(df["label"])

        # ---- 80/20 train/test split for external fit (no SubQueryType here) ----
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        self.logger.info(f"[External fit] Total samples: {len(df)}, Train: {len(train_df)}, Test: {len(test_df)}")

        MAX_LEN = 256  # Define the maximum length of tokenized texts
        batch_size = 16
        tokenizer = self.tokenizer

        tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in train_df['text']]

        # Create attention masks
        train_masks = [[int(token_id > 0) for token_id in input_id] for input_id in tokenized_texts]

        train_inputs = pad_sequences(
            tokenized_texts, maxlen=MAX_LEN, dtype='long', value=0,
            truncating='post', padding='post'
        )
        train_masks = pad_sequences(
            train_masks, maxlen=MAX_LEN, dtype='long', value=0,
            truncating='post', padding='post'
        )
        df_classification = train_df["classification"].astype(np.int64)
        df_sentiments = train_df['sentiment'].astype(np.int64)
        train_data = TensorDataset(
            torch.tensor(train_inputs),
            torch.tensor(train_masks),
            torch.tensor(df_classification.values),
            torch.tensor(df_sentiments.values)
        )
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        model = self.model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        model.to(device)

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

                    classification_logits, sentiment_logits, classification_probs, sentiment_probs = \
                        model(input_ids, attention_mask)

                    classification_loss = criterion(classification_logits, classification_labels)
                    sentiment_loss = criterion(sentiment_logits, sentiment_labels)

                    loss = classification_loss + sentiment_loss

                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)

        directory = os.path.dirname(self.finedtunnedmodelpath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model.SaveModel(self.finedtunnedmodelpath)
        tokenizer.save_pretrained(self.finedtunnedmodelpath)

        # ---------- EVALUATION ON 20% TEST SPLIT (EXTERNAL) ----------
        if len(test_df) > 0:
            self.logger.info("Running external evaluation on 20% test split...")

            model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            test_tokenized = [tokenizer.encode(text, add_special_tokens=True) for text in test_df["text"]]
            test_masks = [[int(token_id > 0) for token_id in input_id] for input_id in test_tokenized]

            test_inputs = pad_sequences(
                test_tokenized, maxlen=MAX_LEN, dtype='long', value=0,
                truncating='post', padding='post'
            )
            test_masks = pad_sequences(
                test_masks, maxlen=MAX_LEN, dtype='long', value=0,
                truncating='post', padding='post'
            )

            test_dataset = TensorDataset(
                torch.tensor(test_inputs),
                torch.tensor(test_masks),
            )
            test_sampler = SequentialSampler(test_dataset)
            test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

            all_class_true_ids = test_df["classification"].astype(np.int64).tolist()
            all_sent_true_ids = test_df["sentiment"].astype(np.int64).tolist()

            all_class_pred_ids = []
            all_sent_pred_ids = []
            all_class_score = []
            all_sent_score = []

            with torch.no_grad():
                for batch in test_loader:
                    input_ids, attention_mask = [t.to(device) for t in batch]

                    classification_logits, sentiment_logits, _, _ = model(input_ids, attention_mask)

                    class_probs = nn.Softmax(dim=1)(classification_logits)
                    sent_probs = nn.Softmax(dim=1)(sentiment_logits)

                    class_preds = torch.argmax(class_probs, dim=1)
                    sent_preds = torch.argmax(sent_probs, dim=1)

                    all_class_pred_ids.extend(class_preds.cpu().numpy().tolist())
                    all_sent_pred_ids.extend(sent_preds.cpu().numpy().tolist())

                    all_class_score.extend(
                        class_probs.gather(1, class_preds.unsqueeze(1)).squeeze(1).cpu().numpy().tolist()
                    )
                    all_sent_score.extend(
                        sent_probs.gather(1, sent_preds.unsqueeze(1)).squeeze(1).cpu().numpy().tolist()
                    )

            # Decode ids back to label strings
            class_encoder = self.processed_label_encoders['classification']
            sent_encoder = self.processed_label_encoders['sentiment']

            decoded_class_true = class_encoder.inverse_transform(np.array(all_class_true_ids))
            decoded_class_pred = class_encoder.inverse_transform(np.array(all_class_pred_ids))

            decoded_sent_true = sent_encoder.inverse_transform(np.array(all_sent_true_ids))
            decoded_sent_pred = sent_encoder.inverse_transform(np.array(all_sent_pred_ids))

            eval_df = pd.DataFrame({
                "text": test_df["text"],
                "classification_true": decoded_class_true,
                "classification_pred": decoded_class_pred,
                "classification_score": all_class_score,
                "sentiment_true": decoded_sent_true,
                "sentiment_pred": decoded_sent_pred,
                "sentiment_score": all_sent_score,
            })

            eval_dir = os.path.join(directory, "evaluation_external")
            os.makedirs(eval_dir, exist_ok=True)
            eval_path = os.path.join(eval_dir, "test_predictions_external.csv")

            eval_df.to_csv(eval_path, index=False)
            self.logger.info(f"Saved external test predictions to {eval_path}")

            # ---------- METRICS (EXTERNAL) ----------
            self.logger.info("Calculating external evaluation metrics...")

            class_accuracy = accuracy_score(decoded_class_true, decoded_class_pred)
            class_f1_weighted = f1_score(decoded_class_true, decoded_class_pred, average='weighted')
            class_report = classification_report(decoded_class_true, decoded_class_pred)

            sent_accuracy = accuracy_score(decoded_sent_true, decoded_sent_pred)
            sent_f1_weighted = f1_score(decoded_sent_true, decoded_sent_pred, average='weighted')
            sent_report = classification_report(decoded_sent_true, decoded_sent_pred)

            self.logger.info(f"[External] Classification Accuracy: {class_accuracy:.4f}")
            self.logger.info(f"[External] Classification F1 (weighted): {class_f1_weighted:.4f}")
            self.logger.info(f"[External] Classification report:\n{class_report}")

            self.logger.info(f"[External] Sentiment Accuracy: {sent_accuracy:.4f}")
            self.logger.info(f"[External] Sentiment F1 (weighted): {sent_f1_weighted:.4f}")
            self.logger.info(f"[External] Sentiment report:\n{sent_report}")

            metrics_path = os.path.join(eval_dir, "metrics_external.txt")
            with open(metrics_path, "w", encoding="utf-8") as f:
                f.write("=== External Classification Metrics ===\n")
                f.write(f"Accuracy: {class_accuracy:.4f}\n")
                f.write(f"F1 (weighted): {class_f1_weighted:.4f}\n\n")
                f.write("Classification report:\n")
                f.write(class_report)
                f.write("\n\n=== External Sentiment Metrics ===\n")
                f.write(f"Accuracy: {sent_accuracy:.4f}\n")
                f.write(f"F1 (weighted): {sent_f1_weighted:.4f}\n\n")
                f.write("Sentiment report:\n")
                f.write(sent_report)

            self.logger.info(f"Saved external evaluation metrics to {metrics_path}")

    def predict(self, tasks: List[Dict], texts: str, context: Optional[Dict] = None, **kwargs):
        """Inference logic"""
        def getClassificationAttrName(attrs):
            return attrs == 'classification'

        def getSentimentAttrName(attrs):
            return attrs == 'sentiment'

        li = self.label_interface
        from_name_classification, to_name_classification, value_classification = \
            self.label_interface.get_first_tag_occurence('Taxonomy', 'HyperText', getClassificationAttrName)
        from_name_sentiment, to_name_sentiment, value_sentiment = \
            self.label_interface.get_first_tag_occurence('Taxonomy', 'HyperText', getSentimentAttrName)

        tokenizer = self.tokenizer
        tokenized_texts = [
            tokenizer.encode(text['text'] if not isinstance(text, str) else text, add_special_tokens=True)
            for text in texts
        ]

        attention_mask = [[int(token_id > 0) for token_id in input_id] for input_id in tokenized_texts]
        MAX_LEN = 512
        batch_size = 16
        _inputs = pad_sequences(
            tokenized_texts, maxlen=MAX_LEN, dtype='long', value=0,
            truncating='post', padding='post'
        )
        _masks = pad_sequences(
            attention_mask, maxlen=MAX_LEN, dtype='long', value=0,
            truncating='post', padding='post'
        )

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

                classification_logits, sentiment_logits, classification_probs1, sentiment_probs1 = \
                    self.model(input_ids, attention_mask)

                classification_probs = nn.Softmax(dim=1)(classification_logits)
                sentiment_probs = nn.Softmax(dim=1)(sentiment_logits)

                classification_preds = torch.argmax(classification_probs, dim=1)
                sentiment_preds = torch.argmax(sentiment_probs, dim=1)

                decoded_classification_preds = self.processed_label_encoders['classification'].inverse_transform(
                    classification_preds.cpu().numpy()
                )
                decoded_sentiment_preds = self.processed_label_encoders['sentiment'].inverse_transform(
                    sentiment_preds.cpu().numpy()
                )

                for i in range(len(classification_probs)):
                    predictions.append({
                        'from_name': from_name_classification,
                        'to_name': to_name_classification,
                        'type': 'taxonomy',
                        'value': {
                            'taxonomy': [decoded_classification_preds[i].split(' > ')],
                            'score': classification_probs.gather(
                                1, classification_preds.unsqueeze(1)
                            ).squeeze(1)[0].item()
                        },
                    })
                    predictions.append({
                        'from_name': from_name_sentiment,
                        'to_name': to_name_sentiment,
                        'type': 'taxonomy',
                        'value': {
                            'taxonomy': [[decoded_sentiment_preds[i]]],
                            'score': sentiment_probs.gather(
                                1, sentiment_preds.unsqueeze(1)
                            ).squeeze(1)[0].item()
                        },
                    })
        return predictions
