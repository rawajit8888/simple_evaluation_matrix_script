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

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.utils import pad_sequences
from transformers import pipeline, Pipeline, AdamW
from itertools import groupby
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer
from transformers import DataCollatorForTokenClassification
from datasets import Dataset, ClassLabel, Value, Sequence, Features
from functools import partial
from dataprocessing.processlabels import ProcessLabels
from multitask_nn_model import MultiTaskNNModel
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s'
)
logger = logging.getLogger(__name__)


class MultiTaskBertModel:

    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', '5d1a32bb14129720a48ddeb09b5cc99f1e39cc7c')
    START_TRAINING_EACH_N_UPDATES = int(os.getenv('START_TRAINING_EACH_N_UPDATES', 10))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-3))
    NUM_TRAIN_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 10))
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', 0.01))

    # <<< CHUNKING: global chunk settings
    CHUNK_SIZE   = int(os.getenv('CHUNK_SIZE',   256))   # tokens per chunk (excl. [CLS]/[SEP])
    CHUNK_STRIDE = int(os.getenv('CHUNK_STRIDE', 128))   # overlap between consecutive chunks

    # ============================================================
    # INIT
    # ============================================================
    def __init__(self, config, logger, label_interface=None, parsed_label_config=None):
        logger.info("[MultiTaskBertModel.__init__] Initializing...")
        self.config = config
        self.logger = logger
        self.label_interface = label_interface
        self.parsed_label_config = parsed_label_config

        if parsed_label_config is not None:
            self.logger.info("[__init__] parsed_label_config provided — initializing ProcessLabels.")
            try:
                self.processed_label_encoders = ProcessLabels(
                    self.parsed_label_config,
                    pathlib.Path(self.config['MODEL_DIR'])
                )
                self.logger.info("[__init__] ProcessLabels initialized.")
            except Exception as e:
                self.logger.error(f"[__init__] ProcessLabels FAILED: {e}", exc_info=True)
                self.processed_label_encoders = None
        else:
            self.logger.warning("[__init__] parsed_label_config is None — ProcessLabels deferred.")
            self.processed_label_encoders = ProcessLabels(
                self.parsed_label_config,
                pathlib.Path(self.config['MODEL_DIR'])
            )

        self.logger.info("[__init__] MultiTaskBertModel initialization complete.")

    # ============================================================
    # <<< CHUNKING HELPER
    # ============================================================
    def _get_chunks(self, token_ids: List[int]) -> List[List[int]]:
        """
        Split a single list of token_ids (already encoded, WITHOUT special tokens)
        into overlapping chunks of CHUNK_SIZE tokens.
        Each chunk is wrapped with [CLS]=101 and [SEP]=102 so BERT sees proper input.

        Returns a list of token-id lists, each of length <= CHUNK_SIZE + 2.
        If the text fits in one chunk it is returned as-is (still wrapped).
        """
        size   = self.CHUNK_SIZE      # e.g. 256
        stride = self.CHUNK_STRIDE    # e.g. 128

        # Strip leading [CLS] / trailing [SEP] if the tokenizer added them
        if token_ids and token_ids[0] == 101:
            token_ids = token_ids[1:]
        if token_ids and token_ids[-1] == 102:
            token_ids = token_ids[:-1]

        chunks = []
        start = 0
        while start < len(token_ids):
            end = start + size
            chunk = [101] + token_ids[start:end] + [102]   # [CLS] … [SEP]
            chunks.append(chunk)
            if end >= len(token_ids):
                break
            start += stride   # slide forward by stride (creates overlap)

        return chunks if chunks else [[101, 102]]   # safety: never return empty

    # ============================================================
    # RELOAD MODEL
    # ============================================================
    def reload_model(self):
        self.logger.info("[reload_model] Starting model reload...")
        self.model = None

        try:
            classification_label_length = len(
                self.processed_label_encoders.labels["classification"]["encoder"].classes_
            ) if (self.processed_label_encoders and
                  self.processed_label_encoders.labels["classification"]["encoder"] is not None) else 1000
        except Exception as e:
            self.logger.warning(f"[reload_model] Could not get classification label length: {e}. Using 1000.")
            classification_label_length = 1000

        try:
            sentiment_label_length = len(
                self.processed_label_encoders.labels["sentiment"]["encoder"].classes_
            ) if (self.processed_label_encoders and
                  self.processed_label_encoders.labels["sentiment"]["encoder"] is not None) else 3
        except Exception as e:
            self.logger.warning(f"[reload_model] Could not get sentiment label length: {e}. Using 3.")
            sentiment_label_length = 3

        self.logger.info(
            f"[reload_model] classification_label_length={classification_label_length}, "
            f"sentiment_label_length={sentiment_label_length}"
        )

        try:
            self.chk_path = str(pathlib.Path(self.config['MODEL_DIR']) / self.config['FINETUNED_MODEL_NAME'])
            self.finedtunnedmodelpath = self.chk_path
            self.logger.info(f"[reload_model] Trying finetuned model at: {self.chk_path}")
            self.model = MultiTaskNNModel(self.chk_path, classification_label_length, sentiment_label_length)
            self.model.LoadModel(self.chk_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.chk_path)
            self.logger.info(f"[reload_model] Finetuned model loaded successfully from: {self.chk_path}")

        except Exception as e:
            self.logger.warning(f"[reload_model] Finetuned model load FAILED: {e}")
            self.logger.info("[reload_model] Falling back to baseline model...")

            try:
                self.chk_path = str(pathlib.Path(self.config['MODEL_DIR']) / self.config['BASELINE_MODEL_NAME'])
                self.logger.info(f"[reload_model] Loading baseline model from: {self.chk_path}")
                self.model = MultiTaskNNModel(self.chk_path, classification_label_length, sentiment_label_length)
                self.tokenizer = AutoTokenizer.from_pretrained(self.chk_path)
                self.finedtunnedmodelpath = str(
                    pathlib.Path(self.config['MODEL_DIR']) / self.config['FINETUNED_MODEL_NAME']
                )
                self.logger.info(f"[reload_model] Baseline model loaded. Will save finetuned to: {self.finedtunnedmodelpath}")

            except Exception as e2:
                self.logger.error(f"[reload_model] Baseline model load ALSO FAILED: {e2}", exc_info=True)

    # ============================================================
    # SQLITE HELPERS
    # ============================================================
    def save_metrics_to_sqlite(self, class_accuracy, class_f1_weighted, sent_accuracy,
                                sent_f1_weighted, train_size, test_size):
        import sqlite3
        from datetime import datetime

        self.logger.info("[save_metrics_to_sqlite] Saving summary metrics to SQLite...")

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
        )""")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            task TEXT,
            metric TEXT,
            value REAL
        )""")

        cur.execute("""
        INSERT INTO run_info (created_at, train_size, test_size)
        VALUES (?, ?, ?)
        """, (datetime.now().isoformat(), train_size, test_size))

        run_id = cur.lastrowid
        self.logger.info(f"[save_metrics_to_sqlite] New run_id={run_id}")

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
        self.logger.info("[save_metrics_to_sqlite] Metrics saved successfully.")

    def save_full_classification_report_to_sqlite(self, run_id, task_name, y_true, y_pred):
        import sqlite3
        from sklearn.metrics import classification_report

        self.logger.info(f"[save_full_classification_report_to_sqlite] task={task_name}, run_id={run_id}")

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
        self.logger.info(f"[save_full_classification_report_to_sqlite] Saved for task={task_name}.")

    def get_latest_metrics_from_db(self):
        import sqlite3

        base_dir = os.path.dirname(self.finedtunnedmodelpath)
        db_path = os.path.join(base_dir, "evaluation", "evaluation.db")
        self.logger.info(f"[get_latest_metrics_from_db] Reading from: {db_path}")

        if not os.path.exists(db_path):
            self.logger.warning("[get_latest_metrics_from_db] evaluation.db not found.")
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

        self.logger.info(f"[get_latest_metrics_from_db] Fetched {len(metrics)} metric rows for run_id={run['run_id']}.")
        return {"run_info": dict(run), "metrics": metrics}

    # ============================================================
    # FIT (Label Studio)
    # ============================================================
    def fit(self, event, data, tasks, **kwargs):
        """Train model on Label Studio annotated tasks"""
        print(f">>> [MultiTaskBertModel.fit] CALLED — event={event}, tasks={len(tasks)}")
        self.logger.info(f"[fit] ============ FIT STARTED ============")
        self.logger.info(f"[fit] event={event}, tasks_count={len(tasks)}")

        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            self.logger.info(f"[fit] Unsupported event: {event}. Skipping.")
            return

        if self.label_interface is None:
            self.logger.error("[fit] label_interface is None! Cannot proceed with training.")
            return

        ds_raw = []

        def getClassificationAttrName(attrs):
            return attrs == 'classification'

        def getSentimentAttrName(attrs):
            return attrs == 'sentiment'

        try:
            from_name_classification, to_name_classification, value_classification = \
                self.label_interface.get_first_tag_occurence('Taxonomy', 'HyperText', getClassificationAttrName)
            from_name_sentiment, to_name_sentiment, value_sentiment = \
                self.label_interface.get_first_tag_occurence('Taxonomy', 'HyperText', getSentimentAttrName)
            self.logger.info(
                f"[fit] Tag names resolved — "
                f"classification: {from_name_classification}, sentiment: {from_name_sentiment}"
            )
        except Exception as e:
            self.logger.error(f"[fit] FAILED to resolve tag names: {e}", exc_info=True)
            return

        tokenizer = self.tokenizer

        self.logger.info("[fit] Building dataset from tasks...")
        for i, task in enumerate(tasks):
            subquery_type = task['data'].get('SubQueryType', 'Unknown')
            for annotation in task['annotations']:
                if not annotation.get('result'):
                    self.logger.debug(f"[fit] Task {i}: annotation has no result, skipping.")
                    continue

                try:
                    sentiment_label = [r['value']['taxonomy'][0][0] for r in annotation['result']
                                       if r["from_name"] == "sentiment"][0]
                    classification_label = " > ".join(
                        [r['value']['taxonomy'][0] for r in annotation['result']
                         if r["from_name"] == "classification"][0]
                    )

                    match = re.search(r"pre[^>]*>\s*(.*?)\s*</pre>", value_classification, re.DOTALL)
                    value_classification_clean = match.group(1)[1:] if match else value_classification

                    text = self.preload_task_data(task, task['data'][value_classification_clean])

                    sentiment = self.processed_label_encoders['sentiment'].transform([sentiment_label])[0]
                    classification = self.processed_label_encoders['classification'].transform([classification_label])[0]

                    ds_raw.append([text, classification_label, sentiment_label,
                                   classification, sentiment, subquery_type])

                except Exception as e:
                    self.logger.warning(f"[fit] Task {i} annotation parsing FAILED: {e}")
                    continue

        self.logger.info(f"[fit] Dataset built: {len(ds_raw)} samples.")

        if len(ds_raw) == 0:
            self.logger.error("[fit] No valid training samples found! Aborting training.")
            return

        df = pd.DataFrame(
            ds_raw,
            columns=["text", "classification_label", "sentiment_label",
                     "classification", "sentiment", "SubQueryType"]
        )

        # ---- Stratified 80/20 split ----
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df["SubQueryType"] = df["SubQueryType"].fillna("Unknown")

        train_indices, test_indices = [], []
        rng = np.random.default_rng(42)

        for subq, group in df.groupby("SubQueryType"):
            idx = group.index.to_list()
            rng.shuffle(idx)
            n = len(idx)
            if n < 2:
                train_indices.extend(idx)
                continue
            n_train = max(int(np.floor(0.8 * n)), 1)
            train_indices.extend(idx[:n_train])
            test_indices.extend(idx[n_train:])

        if len(test_indices) == 0 and len(df) > 1:
            all_idx = list(df.index)
            rng.shuffle(all_idx)
            split_idx = int(len(all_idx) * 0.8)
            train_indices = all_idx[:split_idx]
            test_indices = all_idx[split_idx:]

        train_df = df.loc[train_indices].reset_index(drop=True)
        test_df  = df.loc[test_indices].reset_index(drop=True)

        self.logger.info(f"[fit] Split — Total: {len(df)}, Train: {len(train_df)}, Test: {len(test_df)}")

        directory = os.path.dirname(self.finedtunnedmodelpath)
        eval_dir  = os.path.join(directory, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        raw_test_path = os.path.join(eval_dir, "test_raw_before_training.csv")
        test_df[["text", "SubQueryType", "classification_label", "sentiment_label"]].to_csv(raw_test_path, index=False)
        self.logger.info(f"[fit] Raw test split saved to: {raw_test_path}")

        # ---- <<< CHUNKING: tokenize + expand by chunks ----
        MAX_LEN    = self.CHUNK_SIZE + 2   # +2 for [CLS] and [SEP]
        batch_size = 16

        self.logger.info(
            f"[fit] Tokenizing with chunking "
            f"(CHUNK_SIZE={self.CHUNK_SIZE}, CHUNK_STRIDE={self.CHUNK_STRIDE})..."
        )

        chunked_inputs, chunked_masks, chunked_class_labels, chunked_sent_labels = [], [], [], []

        for _, row in train_df.iterrows():
            raw_ids = tokenizer.encode(row['text'], add_special_tokens=False)   # no special tokens yet
            chunks  = self._get_chunks(raw_ids)                                  # each chunk has [CLS]…[SEP]

            for chunk in chunks:
                mask = [1] * len(chunk)
                # Pad to MAX_LEN
                pad_len = MAX_LEN - len(chunk)
                chunk  += [0] * pad_len
                mask   += [0] * pad_len
                chunked_inputs.append(chunk[:MAX_LEN])
                chunked_masks.append(mask[:MAX_LEN])
                chunked_class_labels.append(int(row['classification']))
                chunked_sent_labels.append(int(row['sentiment']))

        self.logger.info(
            f"[fit] After chunking: {len(train_df)} samples → {len(chunked_inputs)} chunks"
        )
        # ---- <<< END CHUNKING ----

        train_data = TensorDataset(
            torch.tensor(chunked_inputs,       dtype=torch.long),
            torch.tensor(chunked_masks,        dtype=torch.long),
            torch.tensor(chunked_class_labels, dtype=torch.long),
            torch.tensor(chunked_sent_labels,  dtype=torch.long),
        )
        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

        model  = self.model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"[fit] Training device: {device}")
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        criterion = nn.CrossEntropyLoss()

        self.logger.info(f"[fit] Starting fine-tuning for {self.NUM_TRAIN_EPOCHS} epochs...")
        print(">>> [fit] fine-tuning in process...")

        try:
            with tqdm(total=self.NUM_TRAIN_EPOCHS, desc="Epochs") as pbar:
                for epoch in range(self.NUM_TRAIN_EPOCHS):
                    self.logger.info(f"[fit] Epoch {epoch + 1}/{self.NUM_TRAIN_EPOCHS}")
                    for step, batch in enumerate(train_dataloader):
                        model.train()
                        input_ids          = batch[0].to(device)
                        attention_mask     = batch[1].to(device)
                        classification_labels = batch[2].to(device)
                        sentiment_labels   = batch[3].to(device)

                        optimizer.zero_grad()

                        classification_logits, sentiment_logits, classification_probs, sentiment_probs = \
                            model(input_ids, attention_mask)

                        classification_loss = criterion(classification_logits, classification_labels)
                        sentiment_loss      = criterion(sentiment_logits, sentiment_labels)
                        loss = classification_loss + sentiment_loss

                        loss.backward()
                        optimizer.step()
                        pbar.set_postfix(loss=loss.item())

                    pbar.update(1)

        except Exception as e:
            self.logger.error(f"[fit] Training loop FAILED: {e}", exc_info=True)
            raise

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.logger.info(f"[fit] Saving model to: {self.finedtunnedmodelpath}")
        model.SaveModel(self.finedtunnedmodelpath)
        tokenizer.save_pretrained(self.finedtunnedmodelpath)
        self.logger.info("[fit] Model saved.")

        if len(test_df) > 0:
            self.logger.info("[fit] Running evaluation on test split...")
            self._run_evaluation(model, tokenizer, test_df, MAX_LEN, batch_size, device, directory, eval_dir)
        else:
            self.logger.warning("[fit] No test samples — skipping evaluation.")

        self.logger.info("[fit] ============ FIT COMPLETE ============")
        print(">>> [fit] Training complete!")

    # ============================================================
    # SHARED EVALUATION HELPER
    # ============================================================
    def _run_evaluation(self, model, tokenizer, test_df, MAX_LEN, batch_size, device, directory, eval_dir):
        """Runs evaluation and saves results. Uses chunk-then-average for long texts."""
        self.logger.info(f"[_run_evaluation] Evaluating on {len(test_df)} test samples...")

        model.eval()
        model.to(device)

        # <<< CHUNKING: per-sample chunk inference, then average probs per sample
        all_class_probs_per_sample = []   # list[np.array shape (n_classes,)]
        all_sent_probs_per_sample  = []

        with torch.no_grad():
            for _, row in test_df.iterrows():
                raw_ids    = tokenizer.encode(row['text'], add_special_tokens=False)
                chunks     = self._get_chunks(raw_ids)
                chunk_class_probs = []
                chunk_sent_probs  = []

                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i: i + batch_size]

                    # Pad each chunk to MAX_LEN
                    padded, masks = [], []
                    for ch in batch_chunks:
                        pad_len = MAX_LEN - len(ch)
                        padded.append(ch + [0] * pad_len)
                        masks.append([1] * len(ch) + [0] * pad_len)

                    input_ids_t  = torch.tensor(padded, dtype=torch.long).to(device)
                    attn_mask_t  = torch.tensor(masks,  dtype=torch.long).to(device)

                    classification_logits, sentiment_logits, _, _ = model(input_ids_t, attn_mask_t)

                    cp = nn.Softmax(dim=1)(classification_logits).cpu().numpy()   # (B, n_class)
                    sp = nn.Softmax(dim=1)(sentiment_logits).cpu().numpy()        # (B, n_sent)
                    chunk_class_probs.append(cp)
                    chunk_sent_probs.append(sp)

                # Average across all chunks for this sample
                all_class_probs_per_sample.append(np.vstack(chunk_class_probs).mean(axis=0))
                all_sent_probs_per_sample.append(np.vstack(chunk_sent_probs).mean(axis=0))
        # <<< END CHUNKING

        all_class_probs = np.vstack(all_class_probs_per_sample)   # (N, n_class)
        all_sent_probs  = np.vstack(all_sent_probs_per_sample)    # (N, n_sent)

        all_class_pred_ids = np.argmax(all_class_probs, axis=1).tolist()
        all_sent_pred_ids  = np.argmax(all_sent_probs,  axis=1).tolist()
        all_class_score    = all_class_probs[np.arange(len(all_class_pred_ids)), all_class_pred_ids].tolist()
        all_sent_score     = all_sent_probs [np.arange(len(all_sent_pred_ids)),  all_sent_pred_ids ].tolist()

        all_class_true_ids = test_df["classification"].astype(np.int64).tolist()
        all_sent_true_ids  = test_df["sentiment"].astype(np.int64).tolist()

        class_encoder = self.processed_label_encoders['classification']
        sent_encoder  = self.processed_label_encoders['sentiment']

        decoded_class_true = class_encoder.inverse_transform(np.array(all_class_true_ids))
        decoded_class_pred = class_encoder.inverse_transform(np.array(all_class_pred_ids))
        decoded_sent_true  = sent_encoder.inverse_transform(np.array(all_sent_true_ids))
        decoded_sent_pred  = sent_encoder.inverse_transform(np.array(all_sent_pred_ids))

        has_subquery = "SubQueryType" in test_df.columns
        eval_df = pd.DataFrame({
            "text": test_df["text"].values,
            **({"SubQueryType": test_df["SubQueryType"].values} if has_subquery else {}),
            "classification_true":  decoded_class_true,
            "classification_pred":  decoded_class_pred,
            "classification_score": all_class_score,
            "sentiment_true":  decoded_sent_true,
            "sentiment_pred":  decoded_sent_pred,
            "sentiment_score": all_sent_score,
        })

        eval_path = os.path.join(eval_dir, "test_predictions.csv")
        eval_df.to_csv(eval_path, index=False)
        self.logger.info(f"[_run_evaluation] Predictions saved to: {eval_path}")

        class_accuracy    = accuracy_score(decoded_class_true, decoded_class_pred)
        class_f1_weighted = f1_score(decoded_class_true, decoded_class_pred, average='weighted')
        class_report      = classification_report(decoded_class_true, decoded_class_pred)

        sent_accuracy     = accuracy_score(decoded_sent_true, decoded_sent_pred)
        sent_f1_weighted  = f1_score(decoded_sent_true, decoded_sent_pred, average='weighted')
        sent_report       = classification_report(decoded_sent_true, decoded_sent_pred)

        self.logger.info(f"[_run_evaluation] Classification Accuracy: {class_accuracy:.4f}")
        self.logger.info(f"[_run_evaluation] Classification F1 (weighted): {class_f1_weighted:.4f}")
        self.logger.info(f"[_run_evaluation] Classification Report:\n{class_report}")
        self.logger.info(f"[_run_evaluation] Sentiment Accuracy: {sent_accuracy:.4f}")
        self.logger.info(f"[_run_evaluation] Sentiment F1 (weighted): {sent_f1_weighted:.4f}")
        self.logger.info(f"[_run_evaluation] Sentiment Report:\n{sent_report}")

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
        self.logger.info(f"[_run_evaluation] Metrics saved to: {metrics_path}")

        self.save_metrics_to_sqlite(class_accuracy, class_f1_weighted,
                                    sent_accuracy, sent_f1_weighted,
                                    len(test_df), len(test_df))

        import sqlite3
        base_dir = os.path.dirname(self.finedtunnedmodelpath)
        db_path  = os.path.join(base_dir, "evaluation", "evaluation.db")
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        cur.execute("SELECT MAX(run_id) FROM run_info")
        run_id = cur.fetchone()[0]
        conn.close()

        self.save_full_classification_report_to_sqlite(
            run_id=run_id, task_name="classification",
            y_true=decoded_class_true, y_pred=decoded_class_pred
        )
        self.save_full_classification_report_to_sqlite(
            run_id=run_id, task_name="sentiment",
            y_true=decoded_sent_true, y_pred=decoded_sent_pred
        )
        self.logger.info("[_run_evaluation] Evaluation complete and saved to SQLite.")

    # ============================================================
    # FIT EXTERNAL
    # ============================================================
    def fit_external(self, event, data, tasks, **kwargs):
        """Train on externally provided tasks"""
        print(f">>> [MultiTaskBertModel.fit_external] CALLED — event={event}, tasks={len(tasks)}")
        self.logger.info(f"[fit_external] event={event}, tasks_count={len(tasks)}")

        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            self.logger.info(f"[fit_external] Unsupported event: {event}. Skipping.")
            return

        ds_raw = tasks
        df = pd.DataFrame.from_dict(ds_raw)
        df = df.rename(columns={"sentiment": "sentiment_label"})

        self.logger.info(f"[fit_external] DataFrame columns: {df.columns.tolist()}")

        df["sentiment"]      = self.processed_label_encoders['sentiment'].transform(df["sentiment_label"])
        df["classification"] = self.processed_label_encoders['classification'].transform(df["label"])

        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * 0.8)
        train_df  = df.iloc[:split_idx].reset_index(drop=True)
        test_df   = df.iloc[split_idx:].reset_index(drop=True)

        self.logger.info(f"[fit_external] Split — Total: {len(df)}, Train: {len(train_df)}, Test: {len(test_df)}")

        # <<< CHUNKING: tokenize + expand by chunks
        MAX_LEN    = self.CHUNK_SIZE + 2
        batch_size = 16
        tokenizer  = self.tokenizer

        self.logger.info(
            f"[fit_external] Tokenizing with chunking "
            f"(CHUNK_SIZE={self.CHUNK_SIZE}, CHUNK_STRIDE={self.CHUNK_STRIDE})..."
        )

        chunked_inputs, chunked_masks, chunked_class_labels, chunked_sent_labels = [], [], [], []

        for _, row in train_df.iterrows():
            raw_ids = tokenizer.encode(row['text'], add_special_tokens=False)
            chunks  = self._get_chunks(raw_ids)

            for chunk in chunks:
                mask    = [1] * len(chunk)
                pad_len = MAX_LEN - len(chunk)
                chunk  += [0] * pad_len
                mask   += [0] * pad_len
                chunked_inputs.append(chunk[:MAX_LEN])
                chunked_masks.append(mask[:MAX_LEN])
                chunked_class_labels.append(int(row['classification']))
                chunked_sent_labels.append(int(row['sentiment']))

        self.logger.info(
            f"[fit_external] After chunking: {len(train_df)} samples → {len(chunked_inputs)} chunks"
        )
        # <<< END CHUNKING

        train_data = TensorDataset(
            torch.tensor(chunked_inputs,       dtype=torch.long),
            torch.tensor(chunked_masks,        dtype=torch.long),
            torch.tensor(chunked_class_labels, dtype=torch.long),
            torch.tensor(chunked_sent_labels,  dtype=torch.long),
        )
        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

        model  = self.model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"[fit_external] Training device: {device}")
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        criterion = nn.CrossEntropyLoss()

        self.logger.info(f"[fit_external] Starting fine-tuning for {self.NUM_TRAIN_EPOCHS} epochs...")
        print(">>> [fit_external] fine-tuning in process...")

        with tqdm(total=self.NUM_TRAIN_EPOCHS, desc="Epochs") as pbar:
            for epoch in range(self.NUM_TRAIN_EPOCHS):
                self.logger.info(f"[fit_external] Epoch {epoch + 1}/{self.NUM_TRAIN_EPOCHS}")
                for step, batch in enumerate(train_dataloader):
                    model.train()
                    input_ids          = batch[0].to(device)
                    attention_mask     = batch[1].to(device)
                    classification_labels = batch[2].to(device)
                    sentiment_labels   = batch[3].to(device)

                    optimizer.zero_grad()
                    classification_logits, sentiment_logits, classification_probs, sentiment_probs = \
                        model(input_ids, attention_mask)

                    classification_loss = criterion(classification_logits, classification_labels)
                    sentiment_loss      = criterion(sentiment_logits, sentiment_labels)
                    loss = classification_loss + sentiment_loss

                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        directory = os.path.dirname(self.finedtunnedmodelpath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.logger.info(f"[fit_external] Saving model to: {self.finedtunnedmodelpath}")
        model.SaveModel(self.finedtunnedmodelpath)
        tokenizer.save_pretrained(self.finedtunnedmodelpath)
        self.logger.info("[fit_external] Model saved.")

        if len(test_df) > 0:
            eval_dir = os.path.join(directory, "evaluation_external")
            os.makedirs(eval_dir, exist_ok=True)
            self._run_evaluation(model, tokenizer, test_df, MAX_LEN, batch_size, device, directory, eval_dir)
        else:
            self.logger.warning("[fit_external] No test samples — skipping evaluation.")

        self.logger.info("[fit_external] ============ FIT EXTERNAL COMPLETE ============")
        print(">>> [fit_external] Training complete!")

    # ============================================================
    # PREDICT
    # ============================================================
    def predict(self, tasks: List[Dict], texts: str, context: Optional[Dict] = None, **kwargs):
        """Inference logic — handles long texts via chunk-then-average"""
        self.logger.info(f"[predict] Called. texts count={len(texts) if texts else 0}")

        def getClassificationAttrName(attrs):
            return attrs == 'classification'

        def getSentimentAttrName(attrs):
            return attrs == 'sentiment'

        try:
            from_name_classification, to_name_classification, value_classification = \
                self.label_interface.get_first_tag_occurence('Taxonomy', 'HyperText', getClassificationAttrName)
            from_name_sentiment, to_name_sentiment, value_sentiment = \
                self.label_interface.get_first_tag_occurence('Taxonomy', 'HyperText', getSentimentAttrName)
        except Exception as e:
            self.logger.error(f"[predict] Failed to resolve tag names: {e}", exc_info=True)
            return []

        tokenizer  = self.tokenizer
        MAX_LEN    = self.CHUNK_SIZE + 2
        batch_size = 16
        device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        predictions = []

        # <<< CHUNKING: per-sample chunk inference
        with torch.no_grad():
            for text_item in texts:
                raw_text = text_item['text'] if not isinstance(text_item, str) else text_item
                raw_ids  = tokenizer.encode(raw_text, add_special_tokens=False)
                chunks   = self._get_chunks(raw_ids)

                chunk_class_probs, chunk_sent_probs = [], []

                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i: i + batch_size]

                    padded, masks = [], []
                    for ch in batch_chunks:
                        pad_len = MAX_LEN - len(ch)
                        padded.append(ch + [0] * pad_len)
                        masks.append([1] * len(ch) + [0] * pad_len)

                    input_ids_t = torch.tensor(padded, dtype=torch.long).to(device)
                    attn_mask_t = torch.tensor(masks,  dtype=torch.long).to(device)

                    classification_logits, sentiment_logits, _, _ = self.model(input_ids_t, attn_mask_t)

                    cp = nn.Softmax(dim=1)(classification_logits).cpu().numpy()
                    sp = nn.Softmax(dim=1)(sentiment_logits).cpu().numpy()
                    chunk_class_probs.append(cp)
                    chunk_sent_probs.append(sp)

                # Average across all chunks
                avg_class_probs = np.vstack(chunk_class_probs).mean(axis=0)   # (n_class,)
                avg_sent_probs  = np.vstack(chunk_sent_probs).mean(axis=0)    # (n_sent,)

                class_pred_id = int(np.argmax(avg_class_probs))
                sent_pred_id  = int(np.argmax(avg_sent_probs))
                class_score   = float(avg_class_probs[class_pred_id])
                sent_score    = float(avg_sent_probs[sent_pred_id])

                decoded_class = self.processed_label_encoders['classification'].inverse_transform(
                    np.array([class_pred_id])
                )[0]
                decoded_sent  = self.processed_label_encoders['sentiment'].inverse_transform(
                    np.array([sent_pred_id])
                )[0]
        # <<< END CHUNKING

                predictions.append({
                    'from_name': from_name_classification,
                    'to_name':   to_name_classification,
                    'type':      'taxonomy',
                    'value': {
                        'taxonomy': [decoded_class.split(' > ')],
                        'score':    class_score,
                    },
                })
                predictions.append({
                    'from_name': from_name_sentiment,
                    'to_name':   to_name_sentiment,
                    'type':      'taxonomy',
                    'value': {
                        'taxonomy': [[decoded_sent]],
                        'score':    sent_score,
                    },
                })

        self.logger.info(f"[predict] Returning {len(predictions)} predictions.")
        return predictions