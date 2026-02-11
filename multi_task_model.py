import os
import pathlib
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime, timedelta

import label_studio_sdk
import logging

# SubQueryNNModel and DepartmentNNModel are now defined in this file (below) - no separate import needed

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
from transformers import pipeline, Pipeline, BertModel
from transformers import AdamW
from itertools import groupby
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer, AdamW
from transformers import DataCollatorForTokenClassification
from datasets import Dataset, ClassLabel, Value, Sequence, Features
from functools import partial
from dataprocessing.processlabels import ProcessLabels
from multitask_nn_model import MultiTaskNNModel
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pickle

logger = logging.getLogger(__name__)


# ========== DepartmentNNModel Class (Level 2 Model) ==========

class DepartmentNNModel(nn.Module):
    """
    Department classification model with bundled label encoder.
    Level 2: MasterDepartment → Department
    Uses same BERT backbone as MultiTask model for consistency.
    """
    
    def __init__(self, modelname, num_labels):
        super(DepartmentNNModel, self).__init__()
        
        # Use SAME backbone type as MultiTaskNNModel
        self.bert = BertModel.from_pretrained(modelname)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            num_labels
        )
        
        # Store encoder (will be set during training)
        self.encoder = None
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs[1]  # CLS pooled output
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        return logits
    
    def set_encoder(self, encoder):
        """Set label encoder before saving"""
        self.encoder = encoder
        logger.info("✓ Encoder attached to Department model")
    
    def save(self, directorypath):
        """Save model with bundled encoder"""
        try:
            os.makedirs(directorypath, exist_ok=True)
            
            # Save BERT backbone
            logger.info(f"📦 Saving Department BERT to {directorypath}")
            self.bert.save_pretrained(directorypath)
            
            # Save classifier head
            classifier_path = os.path.join(directorypath, "department_classifier.pth")
            logger.info(f"💾 Saving classifier head to {classifier_path}")
            torch.save(
                self.classifier.state_dict(),
                classifier_path
            )
            
            # Save bundled encoder
            encoder_path = os.path.join(directorypath, 'department_encoder.pkl')
            logger.info(f"💾 Saving department encoder to {encoder_path}")
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.encoder, f)
            
            logger.info("✅ Department model and encoder saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving Department model: {e}")
            raise
    
    def load(self, directorypath):
        """Load model with bundled encoder"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Reload BERT
            logger.info(f"📂 Loading Department BERT from {directorypath}")
            self.bert = BertModel.from_pretrained(directorypath)
            
            # Reload classifier head
            classifier_path = os.path.join(directorypath, "department_classifier.pth")
            logger.info(f"📂 Loading classifier head from {classifier_path}")
            self.classifier.load_state_dict(
                torch.load(
                    classifier_path,
                    map_location=device
                )
            )
            
            # Load bundled encoder
            encoder_path = os.path.join(directorypath, 'department_encoder.pkl')
            if os.path.exists(encoder_path):
                logger.info(f"📂 Loading bundled department encoder from {encoder_path}")
                with open(encoder_path, 'rb') as f:
                    self.encoder = pickle.load(f)
                logger.info("✅ Department model and encoder loaded successfully")
            else:
                logger.warning("⚠️  No bundled encoder found for Department - using external encoder")
            
            self.eval()
            
        except Exception as e:
            logger.error(f"❌ Error loading Department model: {e}")
            raise
    
    def get_encoder(self):
        """Safely retrieve encoder"""
        if self.encoder is None:
            raise ValueError("Department encoder is None")
        return self.encoder


# ========== SubQueryNNModel Class (Level 3 Model) ==========

class SubQueryNNModel(nn.Module):
    """
    QueryType classification model with bundled label encoder.
    Level 3: Department → QueryType
    Uses same BERT backbone as MultiTask model for consistency.
    
    NOTE: This class is now part of multi_task_model.py (no separate file needed)
    """
    
    def __init__(self, modelname, num_labels):
        super(SubQueryNNModel, self).__init__()
        
        # Use SAME backbone type as MultiTaskNNModel
        self.bert = BertModel.from_pretrained(modelname)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            num_labels
        )
        
        # Store encoder (will be set during training)
        self.encoder = None
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs[1]  # CLS pooled output
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        return logits
    
    def set_encoder(self, encoder):
        """Set label encoder before saving"""
        self.encoder = encoder
        logger.info("✓ Encoder attached to QueryType model")
    
    def save(self, directorypath):
        """Save model with bundled encoder"""
        try:
            os.makedirs(directorypath, exist_ok=True)
            
            # Save BERT backbone
            logger.info(f"📦 Saving QueryType BERT to {directorypath}")
            self.bert.save_pretrained(directorypath)
            
            # Save classifier head
            classifier_path = os.path.join(directorypath, "querytype_classifier.pth")
            logger.info(f"💾 Saving classifier head to {classifier_path}")
            torch.save(
                self.classifier.state_dict(),
                classifier_path
            )
            
            # Save bundled encoder
            encoder_path = os.path.join(directorypath, 'querytype_encoder.pkl')
            logger.info(f"💾 Saving querytype encoder to {encoder_path}")
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.encoder, f)
            
            logger.info("✅ QueryType model and encoder saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving QueryType model: {e}")
            raise
    
    def load(self, directorypath):
        """Load model with bundled encoder"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Reload BERT
            logger.info(f"📂 Loading QueryType BERT from {directorypath}")
            self.bert = BertModel.from_pretrained(directorypath)
            
            # Reload classifier head
            classifier_path = os.path.join(directorypath, "querytype_classifier.pth")
            logger.info(f"📂 Loading classifier head from {classifier_path}")
            self.classifier.load_state_dict(
                torch.load(
                    classifier_path,
                    map_location=device
                )
            )
            
            # Load bundled encoder
            encoder_path = os.path.join(directorypath, 'querytype_encoder.pkl')
            if os.path.exists(encoder_path):
                logger.info(f"📂 Loading bundled querytype encoder from {encoder_path}")
                with open(encoder_path, 'rb') as f:
                    self.encoder = pickle.load(f)
                logger.info("✅ QueryType model and encoder loaded successfully")
            else:
                logger.warning("⚠️  No bundled encoder found for QueryType - using external encoder")
            
            self.eval()
            
        except Exception as e:
            logger.error(f"❌ Error loading QueryType model: {e}")
            raise
    
    def get_encoder(self):
        """Safely retrieve encoder"""
        if self.encoder is None:
            raise ValueError("QueryType encoder is None")
        return self.encoder


# ========== TrainingLogger Class ==========

class TrainingLogger:
    """Enhanced training logger with time estimates"""
    
    def __init__(self, logger, total_epochs, total_batches):
        self.logger = logger
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.start_time = None
        self.epoch_times = []
        
    def start_training(self, model_name="Model"):
        """Mark training start"""
        self.start_time = time.time()
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 {model_name} TRAINING STARTED")
        self.logger.info(f"📊 Total Epochs: {self.total_epochs}")
        self.logger.info(f"📦 Total Batches per Epoch: {self.total_batches}")
        self.logger.info(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
        
    def start_epoch(self, epoch):
        """Mark epoch start"""
        self.epoch_start = time.time()
        self.logger.info("")
        self.logger.info(f"📈 EPOCH {epoch + 1}/{self.total_epochs}")
        self.logger.info("-" * 80)
        
    def log_batch(self, epoch, batch_idx, loss, **extra_losses):
        """Log batch progress"""
        if batch_idx % 10 == 0:  # Log every 10 batches
            progress = (batch_idx / self.total_batches) * 100
            loss_str = f"Loss: {loss:.4f}"
            
            # Add extra losses if provided
            for name, value in extra_losses.items():
                loss_str += f" | {name}: {value:.4f}"
            
            self.logger.info(
                f"  Batch {batch_idx}/{self.total_batches} ({progress:.1f}%) | {loss_str}"
            )
    
    def end_epoch(self, epoch, avg_loss):
        """Mark epoch end and estimate remaining time"""
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
        # Calculate ETA
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.total_epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta = timedelta(seconds=int(eta_seconds))
        
        self.logger.info("-" * 80)
        self.logger.info(
            f"✓ Epoch {epoch + 1} complete | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time:.2f}s | "
            f"ETA: {eta}"
        )
    
    def end_training(self):
        """Mark training end"""
        total_time = time.time() - self.start_time
        self.logger.info("=" * 80)
        self.logger.info(f"🎉 TRAINING COMPLETE")
        self.logger.info(f"⏱️  Total Time: {timedelta(seconds=int(total_time))}")
        self.logger.info("=" * 80)


# ========== MultiTaskBertModel Class ==========

class MultiTaskBertModel:
    """
    3-Level Hierarchical Email Classification System
    
    Features:
    - Level 1: Email → MasterDepartment + Sentiment
    - Level 2: Email + MasterDepartment → Department
    - Level 3: Email + Department → QueryType
    - Bundled encoders (risk-free deployment)
    - Comprehensive logging
    - Automatic inference routing
    """
    
    def __init__(self, config, logger):
        self.logger = logger
        self.config = config
        self.model_dir = config.get('MODEL_DIR', './results/bert-classification-sentiment')
        self.baseline_model_name = config.get('BASELINE_MODEL_NAME', 'baseline-model')
        self.finetuned_model_name = config.get('FINETUNED_MODEL_NAME', 'finetuned_multitask_model')
        
        # Initialize models as None
        self.model = None
        self.department_model = None  # Level 2 model
        self.querytype_model = None   # Level 3 model (formerly subquery_model)
        self.tokenizer = None
        
        # Will be injected from model.py
        self.label_interface = None
        self.processed_label_encoders = None
        self.preload_task_data = None
        
        self.logger.info("✓ MultiTaskBertModel instance created")
    
    def reload_model(self):
        """
        Load all three models:
        - Level 1: MasterDepartment + Sentiment
        - Level 2: Department
        - Level 3: QueryType
        """
        self.logger.info("🔄 Reloading all models...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"🖥️  Using device: {device}")
        
        # ========== LOAD TOKENIZER ==========
        from transformers import AutoTokenizer
        
        tokenizer_path = os.path.join(self.model_dir, self.finetuned_model_name)
        if not os.path.exists(tokenizer_path):
            tokenizer_path = self.baseline_model_name
            
        self.logger.info(f"📂 Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # ========== LOAD LEVEL 1 MODEL (MasterDepartment + Sentiment) ==========
        finetuned_path = os.path.join(self.model_dir, self.finetuned_model_name)
        
        if os.path.exists(finetuned_path):
            self.logger.info(f"📂 Loading Level 1 model from: {finetuned_path}")
            
            # Load encoders first to get label counts
            encoders_path = os.path.join(finetuned_path, 'label_encoders.pkl')
            with open(encoders_path, 'rb') as f:
                encoders = pickle.load(f)
            
            masterdept_encoder = encoders.get('masterdepartment')
            sentiment_encoder = encoders.get('sentiment')
            
            num_masterdept = len(masterdept_encoder.classes_) if masterdept_encoder else 100
            num_sentiment = len(sentiment_encoder.classes_) if sentiment_encoder else 2
            
            # Initialize model with correct dimensions
            self.model = MultiTaskNNModel(
                self.baseline_model_name,
                classificationlabel_length=num_masterdept,
                sentimentlabel_length=num_sentiment
            )
            
            # Load the model
            self.model.LoadModel(finetuned_path)
            self.model.to(device)
            self.model.eval()
            
            self.logger.info(f"✅ Level 1 model loaded | MasterDepartments: {num_masterdept} | Sentiments: {num_sentiment}")
        else:
            self.logger.warning(f"⚠️  Level 1 model not found at {finetuned_path}")
        
        # ========== LOAD LEVEL 2 MODEL (Department) ==========
        department_path = os.path.join(self.model_dir, "department_model")
        
        if os.path.exists(department_path):
            self.logger.info(f"📂 Loading Level 2 (Department) model from: {department_path}")
            
            # Load encoder to get label count
            dept_encoder_path = os.path.join(department_path, 'department_encoder.pkl')
            if os.path.exists(dept_encoder_path):
                with open(dept_encoder_path, 'rb') as f:
                    dept_encoder = pickle.load(f)
                num_departments = len(dept_encoder.classes_)
            else:
                num_departments = 100  # default
            
            self.department_model = DepartmentNNModel(
                self.baseline_model_name,
                num_labels=num_departments
            )
            self.department_model.load(department_path)
            self.department_model.to(device)
            self.department_model.eval()
            
            self.logger.info(f"✅ Level 2 model loaded | Departments: {num_departments}")
        else:
            self.logger.warning(f"⚠️  Level 2 (Department) model not found at {department_path}")
        
        # ========== LOAD LEVEL 3 MODEL (QueryType) ==========
        querytype_path = os.path.join(self.model_dir, "querytype_model")
        
        if os.path.exists(querytype_path):
            self.logger.info(f"📂 Loading Level 3 (QueryType) model from: {querytype_path}")
            
            # Load encoder to get label count
            qt_encoder_path = os.path.join(querytype_path, 'querytype_encoder.pkl')
            if os.path.exists(qt_encoder_path):
                with open(qt_encoder_path, 'rb') as f:
                    qt_encoder = pickle.load(f)
                num_querytypes = len(qt_encoder.classes_)
            else:
                num_querytypes = 100  # default
            
            self.querytype_model = SubQueryNNModel(
                self.baseline_model_name,
                num_labels=num_querytypes
            )
            self.querytype_model.load(querytype_path)
            self.querytype_model.to(device)
            self.querytype_model.eval()
            
            self.logger.info(f"✅ Level 3 model loaded | QueryTypes: {num_querytypes}")
        else:
            self.logger.warning(f"⚠️  Level 3 (QueryType) model not found at {querytype_path}")
        
        self.logger.info("🎉 All models reloaded successfully")
    
    # ========== HELPER FUNCTIONS ==========
    
    def _get_device(self):
        """Get available device (CUDA or CPU)"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _tokenize_and_pad(self, texts, tokenizer, max_len):
        """Tokenize and pad text sequences"""
        input_ids = []
        for text in texts:
            encoded = tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=max_len,
                truncation=True
            )
            input_ids.append(encoded)
        
        # Pad sequences
        input_ids = pad_sequences(
            input_ids,
            maxlen=max_len,
            dtype="long",
            value=0,
            truncating="post",
            padding="post"
        )
        
        # Create attention masks
        attention_masks = []
        for seq in input_ids:
            mask = [int(token_id > 0) for token_id in seq]
            attention_masks.append(mask)
        
        return input_ids, attention_masks
    
    def _create_dataloader(self, input_ids, attention_masks, labels=None, batch_size=16, shuffle=True):
        """Create DataLoader from tensors"""
        inputs = torch.tensor(input_ids)
        masks = torch.tensor(attention_masks)
        
        if labels is not None:
            labels_tensor = torch.tensor(labels)
            dataset = TensorDataset(inputs, masks, labels_tensor)
        else:
            dataset = TensorDataset(inputs, masks)
        
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        
        return dataloader
    
    def _decode_predictions(self, predictions, task_name):
        """Decode numerical predictions to labels"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Get encoder based on task
        if task_name == 'masterdepartment':
            encoder = self.model.get_encoder('classification')
        elif task_name == 'sentiment':
            encoder = self.model.get_encoder('sentiment')
        elif task_name == 'department':
            if self.department_model is None:
                raise ValueError("Department model not loaded")
            encoder = self.department_model.get_encoder()
        elif task_name == 'querytype':
            if self.querytype_model is None:
                raise ValueError("QueryType model not loaded")
            encoder = self.querytype_model.get_encoder()
        else:
            raise ValueError(f"Unknown task: {task_name}")
        
        # Convert to numpy if tensor
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        
        # Decode
        decoded = encoder.inverse_transform(predictions)
        return decoded.tolist() if hasattr(decoded, 'tolist') else list(decoded)
    
    # ========== DATABASE FUNCTIONS ==========
    
    def init_metrics_db(self):
        """Initialize SQLite database for metrics"""
        import sqlite3
        
        db_path = os.path.join(self.model_dir, "metrics.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level INTEGER,
                train_samples INTEGER,
                test_samples INTEGER
            )
        """)
        
        # Create metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                task TEXT,
                metric TEXT,
                value REAL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"✓ Metrics database initialized at {db_path}")
    
    def save_metrics_to_sqlite(self, level, masterdept_accuracy, masterdept_f1, sent_accuracy, sent_f1, train_count, test_count):
        """Save metrics to SQLite database"""
        import sqlite3
        
        db_path = os.path.join(self.model_dir, "metrics.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Insert run
        cursor.execute("""
            INSERT INTO runs (timestamp, level, train_samples, test_samples)
            VALUES (?, ?, ?, ?)
        """, (datetime.now().isoformat(), level, train_count, test_count))
        
        run_id = cursor.lastrowid
        
        # Insert Level 1 metrics
        metrics = [
            (run_id, 'masterdepartment', 'accuracy_overall', masterdept_accuracy),
            (run_id, 'masterdepartment', 'f1_weighted', masterdept_f1),
            (run_id, 'sentiment', 'accuracy_overall', sent_accuracy),
            (run_id, 'sentiment', 'f1_weighted', sent_f1)
        ]
        
        cursor.executemany("""
            INSERT INTO metrics (run_id, task, metric, value)
            VALUES (?, ?, ?, ?)
        """, metrics)
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"💾 Metrics saved to database (run_id: {run_id})")
        return run_id
    
    def save_full_classification_report_to_sqlite(self, run_id, task, y_true, y_pred):
        """Save detailed classification report to database"""
        import sqlite3
        from sklearn.metrics import classification_report
        
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        db_path = os.path.join(self.model_dir, "metrics.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        metrics_to_save = []
        
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if metric_name in ['precision', 'recall', 'f1-score']:
                        metric_key = f"{label}_{metric_name.replace('-', '_')}"
                        metrics_to_save.append((run_id, task, metric_key, value))
        
        cursor.executemany("""
            INSERT INTO metrics (run_id, task, metric, value)
            VALUES (?, ?, ?, ?)
        """, metrics_to_save)
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"💾 Detailed {task} report saved to database")
    
    def get_latest_metrics_from_db(self):
        """Retrieve latest metrics from database"""
        import sqlite3
        
        db_path = os.path.join(self.model_dir, "metrics.db")
        
        if not os.path.exists(db_path):
            return {"error": "No metrics database found"}
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get latest run
        cursor.execute("SELECT MAX(run_id) FROM runs")
        latest_run = cursor.fetchone()[0]
        
        if latest_run is None:
            conn.close()
            return {"error": "No training runs found"}
        
        # Get metrics
        cursor.execute("""
            SELECT task, metric, value
            FROM metrics
            WHERE run_id = ?
        """, (latest_run,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return {
            "run_id": latest_run,
            "metrics": [
                {"task": task, "metric": metric, "value": value}
                for task, metric, value in rows
            ]
        }
    
    # ========== TRAINING PIPELINE ==========
    
    def fit(self, event, data, tasks, **kwargs):
        """
        3-Level Hierarchical Training Pipeline
        
        Level 1: Email → MasterDepartment + Sentiment
        Level 2: Email + MasterDepartment → Department
        Level 3: Email + Department → QueryType
        """
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 3-LEVEL HIERARCHICAL TRAINING INITIATED")
        self.logger.info("=" * 80)
        
        # Initialize database
        self.init_metrics_db()
        
        # ========== PREPARE DATA ==========
        self.logger.info("📊 Preparing training data...")
        
        train_data = []
        for task in tasks:
            text = self.preload_task_data(task, task['data']['html'])
            
            annotations = task.get('annotations', [])
            if not annotations:
                continue
            
            annotation = annotations[0]
            result = annotation.get('result', [])
            
            # Extract all three levels
            masterdepartment = None
            department = None
            querytype = None
            sentiment = None
            
            for item in result:
                value = item.get('value', {})
                taxonomy = value.get('taxonomy', [[]])
                
                if not taxonomy or not taxonomy[0]:
                    continue
                
                from_name = item.get('from_name', '')
                
                # Detect which field this is
                if 'masterdepartment' in from_name.lower():
                    # MasterDepartment > Department > QueryType
                    full_path = taxonomy[0]
                    if len(full_path) >= 1:
                        masterdepartment = full_path[0]
                    if len(full_path) >= 2:
                        department = f"{full_path[0]} > {full_path[1]}"
                    if len(full_path) >= 3:
                        querytype = f"{full_path[0]} > {full_path[1]} > {full_path[2]}"
                        
                elif 'sentiment' in from_name.lower():
                    sentiment = taxonomy[0][0] if taxonomy[0] else None
            
            if masterdepartment and sentiment:
                train_data.append({
                    'text': text,
                    'masterdepartment': masterdepartment,
                    'department': department,
                    'querytype': querytype,
                    'sentiment': sentiment
                })
        
        if len(train_data) == 0:
            self.logger.error("❌ No valid training data found")
            return
        
        self.logger.info(f"✓ Prepared {len(train_data)} training samples")
        
        # Convert to DataFrame
        df = pd.DataFrame(train_data)
        
        # Split train/test
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['masterdepartment'])
        
        self.logger.info(f"📊 Train: {len(train_df)} | Test: {len(test_df)}")
        
        # ========== TRAIN LEVEL 1: MasterDepartment + Sentiment ==========
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🎯 LEVEL 1 TRAINING: MasterDepartment + Sentiment")
        self.logger.info("=" * 80)
        
        self._train_level1(train_df, test_df)
        
        # ========== TRAIN LEVEL 2: Department ==========
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🎯 LEVEL 2 TRAINING: Department")
        self.logger.info("=" * 80)
        
        # Filter samples with department labels
        dept_train_df = train_df[train_df['department'].notna()].copy()
        dept_test_df = test_df[test_df['department'].notna()].copy()
        
        if len(dept_train_df) > 0:
            self._train_level2(dept_train_df, dept_test_df)
        else:
            self.logger.warning("⚠️  No Department labels found - skipping Level 2 training")
        
        # ========== TRAIN LEVEL 3: QueryType ==========
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🎯 LEVEL 3 TRAINING: QueryType")
        self.logger.info("=" * 80)
        
        # Filter samples with querytype labels
        qt_train_df = train_df[train_df['querytype'].notna()].copy()
        qt_test_df = test_df[test_df['querytype'].notna()].copy()
        
        if len(qt_train_df) > 0:
            self._train_level3(qt_train_df, qt_test_df)
        else:
            self.logger.warning("⚠️  No QueryType labels found - skipping Level 3 training")
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🎉 3-LEVEL HIERARCHICAL TRAINING COMPLETE")
        self.logger.info("=" * 80)
    
    def _train_level1(self, train_df, test_df):
        """Train Level 1: MasterDepartment + Sentiment"""
        
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels
        masterdept_encoder = LabelEncoder()
        sentiment_encoder = LabelEncoder()
        
        train_masterdept_encoded = masterdept_encoder.fit_transform(train_df['masterdepartment'])
        train_sentiment_encoded = sentiment_encoder.fit_transform(train_df['sentiment'])
        
        test_masterdept_encoded = masterdept_encoder.transform(test_df['masterdepartment'])
        test_sentiment_encoded = sentiment_encoder.transform(test_df['sentiment'])
        
        self.logger.info(f"📊 MasterDepartment classes: {len(masterdept_encoder.classes_)}")
        self.logger.info(f"📊 Sentiment classes: {len(sentiment_encoder.classes_)}")
        
        # Initialize model
        num_masterdept = len(masterdept_encoder.classes_)
        num_sentiment = len(sentiment_encoder.classes_)
        
        device = self._get_device()
        
        model = MultiTaskNNModel(
            self.baseline_model_name,
            classificationlabel_length=num_masterdept,
            sentimentlabel_length=num_sentiment
        )
        model.to(device)
        
        # Tokenize
        tokenizer = self.tokenizer
        MAX_LEN = 256
        
        train_inputs, train_masks = self._tokenize_and_pad(train_df['text'].tolist(), tokenizer, MAX_LEN)
        test_inputs, test_masks = self._tokenize_and_pad(test_df['text'].tolist(), tokenizer, MAX_LEN)
        
        # Create DataLoaders
        train_dataloader = self._create_dataloader(
            train_inputs, train_masks,
            labels=list(zip(train_masterdept_encoded, train_sentiment_encoded)),
            batch_size=16,
            shuffle=True
        )
        
        # Training setup
        NUM_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 10))
        LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-5))
        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
        
        # Loss functions
        criterion_masterdept = nn.CrossEntropyLoss()
        criterion_sentiment = nn.CrossEntropyLoss()
        
        # Training logger
        training_logger = TrainingLogger(self.logger, NUM_EPOCHS, len(train_dataloader))
        training_logger.start_training("Level 1 (MasterDepartment + Sentiment)")
        
        # Training loop
        for epoch in range(NUM_EPOCHS):
            training_logger.start_epoch(epoch)
            
            model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                input_ids, attention_mask, labels_batch = [t.to(device) for t in batch]
                
                # Unpack labels
                masterdept_labels = labels_batch[:, 0].long()
                sentiment_labels = labels_batch[:, 1].long()
                
                optimizer.zero_grad()
                
                # Forward pass
                masterdept_logits, sentiment_logits, _, _ = model(input_ids, attention_mask)
                
                # Calculate losses
                loss_masterdept = criterion_masterdept(masterdept_logits, masterdept_labels)
                loss_sentiment = criterion_sentiment(sentiment_logits, sentiment_labels)
                
                total_loss_batch = loss_masterdept + loss_sentiment
                
                # Backward pass
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                
                training_logger.log_batch(
                    epoch, batch_idx, total_loss_batch.item(),
                    MasterDept=loss_masterdept.item(),
                    Sentiment=loss_sentiment.item()
                )
            
            avg_loss = total_loss / len(train_dataloader)
            training_logger.end_epoch(epoch, avg_loss)
        
        training_logger.end_training()
        
        # ========== EVALUATION ==========
        self.logger.info("")
        self.logger.info("📊 Evaluating Level 1 model...")
        
        model.eval()
        
        test_dataloader = self._create_dataloader(
            test_inputs, test_masks,
            batch_size=16,
            shuffle=False
        )
        
        masterdept_preds = []
        sentiment_preds = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, attention_mask = [t.to(device) for t in batch]
                
                masterdept_logits, sentiment_logits, _, _ = model(input_ids, attention_mask)
                
                masterdept_preds.extend(torch.argmax(masterdept_logits, dim=1).cpu().numpy())
                sentiment_preds.extend(torch.argmax(sentiment_logits, dim=1).cpu().numpy())
        
        # Calculate metrics
        masterdept_accuracy = accuracy_score(test_masterdept_encoded, masterdept_preds)
        masterdept_f1 = f1_score(test_masterdept_encoded, masterdept_preds, average='weighted', zero_division=0)
        
        sentiment_accuracy = accuracy_score(test_sentiment_encoded, sentiment_preds)
        sentiment_f1 = f1_score(test_sentiment_encoded, sentiment_preds, average='weighted', zero_division=0)
        
        self.logger.info(f"✅ MasterDepartment - Accuracy: {masterdept_accuracy:.4f} | F1: {masterdept_f1:.4f}")
        self.logger.info(f"✅ Sentiment - Accuracy: {sentiment_accuracy:.4f} | F1: {sentiment_f1:.4f}")
        
        # Save model
        model.set_encoders(masterdept_encoder, sentiment_encoder)
        
        save_path = os.path.join(self.model_dir, self.finetuned_model_name)
        model.SaveModel(save_path)
        
        # Save tokenizer
        tokenizer.save_pretrained(save_path)
        
        self.logger.info(f"💾 Level 1 model saved to {save_path}")
        
        # Save metrics
        decoded_masterdept_true = masterdept_encoder.inverse_transform(test_masterdept_encoded)
        decoded_masterdept_pred = masterdept_encoder.inverse_transform(masterdept_preds)
        decoded_sent_true = sentiment_encoder.inverse_transform(test_sentiment_encoded)
        decoded_sent_pred = sentiment_encoder.inverse_transform(sentiment_preds)
        
        run_id = self.save_metrics_to_sqlite(
            1, masterdept_accuracy, masterdept_f1,
            sentiment_accuracy, sentiment_f1,
            len(train_df), len(test_df)
        )
        
        self.save_full_classification_report_to_sqlite(run_id, "masterdepartment", decoded_masterdept_true, decoded_masterdept_pred)
        self.save_full_classification_report_to_sqlite(run_id, "sentiment", decoded_sent_true, decoded_sent_pred)
    
    def _train_level2(self, train_df, test_df):
        """Train Level 2: Department (conditioned on MasterDepartment)"""
        
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels
        dept_encoder = LabelEncoder()
        
        train_dept_encoded = dept_encoder.fit_transform(train_df['department'])
        test_dept_encoded = dept_encoder.transform(test_df['department'])
        
        self.logger.info(f"📊 Department classes: {len(dept_encoder.classes_)}")
        
        # Initialize model
        num_departments = len(dept_encoder.classes_)
        device = self._get_device()
        
        model = DepartmentNNModel(
            self.baseline_model_name,
            num_labels=num_departments
        )
        model.to(device)
        
        # Prepare conditional input: "MasterDepartment: X Email: Y"
        train_texts_conditional = [
            f"MasterDepartment: {row['masterdepartment']} Email: {row['text']}"
            for _, row in train_df.iterrows()
        ]
        
        test_texts_conditional = [
            f"MasterDepartment: {row['masterdepartment']} Email: {row['text']}"
            for _, row in test_df.iterrows()
        ]
        
        # Tokenize
        tokenizer = self.tokenizer
        MAX_LEN = 256
        
        train_inputs, train_masks = self._tokenize_and_pad(train_texts_conditional, tokenizer, MAX_LEN)
        test_inputs, test_masks = self._tokenize_and_pad(test_texts_conditional, tokenizer, MAX_LEN)
        
        # Create DataLoaders
        train_dataloader = self._create_dataloader(
            train_inputs, train_masks,
            labels=train_dept_encoded,
            batch_size=16,
            shuffle=True
        )
        
        # Training setup
        NUM_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 10))
        LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-5))
        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
        criterion = nn.CrossEntropyLoss()
        
        # Training logger
        training_logger = TrainingLogger(self.logger, NUM_EPOCHS, len(train_dataloader))
        training_logger.start_training("Level 2 (Department)")
        
        # Training loop
        for epoch in range(NUM_EPOCHS):
            training_logger.start_epoch(epoch)
            
            model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                input_ids, attention_mask, labels_batch = [t.to(device) for t in batch]
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = model(input_ids, attention_mask)
                
                # Calculate loss
                loss = criterion(logits, labels_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                training_logger.log_batch(epoch, batch_idx, loss.item())
            
            avg_loss = total_loss / len(train_dataloader)
            training_logger.end_epoch(epoch, avg_loss)
        
        training_logger.end_training()
        
        # ========== EVALUATION ==========
        self.logger.info("")
        self.logger.info("📊 Evaluating Level 2 model...")
        
        model.eval()
        
        test_dataloader = self._create_dataloader(
            test_inputs, test_masks,
            batch_size=16,
            shuffle=False
        )
        
        dept_preds = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, attention_mask = [t.to(device) for t in batch]
                
                logits = model(input_ids, attention_mask)
                dept_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        
        # Calculate metrics
        dept_accuracy = accuracy_score(test_dept_encoded, dept_preds)
        dept_f1 = f1_score(test_dept_encoded, dept_preds, average='weighted', zero_division=0)
        
        self.logger.info(f"✅ Department - Accuracy: {dept_accuracy:.4f} | F1: {dept_f1:.4f}")
        
        # Save model
        model.set_encoder(dept_encoder)
        
        save_path = os.path.join(self.model_dir, "department_model")
        model.save(save_path)
        
        self.logger.info(f"💾 Level 2 model saved to {save_path}")
        
        # Save metrics
        decoded_dept_true = dept_encoder.inverse_transform(test_dept_encoded)
        decoded_dept_pred = dept_encoder.inverse_transform(dept_preds)
        
        run_id = self.save_metrics_to_sqlite(
            2, dept_accuracy, dept_f1, 0, 0,
            len(train_df), len(test_df)
        )
        
        self.save_full_classification_report_to_sqlite(run_id, "department", decoded_dept_true, decoded_dept_pred)
    
    def _train_level3(self, train_df, test_df):
        """Train Level 3: QueryType (conditioned on Department)"""
        
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels
        qt_encoder = LabelEncoder()
        
        train_qt_encoded = qt_encoder.fit_transform(train_df['querytype'])
        test_qt_encoded = qt_encoder.transform(test_df['querytype'])
        
        self.logger.info(f"📊 QueryType classes: {len(qt_encoder.classes_)}")
        
        # Initialize model
        num_querytypes = len(qt_encoder.classes_)
        device = self._get_device()
        
        model = SubQueryNNModel(
            self.baseline_model_name,
            num_labels=num_querytypes
        )
        model.to(device)
        
        # Prepare conditional input: "Department: X Email: Y"
        train_texts_conditional = [
            f"Department: {row['department']} Email: {row['text']}"
            for _, row in train_df.iterrows()
        ]
        
        test_texts_conditional = [
            f"Department: {row['department']} Email: {row['text']}"
            for _, row in test_df.iterrows()
        ]
        
        # Tokenize
        tokenizer = self.tokenizer
        MAX_LEN = 256
        
        train_inputs, train_masks = self._tokenize_and_pad(train_texts_conditional, tokenizer, MAX_LEN)
        test_inputs, test_masks = self._tokenize_and_pad(test_texts_conditional, tokenizer, MAX_LEN)
        
        # Create DataLoaders
        train_dataloader = self._create_dataloader(
            train_inputs, train_masks,
            labels=train_qt_encoded,
            batch_size=16,
            shuffle=True
        )
        
        # Training setup
        NUM_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 10))
        LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-5))
        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
        criterion = nn.CrossEntropyLoss()
        
        # Training logger
        training_logger = TrainingLogger(self.logger, NUM_EPOCHS, len(train_dataloader))
        training_logger.start_training("Level 3 (QueryType)")
        
        # Training loop
        for epoch in range(NUM_EPOCHS):
            training_logger.start_epoch(epoch)
            
            model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                input_ids, attention_mask, labels_batch = [t.to(device) for t in batch]
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = model(input_ids, attention_mask)
                
                # Calculate loss
                loss = criterion(logits, labels_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                training_logger.log_batch(epoch, batch_idx, loss.item())
            
            avg_loss = total_loss / len(train_dataloader)
            training_logger.end_epoch(epoch, avg_loss)
        
        training_logger.end_training()
        
        # ========== EVALUATION ==========
        self.logger.info("")
        self.logger.info("📊 Evaluating Level 3 model...")
        
        model.eval()
        
        test_dataloader = self._create_dataloader(
            test_inputs, test_masks,
            batch_size=16,
            shuffle=False
        )
        
        qt_preds = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, attention_mask = [t.to(device) for t in batch]
                
                logits = model(input_ids, attention_mask)
                qt_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        
        # Calculate metrics
        qt_accuracy = accuracy_score(test_qt_encoded, qt_preds)
        qt_f1 = f1_score(test_qt_encoded, qt_preds, average='weighted', zero_division=0)
        
        self.logger.info(f"✅ QueryType - Accuracy: {qt_accuracy:.4f} | F1: {qt_f1:.4f}")
        
        # Save model
        model.set_encoder(qt_encoder)
        
        save_path = os.path.join(self.model_dir, "querytype_model")
        model.save(save_path)
        
        self.logger.info(f"💾 Level 3 model saved to {save_path}")
        
        # Save metrics
        decoded_qt_true = qt_encoder.inverse_transform(test_qt_encoded)
        decoded_qt_pred = qt_encoder.inverse_transform(qt_preds)
        
        run_id = self.save_metrics_to_sqlite(
            3, qt_accuracy, qt_f1, 0, 0,
            len(train_df), len(test_df)
        )
        
        self.save_full_classification_report_to_sqlite(run_id, "querytype", decoded_qt_true, decoded_qt_pred)

    def fit_external(self, event, data, tasks, **kwargs):
        """External training wrapper"""
        self.logger.info("🚀 External training initiated")
        return self.fit(event, data, tasks, **kwargs)

    def predict(self, tasks: List[Dict], texts: str, context: Optional[Dict] = None, **kwargs):
        """
        3-LEVEL HIERARCHICAL INFERENCE PIPELINE
        
        Just pass email text, automatically routes through:
        1. Level 1: Email → MasterDepartment + Sentiment
        2. Level 2: Email + predicted MasterDepartment → Department
        3. Level 3: Email + predicted Department → QueryType
        
        Returns all predictions in a single result.
        """
        
        self.logger.info(f"🔍 Running inference on {len(texts)} email(s)...")

        def getMasterDepartmentAttrName(attrs):
            return attrs == "masterdepartment"

        def getDepartmentAttrName(attrs):
            return attrs == "department"

        def getQueryTypeAttrName(attrs):
            return attrs == "querytype"

        def getSentimentAttrName(attrs):
            return attrs == 'sentiment'

        from_name_masterdepartment, to_name_masterdepartment, _ = \
            self.label_interface.get_first_tag_occurence(
                'Taxonomy', 'HyperText', getMasterDepartmentAttrName
            )

        from_name_department, to_name_department, _ = \
            self.label_interface.get_first_tag_occurence(
                "Taxonomy", "HyperText", getDepartmentAttrName
            )

        from_name_querytype, to_name_querytype, _ = \
            self.label_interface.get_first_tag_occurence(
                "Taxonomy", "HyperText", getQueryTypeAttrName
            )

        from_name_sentiment, to_name_sentiment, _ = \
            self.label_interface.get_first_tag_occurence(
                'Taxonomy', 'HyperText', getSentimentAttrName
            )

        tokenizer = self.tokenizer

        # Extract text from dict or use string directly
        text_list = [
            text['text'] if not isinstance(text, str) else text
            for text in texts
        ]

        MAX_LEN = 256
        batch_size = 16

        # Using helper function for tokenization and padding
        _inputs, _masks = self._tokenize_and_pad(text_list, tokenizer, MAX_LEN)

        # Using helper function for DataLoader creation
        dataloader = self._create_dataloader(
            _inputs, _masks, batch_size=batch_size, shuffle=False
        )

        device = self._get_device()  # Using helper function

        self.model.to(device)
        self.model.eval()

        if self.department_model is not None:
            self.department_model.to(device)
            self.department_model.eval()
            self.logger.info("✓ Level 2 model active - Department predictions enabled")
        else:
            self.logger.info("⚠️  Level 2 model not loaded - Department predictions skipped")

        if self.querytype_model is not None:
            self.querytype_model.to(device)
            self.querytype_model.eval()
            self.logger.info("✓ Level 3 model active - QueryType predictions enabled")
        else:
            self.logger.info("⚠️  Level 3 model not loaded - QueryType predictions skipped")

        predictions = []

        # Global index to align batch ↔ texts
        global_text_idx = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = [t.to(device) for t in batch]
    
                # ========== LEVEL 1: MasterDepartment + Sentiment ==========
                masterdept_logits, sentiment_logits, _, _ = \
                    self.model(input_ids, attention_mask)

                masterdept_probs = torch.softmax(masterdept_logits, dim=1)
                sentiment_probs = torch.softmax(sentiment_logits, dim=1)

                masterdept_preds = torch.argmax(masterdept_probs, dim=1)
                sentiment_preds = torch.argmax(sentiment_probs, dim=1)

                # Using helper function for decoding
                decoded_masterdept_preds = self._decode_predictions(
                    masterdept_preds, 'masterdepartment'
                )
                decoded_sentiment_preds = self._decode_predictions(
                    sentiment_preds, 'sentiment'
                )

                for i in range(len(masterdept_preds)):
                    text = texts[global_text_idx]
                    global_text_idx += 1

                    # ---------- MASTER DEPARTMENT ----------
                    predictions.append({
                        "from_name": from_name_masterdepartment,
                        "to_name": to_name_masterdepartment,
                        "type": "taxonomy",
                        "value": {
                            "taxonomy": [
                                [decoded_masterdept_preds[i]]
                            ],
                            "score": masterdept_probs[i][masterdept_preds[i]].item()
                        }
                    })

                    # ---------- SENTIMENT ----------
                    predictions.append({
                        "from_name": from_name_sentiment,
                        "to_name": to_name_sentiment,
                        "type": "taxonomy",
                        "value": {
                            "taxonomy": [[decoded_sentiment_preds[i]]],
                            "score": sentiment_probs[i][sentiment_preds[i]].item()
                        }
                    })

                    # ========== LEVEL 2: DEPARTMENT ==========
                    if self.department_model is not None:
                        # Use PREDICTED MasterDepartment from Level 1
                        conditional_text = (
                            f"MasterDepartment: {decoded_masterdept_preds[i]} "
                            f"Email: {text}"
                        )

                        dept_ids = tokenizer.encode(
                            conditional_text,
                            add_special_tokens=True,
                            max_length=256,
                            truncation=True
                        )

                        dept_mask = [int(t > 0) for t in dept_ids]

                        dept_logits = self.department_model(
                            torch.tensor([dept_ids]).to(device),
                            torch.tensor([dept_mask]).to(device)
                        )

                        # Unwrap if model returns tuple
                        if isinstance(dept_logits, (tuple, list)):
                            dept_logits = dept_logits[0]

                        dept_probs = torch.softmax(dept_logits, dim=1)
                        dept_idx = torch.argmax(dept_probs, dim=1).item()

                        # Using helper function for decoding
                        department_label = self._decode_predictions(
                            [dept_idx], "department"
                        )[0]

                        predictions.append({
                            "from_name": from_name_department,
                            "to_name": to_name_department,
                            "type": "taxonomy",
                            "value": {
                                "taxonomy": [department_label.split(" > ")],
                                "score": dept_probs[0][dept_idx].item()
                            }
                        })

                        # ========== LEVEL 3: QUERY TYPE ==========
                        if self.querytype_model is not None:
                            # Use PREDICTED Department from Level 2
                            conditional_text_qt = (
                                f"Department: {department_label} "
                                f"Email: {text}"
                            )

                            qt_ids = tokenizer.encode(
                                conditional_text_qt,
                                add_special_tokens=True,
                                max_length=256,
                                truncation=True
                            )

                            qt_mask = [int(t > 0) for t in qt_ids]

                            qt_logits = self.querytype_model(
                                torch.tensor([qt_ids]).to(device),
                                torch.tensor([qt_mask]).to(device)
                            )

                            # Unwrap if model returns tuple
                            if isinstance(qt_logits, (tuple, list)):
                                qt_logits = qt_logits[0]

                            qt_probs = torch.softmax(qt_logits, dim=1)
                            qt_idx = torch.argmax(qt_probs, dim=1).item()

                            # Using helper function for decoding
                            querytype_label = self._decode_predictions(
                                [qt_idx], "querytype"
                            )[0]

                            predictions.append({
                                "from_name": from_name_querytype,
                                "to_name": to_name_querytype,
                                "type": "taxonomy",
                                "value": {
                                    "taxonomy": [querytype_label.split(" > ")],
                                    "score": qt_probs[0][qt_idx].item()
                                }
                            })

        self.logger.info(f"✅ Inference complete - Generated {len(predictions)} predictions")
        return predictions
