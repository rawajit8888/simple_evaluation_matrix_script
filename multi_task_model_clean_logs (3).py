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

# QueryTypeNNModel and DepartmentNNModel are now defined in this file (below) - no separate import needed

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
        self.dropout = nn.Dropout(0.2)  # increased 0.1 → 0.2 for better generalisation
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


# ========== QueryTypeNNModel Class (Level 3 Model) ==========

class QueryTypeNNModel(nn.Module):
    """
    QueryType classification model with bundled label encoder.
    Level 3: Department → QueryType
    Uses same BERT backbone as MultiTask model for consistency.
    """
    
    def __init__(self, modelname, num_labels):
        super(QueryTypeNNModel, self).__init__()
        
        # Use SAME backbone type as MultiTaskNNModel
        self.bert = BertModel.from_pretrained(modelname)
        self.dropout = nn.Dropout(0.2)  # increased 0.1 → 0.2 for better generalisation
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


# ========== Simple Single-Task Model (Level 1) ==========

class MasterDepartmentModel(nn.Module):
    """
    Simple single-task model for MasterDepartment classification.
    Level 1: Email → MasterDepartment ONLY
    """
    
    def __init__(self, modelname, num_labels):
        super(MasterDepartmentModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(modelname)
        self.dropout = nn.Dropout(0.2)  # increased 0.1 → 0.2 for better generalisation
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            num_labels
        )
        
        # Store encoder
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
        logger.info("✓ Encoder attached to MasterDepartment model")
    
    def save(self, directorypath):
        """Save model with bundled encoder"""
        try:
            os.makedirs(directorypath, exist_ok=True)
            
            # Save BERT backbone
            logger.info(f"📦 Saving MasterDepartment BERT to {directorypath}")
            self.bert.save_pretrained(directorypath)
            
            # Save classifier head
            classifier_path = os.path.join(directorypath, "masterdepartment_classifier.pth")
            logger.info(f"💾 Saving classifier head to {classifier_path}")
            torch.save(
                self.classifier.state_dict(),
                classifier_path
            )
            
            # Save bundled encoder
            encoder_path = os.path.join(directorypath, 'masterdepartment_encoder.pkl')
            logger.info(f"💾 Saving masterdepartment encoder to {encoder_path}")
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.encoder, f)
            
            logger.info("✅ MasterDepartment model and encoder saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving MasterDepartment model: {e}")
            raise
    
    def load(self, directorypath):
        """Load model with bundled encoder"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Reload BERT
            logger.info(f"📂 Loading MasterDepartment BERT from {directorypath}")
            self.bert = BertModel.from_pretrained(directorypath)
            
            # Reload classifier head
            classifier_path = os.path.join(directorypath, "masterdepartment_classifier.pth")
            logger.info(f"📂 Loading classifier head from {classifier_path}")
            self.classifier.load_state_dict(
                torch.load(
                    classifier_path,
                    map_location=device
                )
            )
            
            # Load bundled encoder
            encoder_path = os.path.join(directorypath, 'masterdepartment_encoder.pkl')
            if os.path.exists(encoder_path):
                logger.info(f"📂 Loading bundled masterdepartment encoder from {encoder_path}")
                with open(encoder_path, 'rb') as f:
                    self.encoder = pickle.load(f)
                logger.info("✅ MasterDepartment model and encoder loaded successfully")
            else:
                logger.warning("⚠️  No bundled encoder found for MasterDepartment")
            
            self.eval()
            
        except Exception as e:
            logger.error(f"❌ Error loading MasterDepartment model: {e}")
            raise
    
    def get_encoder(self):
        """Safely retrieve encoder"""
        if self.encoder is None:
            raise ValueError("MasterDepartment encoder is None")
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
    - Level 1: Email → MasterDepartment ONLY
    - Level 2: Email + MasterDepartment → Department
    - Level 3: Email + MasterDepartment + Department → QueryType
    - Bundled encoders (risk-free deployment)
    - Comprehensive logging
    - Automatic inference routing
    """
    
    def __init__(self, config, logger):
        self.logger = logger
        self.config = config
        self.model_dir = config.get('MODEL_DIR', './results/bert-classification-sentiment3level_training')
        
        # CRITICAL FIX: Use actual local model path or valid HuggingFace model
        # Check if lsmb-base-model exists locally, otherwise use bert-base-uncased
        local_model_path = os.path.join(self.model_dir, 'lsmb-base-model')
        if os.path.exists(local_model_path):
            self.baseline_model_name = local_model_path
            self.logger.info(f"✓ Using LOCAL model: {local_model_path}")
        else:
            self.baseline_model_name = 'bert-base-uncased'
            self.logger.warning(f"⚠️  Local model not found, using HuggingFace: {self.baseline_model_name}")
        
        self.finetuned_model_name = config.get('FINETUNED_MODEL_NAME', 'masterdepartment_model')
        
        # Initialize models as None
        self.model = None              # Level 1 model (MasterDepartment ONLY)
        self.department_model = None   # Level 2 model
        self.querytype_model = None    # Level 3 model
        self.tokenizer = None
        
        # Will be injected from model.py
        self.label_interface = None
        self.processed_label_encoders = None
        self.preload_task_data = None
        
        self.logger.info("✓ MultiTaskBertModel instance created")
    
    def reload_model(self):
        """
        Load all three models:
        - Level 1: MasterDepartment
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
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            self.logger.info("✓ Tokenizer loaded from local files")
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to load tokenizer locally: {e}")
            self.logger.info(f"📥 Downloading tokenizer from: {tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # ========== LOAD LEVEL 1 MODEL (MasterDepartment ONLY) ==========
        masterdept_path = os.path.join(self.model_dir, self.finetuned_model_name)
        
        if os.path.exists(masterdept_path):
            self.logger.info(f"📂 Loading Level 1 model from: {masterdept_path}")
            
            # Load encoder to get label count
            encoder_path = os.path.join(masterdept_path, 'masterdepartment_encoder.pkl')
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    encoder = pickle.load(f)
                num_masterdept = len(encoder.classes_)
            else:
                num_masterdept = 100  # default
            
            self.model = MasterDepartmentModel(
                self.baseline_model_name,
                num_labels=num_masterdept
            )
            self.model.load(masterdept_path)
            self.model.to(device)
            self.model.eval()
            
            self.logger.info(f"✅ Level 1 model loaded | MasterDepartments: {num_masterdept}")
        else:
            self.logger.warning(f"⚠️  Level 1 model not found at {masterdept_path} - will be created during training")
        
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
            self.logger.warning(f"⚠️  Level 2 (Department) model not found at {department_path} - will be created during training")
        
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
            
            self.querytype_model = QueryTypeNNModel(
                self.baseline_model_name,
                num_labels=num_querytypes
            )
            self.querytype_model.load(querytype_path)
            self.querytype_model.to(device)
            self.querytype_model.eval()
            
            self.logger.info(f"✅ Level 3 model loaded | QueryTypes: {num_querytypes}")
        else:
            self.logger.warning(f"⚠️  Level 3 (QueryType) model not found at {querytype_path} - will be created during training")
        
        self.logger.info("🎉 Model loading complete")
    
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
        if task_name == 'masterdepartment':
            if self.model is None:
                raise ValueError("MasterDepartment model not loaded")
            encoder = self.model.get_encoder()
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
    
    def save_metrics_to_sqlite(self, level, accuracy, f1, train_count, test_count):
        """Save overall metrics to SQLite database — returns run_id for use in full report save"""
        import sqlite3

        db_path = os.path.join(self.model_dir, "metrics.db")
        conn    = sqlite3.connect(db_path)
        cursor  = conn.cursor()

        # Insert a new run row and get its ID
        cursor.execute("""
            INSERT INTO runs (timestamp, level, train_samples, test_samples)
            VALUES (?, ?, ?, ?)
        """, (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), level, train_count, test_count))

        run_id = cursor.lastrowid

        # Map level number to task name
        task_name = {1: 'masterdepartment', 2: 'department', 3: 'querytype'}.get(level, f'level{level}')

        # Save top-level accuracy and weighted-F1
        cursor.executemany("""
            INSERT INTO metrics (run_id, task, metric, value)
            VALUES (?, ?, ?, ?)
        """, [
            (run_id, task_name, 'accuracy',    accuracy),
            (run_id, task_name, 'weighted_f1', f1),
        ])

        conn.commit()
        conn.close()

        self.logger.info(f"💾 Metrics saved to DB — run_id: {run_id} | level: {level} | "
                         f"accuracy: {accuracy:.4f} | f1: {f1:.4f}")
        return run_id

    def save_full_classification_report_to_sqlite(self, run_id, task, y_true, y_pred):
        """Save per-class precision/recall/f1 from classification_report into metrics DB"""
        import sqlite3

        if run_id is None:
            self.logger.warning("⚠️  save_full_classification_report_to_sqlite: run_id is None, skipping")
            return

        try:
            report_dict = classification_report(
                y_true, y_pred,
                output_dict=True,
                zero_division=0
            )
        except Exception as e:
            self.logger.warning(f"⚠️  Could not generate classification report: {e}")
            return

        db_path = os.path.join(self.model_dir, "metrics.db")
        conn    = sqlite3.connect(db_path)
        cursor  = conn.cursor()

        metrics_to_save = []
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if metric_name in ['precision', 'recall', 'f1-score']:
                        metric_key = f"{label}__{metric_name.replace('-', '_')}"
                        metrics_to_save.append((run_id, task, metric_key, float(value)))

        if metrics_to_save:
            cursor.executemany("""
                INSERT INTO metrics (run_id, task, metric, value)
                VALUES (?, ?, ?, ?)
            """, metrics_to_save)

        conn.commit()
        conn.close()

        self.logger.info(f"💾 Full classification report saved to DB — task: {task} | "
                         f"classes saved: {len(metrics_to_save)}")
    
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
    
    # ========== DATA EXTRACTION HELPER ==========
    
    def _extract_taxonomy_from_task(self, task):
        """
        CRITICAL FIX: Extract 3-level taxonomy from Label Studio task
        
        Returns: dict with masterdepartment, department, querytype
        """
        result = {
            'masterdepartment': None,
            'department': None,
            'querytype': None,
            'text': None
        }
        
        # Extract text
        if 'data' in task and 'html' in task['data']:
            result['text'] = self.preload_task_data(task, task['data']['html'])
        elif 'data' in task and 'text' in task['data']:
            result['text'] = task['data']['text']
        
        # Extract annotations
        annotations = task.get('annotations', [])
        if not annotations:
            return None
        
        annotation = annotations[0]
        annotation_result = annotation.get('result', [])
        

        #debug

        if len(annotation_result) > 0:
            first_item=annotation_result[0]
            self.logger.debug(f"debug first annotation item :")
            self.logger.debug(f" from_name: {first_item.get('from_name', 'N/A')}")
            self.logger.debug(f" type: {first_item.get('type', 'N/A')}")
            taxonomy_val=first_item.get('value',{}).get('taxonomy',[])
            if taxonomy_val:
                self.logger.debug(f" taxonomy:{taxonomy_val}")

        for item in annotation_result:
            value = item.get('value', {})
            taxonomy = value.get('taxonomy', [[]])
            from_name = item.get('from_name', '').lower()
            
            if not taxonomy or not taxonomy[0]:
                continue
            full_path=taxonomy[0]

            if len(full_path) >= 3:
                result['masterdepartment']=full_path[0]
                result['department']=f"{full_path[0]} > {full_path[1]}"
                result['querytype']=f"{full_path[0]} > {full_path[1]} > {full_path[2]}"
                self.logger.debug(f"Extracted all 3 levels from field '{from_name}'")
                break
            
            elif len(full_path) >=2 and not result['department']:
                result['masterdepartment'] = full_path[0]
                result['department'] = f"{full_path[0]} > {full_path[1]}"
                self.logger.debug(f"Extracted level 2 from field '{from_name}'")
            
            elif len(full_path) >=1 and not result['masterdepartment']:
                result['masterdepartment'] = full_path[0]
                self.logger.debug(f"Extracted level 1 from field '{from_name}'")

        if not result['masterdepartment']:
            self.logger.warning(f"No masterdepartment found in task {task.get('id', 'unknown')}")
            return None
        
        return result
        

        
    
    def _save_evaluation_folder(self, level_name, y_true, y_pred, train_count, test_count):
        """
        Creates an evaluation folder with:
          - classification_report.txt  → full per-class precision/recall/F1
          - metrics_summary.txt        → accuracy, weighted-F1, macro-F1 at a glance
          - predictions.csv            → side-by-side true vs predicted for every test sample
        Folder: model_dir/evaluation/<level_name>_<timestamp>/
        """
        import csv

        # Create timestamped folder so reruns never overwrite old results
        eval_dir = os.path.join(
            self.model_dir, "evaluation",
            f"{level_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        os.makedirs(eval_dir, exist_ok=True)
        self.logger.info(f"📁 Evaluation folder: {eval_dir}")

        # ── 1. classification_report.txt ──────────────────────────────────────
        report_str = classification_report(y_true, y_pred, zero_division=0)
        report_path = os.path.join(eval_dir, "classification_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*70}\n")
            f.write(f"  LEVEL: {level_name.upper()}\n")
            f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Train samples: {train_count} | Test samples: {test_count}\n")
            f.write(f"{'='*70}\n\n")
            f.write(report_str)
        self.logger.info(f"💾 classification_report.txt saved")

        # ── 2. metrics_summary.txt ────────────────────────────────────────────
        from sklearn.metrics import f1_score as _f1, accuracy_score as _acc
        accuracy   = _acc(y_true, y_pred)
        macro_f1   = _f1(y_true, y_pred, average='macro',    zero_division=0)
        weighted_f1= _f1(y_true, y_pred, average='weighted', zero_division=0)

        # Per-class F1 — find worst 10 and best 10
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        class_f1s = {
            k: v['f1-score']
            for k, v in report_dict.items()
            if isinstance(v, dict) and k not in ('accuracy', 'macro avg', 'weighted avg')
        }
        sorted_f1s = sorted(class_f1s.items(), key=lambda x: x[1])
        worst10 = sorted_f1s[:10]
        best10  = sorted_f1s[-10:]

        summary_path = os.path.join(eval_dir, "metrics_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*70}\n")
            f.write(f"  METRICS SUMMARY — {level_name.upper()}\n")
            f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"  Accuracy     : {accuracy:.4f}\n")
            f.write(f"  Macro-F1     : {macro_f1:.4f}   ← use THIS to pick best epoch\n")
            f.write(f"  Weighted-F1  : {weighted_f1:.4f}\n")
            f.write(f"  Train samples: {train_count}\n")
            f.write(f"  Test samples : {test_count}\n")
            f.write(f"  Total classes: {len(class_f1s)}\n")
            f.write(f"  Classes F1=0 : {sum(1 for v in class_f1s.values() if v == 0.0)}\n\n")
            f.write(f"{'─'*70}\n")
            f.write(f"  WORST 10 CLASSES (lowest F1 — need more data or better labels)\n")
            f.write(f"{'─'*70}\n")
            for cls, score in worst10:
                f.write(f"  {score:.4f}  {cls}\n")
            f.write(f"\n{'─'*70}\n")
            f.write(f"  BEST 10 CLASSES (highest F1)\n")
            f.write(f"{'─'*70}\n")
            for cls, score in reversed(best10):
                f.write(f"  {score:.4f}  {cls}\n")
        self.logger.info(f"💾 metrics_summary.txt saved")

        # ── 3. predictions.csv ────────────────────────────────────────────────
        csv_path = os.path.join(eval_dir, "predictions.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['#', 'True Label', 'Predicted Label', 'Correct'])
            for i, (true, pred) in enumerate(zip(y_true, y_pred), 1):
                writer.writerow([i, true, pred, 'YES' if true == pred else 'NO'])
        self.logger.info(f"💾 predictions.csv saved")

        self.logger.info(
            f"✅ Evaluation folder complete → {eval_dir}\n"
            f"   Accuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f} | "
            f"Weighted-F1: {weighted_f1:.4f} | Classes F1=0: {sum(1 for v in class_f1s.values() if v == 0.0)}"
        )
        return eval_dir

    def fit(self, event, data, tasks, **kwargs):
        """
        3-Level Hierarchical Training Pipeline
        
        Level 1: Email → MasterDepartment
        Level 2: Email + MasterDepartment → Department
        Level 3: Email + MasterDepartment + Department → QueryType
        """

        # ── Save ALL training logs to a timestamped .txt file ─────────────────
        # Each training run creates its own file so you never overwrite old logs
        # e.g.  training_log_2025-02-23_17-30-00.txt
        os.makedirs(self.model_dir, exist_ok=True)
        log_filename = os.path.join(
            self.model_dir,
            f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        )
        _file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(_file_handler)
        self.logger.info(f"📝 Training log saving to: {log_filename}")
        # ──────────────────────────────────────────────────────────────────────

        self.logger.info("=" * 80)
        self.logger.info("🚀 3-LEVEL HIERARCHICAL TRAINING INITIATED")
        self.logger.info("=" * 80)
        
        # Initialize database
        self.init_metrics_db()
        
        # ========== PREPARE DATA ==========
        self.logger.info("📊 Preparing training data...")
        
        train_data = []
        for task in tasks:
            extracted = self._extract_taxonomy_from_task(task)
            if extracted and extracted['text']:
                train_data.append(extracted)
        
        if len(train_data) == 0:
            self.logger.error("❌ No valid training data found")
            self.logger.error("Please check your Label Studio annotations:")
            self.logger.error("  - Ensure tasks have 'masterdepartment' labels")
            self.logger.error("  - Check that taxonomy fields are properly configured")
            return
        
        self.logger.info(f"✓ Prepared {len(train_data)} training samples")
        
        # Log sample for debugging
        if train_data:
            sample = train_data[0]
            self.logger.info(f"📝 Sample annotation:")
            self.logger.info(f"   MasterDepartment: {sample['masterdepartment']}")
            self.logger.info(f"   Department: {sample['department']}")
            self.logger.info(f"   QueryType: {sample['querytype']}")
        
        # Log distribution
        masterdept_count = sum(1 for d in train_data if d['masterdepartment'])
        dept_count = sum(1 for d in train_data if d['department'])
        qt_count = sum(1 for d in train_data if d['querytype'])
        
        self.logger.info(f"📊 Label distribution:")
        self.logger.info(f"   Level 1 (MasterDepartment): {masterdept_count} samples")
        self.logger.info(f"   Level 2 (Department): {dept_count} samples")
        self.logger.info(f"   Level 3 (QueryType): {qt_count} samples")
        
        # Convert to DataFrame
        df = pd.DataFrame(train_data)
        
        from sklearn.model_selection import train_test_split
        
        # ========== TRAIN LEVEL 1: MasterDepartment ONLY ==========
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🎯 LEVEL 1 TRAINING: MasterDepartment ONLY")
        self.logger.info("=" * 80)

        # Check if Level 1 model already exists
        level1_model_path = os.path.join(self.model_dir, self.finetuned_model_name)
        level1_classifier_exists = os.path.exists(os.path.join(level1_model_path, 'masterdepartment_classifier.pth'))
        level1_weights_exists = os.path.exists(os.path.join(level1_model_path, 'model.safetensors'))
        level1_encoder_exists = os.path.exists(os.path.join(level1_model_path, 'masterdepartment_encoder.pkl'))

        level1_model_exists = level1_classifier_exists and level1_weights_exists and level1_encoder_exists

        if level1_model_exists:
            self.logger.info("=" * 80)
            self.logger.info("✅ LEVEL 1 MODEL ALREADY EXISTS - SKIPPING TRAINING")
            self.logger.info("=" * 80)
            self.logger.info(f"📁 Model path: {level1_model_path}")
            self.logger.info(f"   ✓ masterdepartment_classifier.pth")
            self.logger.info(f"   ✓ model.safetensors")
            self.logger.info(f"   ✓ masterdepartment_encoder.pkl")
            self.logger.info("💡 To retrain Level 1: Delete the masterdepartment_model folder")
            self.logger.info("=" * 80)
        else:
            # Smart split: Keep single-sample classes in training only
            class_counts = df['masterdepartment'].value_counts()
            single_sample_classes = class_counts[class_counts == 1].index.tolist()
            multi_sample_classes = class_counts[class_counts >= 2].index.tolist()

            self.logger.info(f"📊 Classes with 1 sample: {len(single_sample_classes)} (will be in training only)")
            self.logger.info(f"📊 Classes with 2+ samples: {len(multi_sample_classes)} (will be split)")

            # Separate single-sample data (goes to training only)
            single_sample_df = df[df['masterdepartment'].isin(single_sample_classes)]

            # Separate multi-sample data (will be split)
            multi_sample_df = df[df['masterdepartment'].isin(multi_sample_classes)]

            if len(multi_sample_df) > 0:
                # Split only the multi-sample data
                min_samples = multi_sample_df['masterdepartment'].value_counts().min()
                
                if min_samples >= 2:
                    self.logger.info(f"✓ Using stratified split for multi-sample classes")
                    try:
                        level1_train_multi, level1_test_df = train_test_split(
                            multi_sample_df, test_size=0.2, random_state=42,
                            stratify=multi_sample_df['masterdepartment']
                        )
                    except ValueError as e:
                        self.logger.warning(f"⚠️  Stratified split failed: {e}")
                        self.logger.warning("⚠️  Falling back to random split")
                        level1_train_multi, level1_test_df = train_test_split(
                            multi_sample_df, test_size=0.2, random_state=42
                        )
                else:
                    # Shouldn't happen, but handle it
                    level1_train_multi, level1_test_df = train_test_split(
                        multi_sample_df, test_size=0.2, random_state=42
                    )
                
                # Combine: all single-sample + training portion of multi-sample
                level1_train_df = pd.concat([single_sample_df, level1_train_multi], ignore_index=True)
            else:
                # All classes have only 1 sample
                level1_train_df = single_sample_df
                level1_test_df = pd.DataFrame(columns=df.columns)  # Empty test set
                self.logger.warning("⚠️  All classes have only 1 sample - no test set will be created")

            self.logger.info(f"📊 Level 1 - Train: {len(level1_train_df)} | Test: {len(level1_test_df)}")
            self._train_level1(level1_train_df, level1_test_df)

        
        # ========== TRAIN LEVEL 2: Department ==========
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🎯 LEVEL 2 TRAINING: Department")
        self.logger.info("=" * 80)

        # Filter samples with department labels FIRST
        dept_df = df[df['department'].notna()].copy()

        if len(dept_df) > 0:
            self.logger.info(f"📊 Department samples: {len(dept_df)}")
            
            # Smart split: Keep single-sample classes in training only
            dept_class_counts = dept_df['department'].value_counts()
            dept_single_sample_classes = dept_class_counts[dept_class_counts == 1].index.tolist()
            dept_multi_sample_classes = dept_class_counts[dept_class_counts >= 2].index.tolist()
            
            self.logger.info(f"📊 Departments with 1 sample: {len(dept_single_sample_classes)} (will be in training only)")
            self.logger.info(f"📊 Departments with 2+ samples: {len(dept_multi_sample_classes)} (will be split)")
            
            # Separate single-sample data
            dept_single_sample_df = dept_df[dept_df['department'].isin(dept_single_sample_classes)]
            
            # Separate multi-sample data
            dept_multi_sample_df = dept_df[dept_df['department'].isin(dept_multi_sample_classes)]
            
            if len(dept_multi_sample_df) > 0:
                # Split only the multi-sample data
                dept_min_samples = dept_multi_sample_df['department'].value_counts().min()
                
                if dept_min_samples >= 2:
                    self.logger.info(f"✓ Using stratified split for departments")
                    try:
                        dept_train_multi, dept_test_df = train_test_split(
                            dept_multi_sample_df, test_size=0.2, random_state=42,
                            stratify=dept_multi_sample_df['department']
                        )
                    except ValueError as e:
                        self.logger.warning(f"⚠️  Stratified split failed: {e}")
                        self.logger.warning("⚠️  Falling back to random split")
                        dept_train_multi, dept_test_df = train_test_split(
                            dept_multi_sample_df, test_size=0.2, random_state=42
                        )
                else:
                    dept_train_multi, dept_test_df = train_test_split(
                        dept_multi_sample_df, test_size=0.2, random_state=42
                    )
                
                # Combine: all single-sample + training portion of multi-sample
                dept_train_df = pd.concat([dept_single_sample_df, dept_train_multi], ignore_index=True)
            else:
                # All departments have only 1 sample
                dept_train_df = dept_single_sample_df
                dept_test_df = pd.DataFrame(columns=dept_df.columns)
                self.logger.warning("⚠️  All departments have only 1 sample - no test set will be created")
            
            self.logger.info(f"📊 Level 2 - Train: {len(dept_train_df)} | Test: {len(dept_test_df)}")
            self._train_level2(dept_train_df, dept_test_df)
        else:
            self.logger.warning("⚠️  No Department labels found - skipping Level 2 training")
        
        # ========== TRAIN LEVEL 3: QueryType ==========
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🎯 LEVEL 3 TRAINING: QueryType")
        self.logger.info("=" * 80)

        # Filter samples with querytype labels FIRST
        qt_df = df[df['querytype'].notna()].copy()

        if len(qt_df) > 0:
            self.logger.info(f"📊 QueryType samples: {len(qt_df)}")
            
            # Smart split: Keep single-sample classes in training only
            qt_class_counts = qt_df['querytype'].value_counts()
            qt_single_sample_classes = qt_class_counts[qt_class_counts == 1].index.tolist()
            qt_multi_sample_classes = qt_class_counts[qt_class_counts >= 2].index.tolist()
            
            self.logger.info(f"📊 QueryTypes with 1 sample: {len(qt_single_sample_classes)} (will be in training only)")
            self.logger.info(f"📊 QueryTypes with 2+ samples: {len(qt_multi_sample_classes)} (will be split)")
            
            # Separate single-sample data
            qt_single_sample_df = qt_df[qt_df['querytype'].isin(qt_single_sample_classes)]
            
            # Separate multi-sample data
            qt_multi_sample_df = qt_df[qt_df['querytype'].isin(qt_multi_sample_classes)]
            
            if len(qt_multi_sample_df) > 0:
                # Split only the multi-sample data
                qt_min_samples = qt_multi_sample_df['querytype'].value_counts().min()
                
                if qt_min_samples >= 2:
                    self.logger.info(f"✓ Using stratified split for querytypes")
                    try:
                        qt_train_multi, qt_test_df = train_test_split(
                            qt_multi_sample_df, test_size=0.2, random_state=42,
                            stratify=qt_multi_sample_df['querytype']
                        )
                    except ValueError as e:
                        self.logger.warning(f"⚠️  Stratified split failed: {e}")
                        self.logger.warning("⚠️  Falling back to random split")
                        qt_train_multi, qt_test_df = train_test_split(
                            qt_multi_sample_df, test_size=0.2, random_state=42
                        )
                else:
                    qt_train_multi, qt_test_df = train_test_split(
                        qt_multi_sample_df, test_size=0.2, random_state=42
                    )
                
                # Combine: all single-sample + training portion of multi-sample
                qt_train_df = pd.concat([qt_single_sample_df, qt_train_multi], ignore_index=True)
            else:
                # All querytypes have only 1 sample
                qt_train_df = qt_single_sample_df
                qt_test_df = pd.DataFrame(columns=qt_df.columns)
                self.logger.warning("⚠️  All querytypes have only 1 sample - no test set will be created")
            
            self.logger.info(f"📊 Level 3 - Train: {len(qt_train_df)} | Test: {len(qt_test_df)}")
            self._train_level3(qt_train_df, qt_test_df)
        else:
            self.logger.warning("⚠️  No QueryType labels found - skipping Level 3 training")
    
    def _train_level1(self, train_df, test_df):
        """Train Level 1: MasterDepartment ONLY"""
        
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils import compute_class_weight
        from transformers import get_cosine_schedule_with_warmup
        
        # Encode labels
        masterdept_encoder = LabelEncoder()
        
        train_masterdept_encoded = masterdept_encoder.fit_transform(train_df['masterdepartment'])
        test_masterdept_encoded  = masterdept_encoder.transform(test_df['masterdepartment'])
        
        self.logger.info(f"📊 MasterDepartment classes: {len(masterdept_encoder.classes_)}")
        
        # Initialize model
        num_masterdept = len(masterdept_encoder.classes_)
        device = self._get_device()
        
        model = MasterDepartmentModel(
            self.baseline_model_name,
            num_labels=num_masterdept
        )
        model.to(device)
        
        # Tokenize
        tokenizer = self.tokenizer
        MAX_LEN = 256

        # ═══════════════════════════════════════════════════════════════════
        # PROOF LOG — LEVEL 1 TRAINING INPUT
        # ═══════════════════════════════════════════════════════════════════
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("📋 LEVEL 1 TRAINING — WHAT GOES INTO THE MODEL")
        self.logger.info("=" * 70)
        self.logger.info("  ✅ INPUT  : Raw email text ONLY — no prefix, no context")
        self.logger.info("  ✅ LABEL  : 'masterdepartment' column — read directly from Label Studio annotation")
        self.logger.info("  ✅ SOURCE : Human annotator labelled this, NOT any model output")
        self.logger.info(f"  ✅ LABEL COLUMN USED : df['masterdepartment']")
        self.logger.info(f"  ✅ SAVE PATH : {os.path.join(self.model_dir, self.finetuned_model_name)}")
        self.logger.info("")
        self.logger.info("  📌 SAMPLE INPUTS GOING INTO LEVEL 1 MODEL:")
        for idx, (_, row) in enumerate(train_df.head(3).iterrows()):
            self.logger.info(f"  ┌─ Sample {idx+1} ────────────────────────────────────────────────")
            self.logger.info(f"  │  INPUT TEXT  : \"{str(row['text'])}\"")
            self.logger.info(f"  │  LABEL (GT)  : \"{row['masterdepartment']}\"  ← from Label Studio annotation")
            self.logger.info(f"  └──────────────────────────────────────────────────────────────")
        self.logger.info("=" * 70)
        self.logger.info("")
        # ═══════════════════════════════════════════════════════════════════

        train_inputs, train_masks = self._tokenize_and_pad(train_df['text'].tolist(), tokenizer, MAX_LEN)
        test_inputs,  test_masks  = self._tokenize_and_pad(test_df['text'].tolist(),  tokenizer, MAX_LEN)
        
        # Create DataLoaders
        train_dataloader = self._create_dataloader(
            train_inputs, train_masks,
            labels=train_masterdept_encoded,
            batch_size=16,
            shuffle=True
        )
        
        # ── Training setup ────────────────────────────────────────────────────
        NUM_EPOCHS    = int(os.getenv('L1_TRAIN_EPOCHS', 5))
        LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-5))
        
        # Class weights (balanced, capped at 10x) — forces model to learn rare classes
        raw_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_masterdept_encoded),
            y=train_masterdept_encoded
        )
        raw_weights  = np.clip(raw_weights, 0.1, 10.0)
        class_weights_tensor = torch.tensor(raw_weights, dtype=torch.float32).to(device)
        self.logger.info(f"⚖️  L1 class weights — min: {raw_weights.min():.3f}  max: {raw_weights.max():.3f}  "
                         f"classes at 10x cap: {(raw_weights >= 9.99).sum()}")
        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # LR warmup (100 steps) + cosine decay
        total_steps   = len(train_dataloader) * NUM_EPOCHS
        warmup_steps  = min(100, total_steps // 10)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        self.logger.info(f"📅 L1 — epochs: {NUM_EPOCHS} | warmup steps: {warmup_steps} | total steps: {total_steps}")
        
        # Checkpoint directory — one complete model saved per epoch
        checkpoint_dir = os.path.join(self.model_dir, "masterdepartment_model_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger.info(f"📂 Per-epoch checkpoints → {checkpoint_dir}")
        
        # Training logger
        training_logger = TrainingLogger(self.logger, NUM_EPOCHS, len(train_dataloader))
        training_logger.start_training("Level 1 (MasterDepartment)")

        # Track macro-F1 per epoch to auto-select best
        epoch_macro_scores = []  # list of (epoch_number, macro_f1, checkpoint_dir)
        
        # ── Training loop ─────────────────────────────────────────────────────
        for epoch in range(NUM_EPOCHS):
            training_logger.start_epoch(epoch)
            
            model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                input_ids, attention_mask, labels_batch = [t.to(device) for t in batch]
                
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss   = criterion(logits, labels_batch)
                loss.backward()
                
                # Gradient clipping — prevents instability from high class weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                training_logger.log_batch(epoch, batch_idx, loss.item())
            
            avg_loss = total_loss / len(train_dataloader)
            training_logger.end_epoch(epoch, avg_loss)
            
            # ── Per-epoch evaluation on test set ──────────────────────────────
            model.eval()
            epoch_preds = []
            with torch.no_grad():
                eval_dl = self._create_dataloader(test_inputs, test_masks, batch_size=16, shuffle=False)
                for batch in eval_dl:
                    ids, masks = [t.to(device) for t in batch]
                    epoch_preds.extend(torch.argmax(model(ids, masks), dim=1).cpu().numpy())
            
            from sklearn.metrics import f1_score as _f1
            ep_acc      = accuracy_score(test_masterdept_encoded, epoch_preds)
            ep_macro    = _f1(test_masterdept_encoded, epoch_preds, average='macro',    zero_division=0)
            ep_weighted = _f1(test_masterdept_encoded, epoch_preds, average='weighted', zero_division=0)
            
            self.logger.info(
                f"📊 [L1 Epoch {epoch+1}/{NUM_EPOCHS}] "
                f"Loss: {avg_loss:.4f} | Acc: {ep_acc:.4f} | "
                f"Macro-F1: {ep_macro:.4f} | Weighted-F1: {ep_weighted:.4f}"
            )
            
            # ── Save complete checkpoint after every epoch ─────────────────────
            epoch_ckpt_dir = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
            model.set_encoder(masterdept_encoder)
            model.save(epoch_ckpt_dir)
            tokenizer.save_pretrained(epoch_ckpt_dir)
            self.logger.info(f"💾 L1 Epoch {epoch+1} checkpoint saved → {epoch_ckpt_dir}")

            # Track score for auto-selection
            epoch_macro_scores.append((epoch + 1, ep_macro, epoch_ckpt_dir))
        
        training_logger.end_training()

        # ── Auto-select best epoch by Macro-F1 and copy to production folder ──
        best_epoch_num, best_macro, best_ckpt_dir = max(epoch_macro_scores, key=lambda x: x[1])
        self.logger.info("=" * 80)
        self.logger.info(f"🏆 L1 BEST EPOCH AUTO-SELECTED: Epoch {best_epoch_num} | Macro-F1: {best_macro:.4f}")
        self.logger.info(f"   Source : {best_ckpt_dir}")
        self.logger.info("   Epoch Summary:")
        for ep_num, ep_f1, _ in epoch_macro_scores:
            marker = " ← BEST" if ep_num == best_epoch_num else ""
            self.logger.info(f"     Epoch {ep_num}: Macro-F1 = {ep_f1:.4f}{marker}")

        save_path = os.path.join(self.model_dir, self.finetuned_model_name)
        import shutil
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        shutil.copytree(best_ckpt_dir, save_path)
        self.logger.info(f"✅ Best epoch {best_epoch_num} copied → {save_path}")
        self.logger.info("=" * 80)

        # ── Final evaluation using best epoch weights ──────────────────────────
        self.logger.info("📊 Evaluating Level 1 model (best epoch)...")
        model.load(save_path)
        model.to(device)
        model.eval()

        test_dataloader = self._create_dataloader(test_inputs, test_masks, batch_size=16, shuffle=False)
        masterdept_preds = []
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, attention_mask = [t.to(device) for t in batch]
                logits = model(input_ids, attention_mask)
                masterdept_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

        # ── Log each test email flow for Level 1 ──────────────────────────────
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("🧪 LEVEL 1 — TEST SET EVALUATION (20% split) — EMAIL BY EMAIL")
        self.logger.info("   INPUT  : Raw email text only")
        self.logger.info("   LABEL  : masterdepartment (ground truth from annotation)")
        self.logger.info("=" * 70)
        for idx, (_, row) in enumerate(test_df.iterrows()):
            true_label = row['masterdepartment']
            pred_label = masterdept_encoder.inverse_transform([masterdept_preds[idx]])[0]
            match = "✅ CORRECT" if true_label == pred_label else "❌ WRONG"
            self.logger.info(f"  Test Email [{idx+1}]")
            self.logger.info(f"    INPUT TEXT   : \"{row['text']}\"")
            self.logger.info(f"    GROUND TRUTH : \"{true_label}\"")
            self.logger.info(f"    PREDICTED    : \"{pred_label}\"")
            self.logger.info(f"    RESULT       : {match}")
            self.logger.info("")
        self.logger.info("=" * 70)

        masterdept_accuracy = accuracy_score(test_masterdept_encoded, masterdept_preds)
        masterdept_f1       = f1_score(test_masterdept_encoded, masterdept_preds, average='weighted', zero_division=0)

        self.logger.info(f"✅ MasterDepartment (best epoch {best_epoch_num}) - Accuracy: {masterdept_accuracy:.4f} | F1: {masterdept_f1:.4f}")
        
        # Save metrics to DB
        decoded_masterdept_true = masterdept_encoder.inverse_transform(test_masterdept_encoded)
        decoded_masterdept_pred = masterdept_encoder.inverse_transform(masterdept_preds)
        
        run_id = self.save_metrics_to_sqlite(
            1, masterdept_accuracy, masterdept_f1,
            len(train_df), len(test_df)
        )
        self.save_full_classification_report_to_sqlite(run_id, "masterdepartment", decoded_masterdept_true, decoded_masterdept_pred)

        # Save evaluation folder
        self._save_evaluation_folder(
            "level1_masterdepartment",
            decoded_masterdept_true, decoded_masterdept_pred,
            len(train_df), len(test_df)
        )
    
    def _train_level2(self, train_df, test_df):
        """Train Level 2: Department (conditioned on MasterDepartment)"""
        
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils import compute_class_weight
        from transformers import get_cosine_schedule_with_warmup
        
        # Encode labels
        dept_encoder = LabelEncoder()
        
        train_dept_encoded = dept_encoder.fit_transform(train_df['department'])
        test_dept_encoded  = dept_encoder.transform(test_df['department'])
        
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

        # ═══════════════════════════════════════════════════════════════════
        # PROOF LOG — LEVEL 2 TRAINING INPUT
        # ═══════════════════════════════════════════════════════════════════
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("📋 LEVEL 2 TRAINING — WHAT GOES INTO THE MODEL")
        self.logger.info("=" * 70)
        self.logger.info("  ✅ INPUT  : 'MasterDepartment: [GT]  Email: [text]'")
        self.logger.info("  ✅ LABEL  : 'department' column — read directly from Label Studio annotation")
        self.logger.info("  ✅ SOURCE OF MasterDepartment IN PREFIX:")
        self.logger.info("             row['masterdepartment']  ← comes from DataFrame column")
        self.logger.info("             which is the HUMAN ANNOTATION from Label Studio")
        self.logger.info("             ⚠️  The Level 1 trained model is NOT called here")
        self.logger.info("             ⚠️  Level 1 model output is NOT used during Level 2 training")
        self.logger.info(f"  ✅ LABEL COLUMN USED : df['department']")
        self.logger.info(f"  ✅ SAVE PATH : {os.path.join(self.model_dir, 'department_model')}")
        self.logger.info("")
        self.logger.info("  📌 SAMPLE INPUTS GOING INTO LEVEL 2 MODEL:")
        for idx, (_, row) in enumerate(train_df.head(3).iterrows()):
            combined = f"MasterDepartment: {row['masterdepartment']} Email: {row['text']}"
            self.logger.info(f"  ┌─ Sample {idx+1} ────────────────────────────────────────────────")
            self.logger.info(f"  │  PREFIX (GT annotation)  : \"MasterDepartment: {row['masterdepartment']}\"")
            self.logger.info(f"  │  EMAIL TEXT              : \"{str(row['text'])}\"")
            self.logger.info(f"  │  FULL INPUT TO MODEL     : \"{combined}\"")
            self.logger.info(f"  │  LABEL (GT annotation)   : \"{row['department']}\"")
            self.logger.info(f"  └──────────────────────────────────────────────────────────────")
        self.logger.info("=" * 70)
        self.logger.info("")
        # ═══════════════════════════════════════════════════════════════════

        # Tokenize
        tokenizer = self.tokenizer
        MAX_LEN = 256
        
        train_inputs, train_masks = self._tokenize_and_pad(train_texts_conditional, tokenizer, MAX_LEN)
        test_inputs,  test_masks  = self._tokenize_and_pad(test_texts_conditional,  tokenizer, MAX_LEN)
        
        # Create DataLoaders
        train_dataloader = self._create_dataloader(
            train_inputs, train_masks,
            labels=train_dept_encoded,
            batch_size=16,
            shuffle=True
        )
        
        # ── Training setup ────────────────────────────────────────────────────
        NUM_EPOCHS    = int(os.getenv('L2_TRAIN_EPOCHS', 6))
        LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-5))
        
        # Class weights (balanced, capped at 10x) — critical for departments with few samples
        raw_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_dept_encoded),
            y=train_dept_encoded
        )
        raw_weights  = np.clip(raw_weights, 0.1, 10.0)
        class_weights_tensor = torch.tensor(raw_weights, dtype=torch.float32).to(device)
        self.logger.info(f"⚖️  L2 class weights — min: {raw_weights.min():.3f}  max: {raw_weights.max():.3f}  "
                         f"classes at 10x cap: {(raw_weights >= 9.99).sum()}")
        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # LR warmup (200 steps) + cosine decay
        total_steps  = len(train_dataloader) * NUM_EPOCHS
        warmup_steps = min(200, total_steps // 10)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        self.logger.info(f"📅 L2 — epochs: {NUM_EPOCHS} | warmup steps: {warmup_steps} | total steps: {total_steps}")
        
        # Checkpoint directory — one complete model saved per epoch
        checkpoint_dir = os.path.join(self.model_dir, "department_model_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger.info(f"📂 Per-epoch checkpoints → {checkpoint_dir}")
        
        # Training logger
        training_logger = TrainingLogger(self.logger, NUM_EPOCHS, len(train_dataloader))
        training_logger.start_training("Level 2 (Department)")

        # Track macro-F1 per epoch to auto-select best
        epoch_macro_scores = []
        
        # ── Training loop ─────────────────────────────────────────────────────
        for epoch in range(NUM_EPOCHS):
            training_logger.start_epoch(epoch)
            
            model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                input_ids, attention_mask, labels_batch = [t.to(device) for t in batch]
                
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss   = criterion(logits, labels_batch)
                loss.backward()
                
                # Gradient clipping — prevents instability from high class weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                training_logger.log_batch(epoch, batch_idx, loss.item())
            
            avg_loss = total_loss / len(train_dataloader)
            training_logger.end_epoch(epoch, avg_loss)
            
            # ── Per-epoch evaluation on test set ──────────────────────────────
            model.eval()
            epoch_preds = []
            with torch.no_grad():
                eval_dl = self._create_dataloader(test_inputs, test_masks, batch_size=16, shuffle=False)
                for batch in eval_dl:
                    ids, masks = [t.to(device) for t in batch]
                    epoch_preds.extend(torch.argmax(model(ids, masks), dim=1).cpu().numpy())
            
            from sklearn.metrics import f1_score as _f1
            ep_acc      = accuracy_score(test_dept_encoded, epoch_preds)
            ep_macro    = _f1(test_dept_encoded, epoch_preds, average='macro',    zero_division=0)
            ep_weighted = _f1(test_dept_encoded, epoch_preds, average='weighted', zero_division=0)
            
            self.logger.info(
                f"📊 [L2 Epoch {epoch+1}/{NUM_EPOCHS}] "
                f"Loss: {avg_loss:.4f} | Acc: {ep_acc:.4f} | "
                f"Macro-F1: {ep_macro:.4f} | Weighted-F1: {ep_weighted:.4f}"
            )
            
            # ── Save complete checkpoint after every epoch ─────────────────────
            epoch_ckpt_dir = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
            model.set_encoder(dept_encoder)
            model.save(epoch_ckpt_dir)
            self.logger.info(f"💾 L2 Epoch {epoch+1} checkpoint saved → {epoch_ckpt_dir}")

            epoch_macro_scores.append((epoch + 1, ep_macro, epoch_ckpt_dir))
        
        training_logger.end_training()

        # ── Auto-select best epoch by Macro-F1 and copy to production folder ──
        best_epoch_num, best_macro, best_ckpt_dir = max(epoch_macro_scores, key=lambda x: x[1])
        self.logger.info("=" * 80)
        self.logger.info(f"🏆 L2 BEST EPOCH AUTO-SELECTED: Epoch {best_epoch_num} | Macro-F1: {best_macro:.4f}")
        self.logger.info(f"   Source : {best_ckpt_dir}")
        self.logger.info("   Epoch Summary:")
        for ep_num, ep_f1, _ in epoch_macro_scores:
            marker = " ← BEST" if ep_num == best_epoch_num else ""
            self.logger.info(f"     Epoch {ep_num}: Macro-F1 = {ep_f1:.4f}{marker}")

        save_path = os.path.join(self.model_dir, "department_model")
        import shutil
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        shutil.copytree(best_ckpt_dir, save_path)
        self.logger.info(f"✅ Best epoch {best_epoch_num} copied → {save_path}")
        self.logger.info("=" * 80)

        # ── Final evaluation using best epoch weights ──────────────────────────
        self.logger.info("📊 Evaluating Level 2 model (best epoch)...")
        model.load(save_path)
        model.to(device)
        model.eval()

        test_dataloader = self._create_dataloader(test_inputs, test_masks, batch_size=16, shuffle=False)
        dept_preds = []
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, attention_mask = [t.to(device) for t in batch]
                logits = model(input_ids, attention_mask)
                dept_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

        # ── Log each test email flow for Level 2 ──────────────────────────────
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("🧪 LEVEL 2 — TEST SET EVALUATION (20% split) — EMAIL BY EMAIL")
        self.logger.info("   INPUT  : 'MasterDepartment: [GT]  Email: [text]'")
        self.logger.info("   NOTE   : MasterDepartment prefix here is GROUND TRUTH from annotation")
        self.logger.info("            (same as how it was trained — teacher forcing applies to eval too)")
        self.logger.info("   LABEL  : department (ground truth from annotation)")
        self.logger.info("=" * 70)
        for idx, (_, row) in enumerate(test_df.iterrows()):
            true_label = row['department']
            pred_label = dept_encoder.inverse_transform([dept_preds[idx]])[0]
            full_input = test_texts_conditional[idx]
            match = "✅ CORRECT" if true_label == pred_label else "❌ WRONG"
            self.logger.info(f"  Test Email [{idx+1}]")
            self.logger.info(f"    PREFIX (GT)      : \"MasterDepartment: {row['masterdepartment']}\"")
            self.logger.info(f"    EMAIL TEXT       : \"{row['text']}\"")
            self.logger.info(f"    FULL INPUT SENT  : \"{full_input}\"")
            self.logger.info(f"    GROUND TRUTH     : \"{true_label}\"")
            self.logger.info(f"    PREDICTED        : \"{pred_label}\"")
            self.logger.info(f"    RESULT           : {match}")
            self.logger.info("")
        self.logger.info("=" * 70)

        dept_accuracy = accuracy_score(test_dept_encoded, dept_preds)
        dept_f1       = f1_score(test_dept_encoded, dept_preds, average='weighted', zero_division=0)

        self.logger.info(f"✅ Department (best epoch {best_epoch_num}) - Accuracy: {dept_accuracy:.4f} | F1: {dept_f1:.4f}")
        
        # Save metrics to DB
        decoded_dept_true = dept_encoder.inverse_transform(test_dept_encoded)
        decoded_dept_pred = dept_encoder.inverse_transform(dept_preds)
        
        run_id = self.save_metrics_to_sqlite(
            2, dept_accuracy, dept_f1,
            len(train_df), len(test_df)
        )
        self.save_full_classification_report_to_sqlite(run_id, "department", decoded_dept_true, decoded_dept_pred)

        # Save evaluation folder
        self._save_evaluation_folder(
            "level2_department",
            decoded_dept_true, decoded_dept_pred,
            len(train_df), len(test_df)
        )
    
    def _train_level3(self, train_df, test_df):
        """Train Level 3: QueryType (conditioned on MasterDepartment + Department)"""
        
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils import compute_class_weight
        from transformers import get_cosine_schedule_with_warmup
        
        # Encode labels
        qt_encoder = LabelEncoder()
        
        train_qt_encoded = qt_encoder.fit_transform(train_df['querytype'])
        test_qt_encoded  = qt_encoder.transform(test_df['querytype'])
        
        self.logger.info(f"📊 QueryType classes: {len(qt_encoder.classes_)}")
        
        # Initialize model
        num_querytypes = len(qt_encoder.classes_)
        device = self._get_device()
        
        model = QueryTypeNNModel(
            self.baseline_model_name,
            num_labels=num_querytypes
        )
        model.to(device)
        
        # Prepare conditional input: "MasterDepartment: X Department: Y Email: Z"
        train_texts_conditional = [
            f"MasterDepartment: {row['masterdepartment']} Department: {row['department']} Email: {row['text']}"
            for _, row in train_df.iterrows()
        ]
        
        test_texts_conditional = [
            f"MasterDepartment: {row['masterdepartment']} Department: {row['department']} Email: {row['text']}"
            for _, row in test_df.iterrows()
        ]

        # ═══════════════════════════════════════════════════════════════════
        # PROOF LOG — LEVEL 3 TRAINING INPUT
        # ═══════════════════════════════════════════════════════════════════
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("📋 LEVEL 3 TRAINING — WHAT GOES INTO THE MODEL")
        self.logger.info("=" * 70)
        self.logger.info("  ✅ INPUT  : 'MasterDepartment: [GT]  Department: [GT]  Email: [text]'")
        self.logger.info("  ✅ LABEL  : 'querytype' column — read directly from Label Studio annotation")
        self.logger.info("  ✅ SOURCE OF MasterDepartment IN PREFIX:")
        self.logger.info("             row['masterdepartment']  ← HUMAN ANNOTATION from Label Studio")
        self.logger.info("             ⚠️  Level 1 model is NOT called here")
        self.logger.info("  ✅ SOURCE OF Department IN PREFIX:")
        self.logger.info("             row['department']  ← HUMAN ANNOTATION from Label Studio")
        self.logger.info("             ⚠️  Level 2 model is NOT called here")
        self.logger.info("  ✅ BOTH PREFIXES come from ground truth — no model output used")
        self.logger.info(f"  ✅ LABEL COLUMN USED : df['querytype']")
        self.logger.info(f"  ✅ SAVE PATH : {os.path.join(self.model_dir, 'querytype_model')}")
        self.logger.info("")
        self.logger.info("  📌 SAMPLE INPUTS GOING INTO LEVEL 3 MODEL:")
        for idx, (_, row) in enumerate(train_df.head(3).iterrows()):
            combined = f"MasterDepartment: {row['masterdepartment']} Department: {row['department']} Email: {row['text']}"
            self.logger.info(f"  ┌─ Sample {idx+1} ────────────────────────────────────────────────")
            self.logger.info(f"  │  PREFIX-1 (GT)           : \"MasterDepartment: {row['masterdepartment']}\"")
            self.logger.info(f"  │  PREFIX-2 (GT)           : \"Department: {row['department']}\"")
            self.logger.info(f"  │  EMAIL TEXT              : \"{str(row['text'])}\"")
            self.logger.info(f"  │  FULL INPUT TO MODEL     : \"{combined}\"")
            self.logger.info(f"  │  LABEL (GT annotation)   : \"{row['querytype']}\"")
            self.logger.info(f"  └──────────────────────────────────────────────────────────────")
        self.logger.info("=" * 70)
        self.logger.info("")
        # ═══════════════════════════════════════════════════════════════════

        # Tokenize
        tokenizer = self.tokenizer
        MAX_LEN = 256
        
        train_inputs, train_masks = self._tokenize_and_pad(train_texts_conditional, tokenizer, MAX_LEN)
        test_inputs,  test_masks  = self._tokenize_and_pad(test_texts_conditional,  tokenizer, MAX_LEN)
        
        # Create DataLoaders
        train_dataloader = self._create_dataloader(
            train_inputs, train_masks,
            labels=train_qt_encoded,
            batch_size=16,
            shuffle=True
        )
        
        # ── Training setup ────────────────────────────────────────────────────
        NUM_EPOCHS    = int(os.getenv('L3_TRAIN_EPOCHS', 8))
        LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-5))
        
        # Class weights (balanced, capped at 10x) — most important for L3
        # 148 classes with 62 having <10 samples — without weights model ignores them completely
        raw_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_qt_encoded),
            y=train_qt_encoded
        )
        raw_weights  = np.clip(raw_weights, 0.1, 10.0)
        class_weights_tensor = torch.tensor(raw_weights, dtype=torch.float32).to(device)
        self.logger.info(f"⚖️  L3 class weights — min: {raw_weights.min():.3f}  max: {raw_weights.max():.3f}  "
                         f"classes at 10x cap: {(raw_weights >= 9.99).sum()}")
        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # LR warmup (500 steps) + cosine decay
        # L3 needs longer warmup — 148 classes, many rare, model needs stable early learning
        total_steps  = len(train_dataloader) * NUM_EPOCHS
        warmup_steps = min(500, total_steps // 10)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        self.logger.info(f"📅 L3 — epochs: {NUM_EPOCHS} | warmup steps: {warmup_steps} | total steps: {total_steps}")
        
        # Checkpoint directory — one complete model saved per epoch
        checkpoint_dir = os.path.join(self.model_dir, "querytype_model_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger.info(f"📂 Per-epoch checkpoints → {checkpoint_dir}")
        
        # Training logger
        training_logger = TrainingLogger(self.logger, NUM_EPOCHS, len(train_dataloader))
        training_logger.start_training("Level 3 (QueryType)")

        # Track macro-F1 per epoch to auto-select best
        epoch_macro_scores = []
        
        # ── Training loop ─────────────────────────────────────────────────────
        for epoch in range(NUM_EPOCHS):
            training_logger.start_epoch(epoch)
            
            model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                input_ids, attention_mask, labels_batch = [t.to(device) for t in batch]
                
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss   = criterion(logits, labels_batch)
                loss.backward()
                
                # Gradient clipping — prevents instability from high class weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                training_logger.log_batch(epoch, batch_idx, loss.item())
            
            avg_loss = total_loss / len(train_dataloader)
            training_logger.end_epoch(epoch, avg_loss)
            
            # ── Per-epoch evaluation on test set ──────────────────────────────
            model.eval()
            epoch_preds = []
            with torch.no_grad():
                eval_dl = self._create_dataloader(test_inputs, test_masks, batch_size=16, shuffle=False)
                for batch in eval_dl:
                    ids, masks = [t.to(device) for t in batch]
                    epoch_preds.extend(torch.argmax(model(ids, masks), dim=1).cpu().numpy())
            
            from sklearn.metrics import f1_score as _f1
            ep_acc      = accuracy_score(test_qt_encoded, epoch_preds)
            ep_macro    = _f1(test_qt_encoded, epoch_preds, average='macro',    zero_division=0)
            ep_weighted = _f1(test_qt_encoded, epoch_preds, average='weighted', zero_division=0)
            
            self.logger.info(
                f"📊 [L3 Epoch {epoch+1}/{NUM_EPOCHS}] "
                f"Loss: {avg_loss:.4f} | Acc: {ep_acc:.4f} | "
                f"Macro-F1: {ep_macro:.4f} | Weighted-F1: {ep_weighted:.4f}"
            )
            
            # ── Save complete checkpoint after every epoch ─────────────────────
            epoch_ckpt_dir = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
            model.set_encoder(qt_encoder)
            model.save(epoch_ckpt_dir)
            self.logger.info(f"💾 L3 Epoch {epoch+1} checkpoint saved → {epoch_ckpt_dir}")

            epoch_macro_scores.append((epoch + 1, ep_macro, epoch_ckpt_dir))
        
        training_logger.end_training()

        # ── Auto-select best epoch by Macro-F1 and copy to production folder ──
        best_epoch_num, best_macro, best_ckpt_dir = max(epoch_macro_scores, key=lambda x: x[1])
        self.logger.info("=" * 80)
        self.logger.info(f"🏆 L3 BEST EPOCH AUTO-SELECTED: Epoch {best_epoch_num} | Macro-F1: {best_macro:.4f}")
        self.logger.info(f"   Source : {best_ckpt_dir}")
        self.logger.info("   Epoch Summary:")
        for ep_num, ep_f1, _ in epoch_macro_scores:
            marker = " ← BEST" if ep_num == best_epoch_num else ""
            self.logger.info(f"     Epoch {ep_num}: Macro-F1 = {ep_f1:.4f}{marker}")

        save_path = os.path.join(self.model_dir, "querytype_model")
        import shutil
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        shutil.copytree(best_ckpt_dir, save_path)
        self.logger.info(f"✅ Best epoch {best_epoch_num} copied → {save_path}")
        self.logger.info("=" * 80)

        # ── Final evaluation using best epoch weights ──────────────────────────
        self.logger.info("📊 Evaluating Level 3 model (best epoch)...")
        model.load(save_path)
        model.to(device)
        model.eval()

        test_dataloader = self._create_dataloader(test_inputs, test_masks, batch_size=16, shuffle=False)
        qt_preds = []
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, attention_mask = [t.to(device) for t in batch]
                logits = model(input_ids, attention_mask)
                qt_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

        # ── Log each test email flow for Level 3 ──────────────────────────────
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("🧪 LEVEL 3 — TEST SET EVALUATION (20% split) — EMAIL BY EMAIL")
        self.logger.info("   INPUT  : 'MasterDepartment: [GT]  Department: [GT]  Email: [text]'")
        self.logger.info("   NOTE   : Both prefixes are GROUND TRUTH from annotation")
        self.logger.info("            (teacher forcing — same as training, NOT chained model outputs)")
        self.logger.info("   LABEL  : querytype (ground truth from annotation)")
        self.logger.info("=" * 70)
        for idx, (_, row) in enumerate(test_df.iterrows()):
            true_label = row['querytype']
            pred_label = qt_encoder.inverse_transform([qt_preds[idx]])[0]
            full_input = test_texts_conditional[idx]
            match = "✅ CORRECT" if true_label == pred_label else "❌ WRONG"
            self.logger.info(f"  Test Email [{idx+1}]")
            self.logger.info(f"    PREFIX-1 (GT)    : \"MasterDepartment: {row['masterdepartment']}\"")
            self.logger.info(f"    PREFIX-2 (GT)    : \"Department: {row['department']}\"")
            self.logger.info(f"    EMAIL TEXT       : \"{row['text']}\"")
            self.logger.info(f"    FULL INPUT SENT  : \"{full_input}\"")
            self.logger.info(f"    GROUND TRUTH     : \"{true_label}\"")
            self.logger.info(f"    PREDICTED        : \"{pred_label}\"")
            self.logger.info(f"    RESULT           : {match}")
            self.logger.info("")
        self.logger.info("=" * 70)

        qt_accuracy = accuracy_score(test_qt_encoded, qt_preds)
        qt_f1       = f1_score(test_qt_encoded, qt_preds, average='weighted', zero_division=0)

        self.logger.info(f"✅ QueryType (best epoch {best_epoch_num}) - Accuracy: {qt_accuracy:.4f} | F1: {qt_f1:.4f}")
        
        # Save metrics to DB
        decoded_qt_true = qt_encoder.inverse_transform(test_qt_encoded)
        decoded_qt_pred = qt_encoder.inverse_transform(qt_preds)
        
        run_id = self.save_metrics_to_sqlite(
            3, qt_accuracy, qt_f1,
            len(train_df), len(test_df)
        )
        self.save_full_classification_report_to_sqlite(run_id, "querytype", decoded_qt_true, decoded_qt_pred)

        # Save evaluation folder
        self._save_evaluation_folder(
            "level3_querytype",
            decoded_qt_true, decoded_qt_pred,
            len(train_df), len(test_df)
        )

    def fit_external(self, event, data, tasks, **kwargs):
        """External training wrapper"""
        self.logger.info("🚀 External training initiated")
        return self.fit(event, data, tasks, **kwargs)

    def predict(self, tasks: List[Dict], texts: str, context: Optional[Dict] = None, **kwargs):
        """
        3-LEVEL HIERARCHICAL INFERENCE PIPELINE
        
        Just pass email text, automatically routes through:
        1. Level 1: Email → MasterDepartment
        2. Level 2: Email + MasterDepartment → Department
        3. Level 3: Email + MasterDepartment + Department → QueryType
        
        Returns all predictions in a single result.
        """
        
        try:
            self.logger.info("=" * 80)
            self.logger.info("🔍 PREDICTION REQUEST RECEIVED")
            self.logger.info("=" * 80)
            
            # ========== VALIDATION ==========
            if self.model is None:
                error_msg = "❌ Level 1 model not loaded - cannot make predictions"
                self.logger.error(error_msg)
                return []
            
            if self.tokenizer is None:
                error_msg = "❌ Tokenizer not loaded - cannot make predictions"
                self.logger.error(error_msg)
                return []
            
            # NOTE: label_interface is optional - only needed for Label Studio mode
            
            self.logger.info("✓ All required components loaded")

            # ========== GET LABEL STUDIO TAGS (Optional for external calls) ==========
            try:
                if self.label_interface is not None:
                    # Label Studio mode - get proper tags
                    def getMasterDepartmentAttrName(attrs):
                        return attrs == "masterdepartment"

                    def getDepartmentAttrName(attrs):
                        return attrs == "department"

                    def getQueryTypeAttrName(attrs):
                        return attrs == "querytype"

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
                    
                    self.logger.info("✓ Label Studio tags retrieved successfully")
                else:
                    # External API mode - use default tags
                    from_name_masterdepartment = "masterdepartment"
                    to_name_masterdepartment = "text"
                    from_name_department = "department"
                    to_name_department = "text"
                    from_name_querytype = "querytype"
                    to_name_querytype = "text"
                    
                    self.logger.info("✓ Using default tags (external API mode)")
                
            except Exception as e:
                self.logger.error(f"❌ Error getting tags: {e}")
                # Use defaults as fallback
                from_name_masterdepartment = "masterdepartment"
                to_name_masterdepartment = "text"
                from_name_department = "department"
                to_name_department = "text"
                from_name_querytype = "querytype"
                to_name_querytype = "text"
                self.logger.info("✓ Using fallback default tags")

            # ========== EXTRACT TEXT FROM INPUT ==========
            try:
                self.logger.info(f"📝 Input type: {type(texts)}")
                self.logger.info(f"📝 Input length: {len(texts) if isinstance(texts, (list, str)) else 'N/A'}")
                
                text_list = []
                
                if isinstance(texts, str):
                    # Single string
                    text_list = [texts]
                    self.logger.info("✓ Processing single string input")
                    
                elif isinstance(texts, list):
                    # List of texts or dicts
                    for idx, text in enumerate(texts):
                        if isinstance(text, dict):
                            if 'text' in text:
                                text_list.append(text['text'])
                            elif 'html' in text:
                                text_list.append(text['html'])
                            else:
                                text_list.append(str(text))
                        elif isinstance(text, str):
                            text_list.append(text)
                        else:
                            text_list.append(str(text))
                    
                    self.logger.info(f"✓ Processing list with {len(text_list)} items")
                    
                elif isinstance(texts, dict):
                    # Single dict
                    if 'text' in texts:
                        text_list = [texts['text']]
                    elif 'html' in texts:
                        text_list = [texts['html']]
                    else:
                        text_list = [str(texts)]
                    
                    self.logger.info("✓ Processing single dict input")
                    
                else:
                    # Fallback
                    text_list = [str(texts)]
                    self.logger.warning(f"⚠️  Unknown input type, converting to string")
                
                if not text_list or all(not t for t in text_list):
                    error_msg = "❌ No valid text found in input"
                    self.logger.error(error_msg)
                    return []
                
                self.logger.info(f"📊 Successfully extracted {len(text_list)} text(s) for processing")
                
                # Log first text sample for debugging
                if text_list:
                    sample_text = text_list[0][:100] + "..." if len(text_list[0]) > 100 else text_list[0]
                    self.logger.info(f"📄 Sample text: {sample_text}")
                
            except Exception as e:
                self.logger.error(f"❌ Error extracting text from input: {e}")
                self.logger.exception("Full traceback:")
                return []

            # ========== TOKENIZATION ==========
            try:
                tokenizer = self.tokenizer
                MAX_LEN = 256
                batch_size = 16

                self.logger.info(f"🔤 Tokenizing {len(text_list)} text(s)...")
                
                # Tokenize and pad
                _inputs, _masks = self._tokenize_and_pad(text_list, tokenizer, MAX_LEN)
                
                self.logger.info(f"✓ Tokenization complete - Shape: {_inputs.shape}")

                # Create DataLoader
                dataloader = self._create_dataloader(
                    _inputs, _masks, batch_size=batch_size, shuffle=False
                )
                
                self.logger.info(f"✓ DataLoader created - {len(dataloader)} batches")
                
            except Exception as e:
                self.logger.error(f"❌ Error during tokenization: {e}")
                self.logger.exception("Full traceback:")
                return []

            # ========== SETUP MODELS ==========
            try:
                device = self._get_device()
                self.logger.info(f"🖥️  Using device: {device}")
                
                # Move models to device and set to eval mode
                self.model.to(device)
                self.model.eval()
                self.logger.info("✓ Level 1 model ready")

                if self.department_model is not None:
                    self.department_model.to(device)
                    self.department_model.eval()
                    self.logger.info("✓ Level 2 model ready - Department predictions enabled")
                else:
                    self.logger.info("⚠️  Level 2 model not loaded - Department predictions will be skipped")

                if self.querytype_model is not None:
                    self.querytype_model.to(device)
                    self.querytype_model.eval()
                    self.logger.info("✓ Level 3 model ready - QueryType predictions enabled")
                else:
                    self.logger.info("⚠️  Level 3 model not loaded - QueryType predictions will be skipped")
                    
            except Exception as e:
                self.logger.error(f"❌ Error setting up models: {e}")
                self.logger.exception("Full traceback:")
                return []

            # ========== RUN INFERENCE ==========
            try:
                self.logger.info("🚀 Starting inference...")
                
                predictions = []
                global_text_idx = 0

                with torch.no_grad():
                    for batch_idx, batch in enumerate(dataloader):
                        self.logger.info(f"   Processing batch {batch_idx + 1}/{len(dataloader)}...")
                        
                        input_ids, attention_mask = [t.to(device) for t in batch]
            
                        # ========== LEVEL 1: MasterDepartment ==========
                        try:
                            self.logger.info("")
                            self.logger.info("=" * 70)
                            self.logger.info("🔍 INFERENCE — LEVEL 1 : MasterDepartmentModel")
                            self.logger.info("=" * 70)
                            self.logger.info("  INPUT : Raw email text only — no prefix")
                            self.logger.info(f"  MODEL : MasterDepartmentModel")
                            self.logger.info(f"  BATCH : {input_ids.shape[0]} email(s) in this batch")

                            masterdept_logits = self.model(input_ids, attention_mask)
                            masterdept_probs = torch.softmax(masterdept_logits, dim=1)
                            masterdept_preds = torch.argmax(masterdept_probs, dim=1)

                            decoded_masterdept_preds = self._decode_predictions(
                                masterdept_preds, 'masterdepartment'
                            )

                            self.logger.info("  OUTPUT:")
                            for idx_l1, pred_l1 in enumerate(decoded_masterdept_preds):
                                score_l1 = float(masterdept_probs[idx_l1][masterdept_preds[idx_l1]].item())
                                self.logger.info(f"    Email[{idx_l1}] → MasterDepartment = '{pred_l1}'  (score: {score_l1:.4f})")
                            self.logger.info("=" * 70)
                            
                        except Exception as e:
                            self.logger.error(f"❌ Error in Level 1 prediction: {e}")
                            self.logger.exception("Full traceback:")
                            raise

                        for i in range(len(masterdept_preds)):
                            try:
                                current_text = text_list[global_text_idx]
                                global_text_idx += 1

                                self.logger.info("")
                                self.logger.info("=" * 70)
                                self.logger.info(f"📧 EMAIL [{global_text_idx}] — FULL RAW TEXT RECEIVED FOR PREDICTION:")
                                self.logger.info(f"    {current_text}")
                                self.logger.info(f"  L1 RESULT → MasterDepartment = '{decoded_masterdept_preds[i]}'  (score: {float(masterdept_probs[i][masterdept_preds[i]].item()):.4f})")
                                self.logger.info("=" * 70)

                                # ---------- MASTER DEPARTMENT ----------
                                predictions.append({
                                    "from_name": from_name_masterdepartment,
                                    "to_name": to_name_masterdepartment,
                                    "type": "taxonomy",
                                    "value": {
                                        "taxonomy": [
                                            [decoded_masterdept_preds[i]]
                                        ],
                                        "score": float(masterdept_probs[i][masterdept_preds[i]].item())
                                    }
                                })

                                # ========== LEVEL 2: DEPARTMENT ==========
                                if self.department_model is not None:
                                    try:
                                        conditional_text = (
                                            f"MasterDepartment: {decoded_masterdept_preds[i]} "
                                            f"Email: {current_text}"
                                        )

                                        self.logger.info("")
                                        self.logger.info("=" * 70)
                                        self.logger.info("🔍 INFERENCE — LEVEL 2 : DepartmentNNModel")
                                        self.logger.info("=" * 70)
                                        self.logger.info("  HOW INPUT IS BUILT:")
                                        self.logger.info(f"    PREFIX  → 'MasterDepartment: {decoded_masterdept_preds[i]}'")
                                        self.logger.info(f"              ↑ this comes from Level 1 model output (NOT annotation)")
                                        self.logger.info(f"    EMAIL   → '{current_text}'")
                                        self.logger.info(f"  FULL STRING SENT TO MODEL:")
                                        self.logger.info(f"    >>> '{conditional_text}'")
                                        self.logger.info("=" * 70)

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

                                        if isinstance(dept_logits, (tuple, list)):
                                            dept_logits = dept_logits[0]

                                        dept_probs = torch.softmax(dept_logits, dim=1)
                                        
                                        # ========== HIERARCHICAL CONSTRAINT FOR LEVEL 2 ==========
                                        dept_encoder = self.department_model.get_encoder()
                                        all_departments = dept_encoder.classes_
                                        
                                        current_masterdept = decoded_masterdept_preds[i]
                                        
                                        # Filter to only Departments that start with current MasterDepartment
                                        valid_indices = []
                                        valid_labels = []
                                        
                                        for idx, dept_label in enumerate(all_departments):
                                            if dept_label.startswith(current_masterdept):
                                                valid_indices.append(idx)
                                                valid_labels.append(dept_label)
                                        
                                        if len(valid_indices) == 0:
                                            self.logger.warning(f"⚠️  No Departments found for MasterDepartment: {current_masterdept}")
                                            continue
                                        
                                        # Get probabilities only for valid indices
                                        valid_probs = dept_probs[0][valid_indices]
                                        
                                        # Find the highest probability among valid options
                                        best_valid_idx = torch.argmax(valid_probs).item()
                                        dept_idx = valid_indices[best_valid_idx]
                                        department_label = valid_labels[best_valid_idx]
                                        final_score = valid_probs[best_valid_idx].item()
                                        
                                        # Log the constraint application
                                        original_prediction_idx = torch.argmax(dept_probs, dim=1).item()
                                        original_prediction = all_departments[original_prediction_idx]
                                        
                                        if original_prediction_idx != dept_idx:
                                            self.logger.warning(f"  ⚠️  HIERARCHICAL CONSTRAINT changed prediction!")
                                            self.logger.warning(f"      Before : '{original_prediction}' (score: {dept_probs[0][original_prediction_idx].item():.3f})")
                                            self.logger.warning(f"      After  : '{department_label}' (score: {final_score:.3f})")
                                            self.logger.warning(f"      Reason : original prediction did not belong to MasterDept '{current_masterdept}'")
                                        else:
                                            self.logger.info(f"  ✅ LEVEL 2 RESULT : '{department_label}'  (score: {final_score:.3f})")

                                        predictions.append({
                                            "from_name": from_name_department,
                                            "to_name": to_name_department,
                                            "type": "taxonomy",
                                            "value": {
                                                "taxonomy": [department_label.split(" > ")],
                                                "score": float(final_score)
                                            }
                                        })

                                        # ========== LEVEL 3: QUERY TYPE ==========
                                        if self.querytype_model is not None:
                                            try:
                                                conditional_text_qt = (
                                                    f"MasterDepartment: {decoded_masterdept_preds[i]} "
                                                    f"Department: {department_label} "
                                                    f"Email: {current_text}"
                                                )

                                                self.logger.info("")
                                                self.logger.info("=" * 70)
                                                self.logger.info("🔍 INFERENCE — LEVEL 3 : QueryTypeNNModel")
                                                self.logger.info("=" * 70)
                                                self.logger.info("  HOW INPUT IS BUILT:")
                                                self.logger.info(f"    PREFIX-1 → 'MasterDepartment: {decoded_masterdept_preds[i]}'")
                                                self.logger.info(f"               ↑ from Level 1 model output")
                                                self.logger.info(f"    PREFIX-2 → 'Department: {department_label}'")
                                                self.logger.info(f"               ↑ from Level 2 model output")
                                                self.logger.info(f"    EMAIL    → '{current_text}'")
                                                self.logger.info(f"  FULL STRING SENT TO MODEL:")
                                                self.logger.info(f"    >>> '{conditional_text_qt}'")
                                                self.logger.info("=" * 70)

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

                                                if isinstance(qt_logits, (tuple, list)):
                                                    qt_logits = qt_logits[0]

                                                qt_probs = torch.softmax(qt_logits, dim=1)
                                                
                                                # ========== HIERARCHICAL CONSTRAINT ==========
                                                qt_encoder = self.querytype_model.get_encoder()
                                                all_querytypes = qt_encoder.classes_
                                                
                                                # Filter to only QueryTypes that start with current department
                                                valid_indices = []
                                                valid_labels = []
                                                
                                                for idx, qt_label in enumerate(all_querytypes):
                                                    if qt_label.startswith(department_label):
                                                        valid_indices.append(idx)
                                                        valid_labels.append(qt_label)
                                                
                                                if len(valid_indices) == 0:
                                                    self.logger.warning(f"⚠️  No QueryTypes found for department: {department_label}")
                                                    continue
                                                
                                                # Get probabilities only for valid indices
                                                valid_probs = qt_probs[0][valid_indices]
                                                
                                                # Find the highest probability among valid options
                                                best_valid_idx = torch.argmax(valid_probs).item()
                                                qt_idx = valid_indices[best_valid_idx]
                                                querytype_label = valid_labels[best_valid_idx]
                                                final_score = valid_probs[best_valid_idx].item()
                                                
                                                # Log the constraint application
                                                original_prediction_idx = torch.argmax(qt_probs, dim=1).item()
                                                original_prediction = all_querytypes[original_prediction_idx]
                                                
                                                if original_prediction_idx != qt_idx:
                                                    self.logger.warning(f"  ⚠️  HIERARCHICAL CONSTRAINT changed prediction!")
                                                    self.logger.warning(f"      Before : '{original_prediction}' (score: {qt_probs[0][original_prediction_idx].item():.3f})")
                                                    self.logger.warning(f"      After  : '{querytype_label}' (score: {final_score:.3f})")
                                                    self.logger.warning(f"      Reason : original did not belong to Department '{department_label}'")
                                                else:
                                                    self.logger.info(f"  ✅ LEVEL 3 RESULT : '{querytype_label}'  (score: {final_score:.3f})")

                                                predictions.append({
                                                    "from_name": from_name_querytype,
                                                    "to_name": to_name_querytype,
                                                    "type": "taxonomy",
                                                    "value": {
                                                        "taxonomy": [querytype_label.split(" > ")],
                                                        "score": float(final_score)
                                                    }
                                                })

                                                self.logger.info("")
                                                self.logger.info("╔══════════════════════════════════════════════════════════════╗")
                                                self.logger.info("║  FINAL PREDICTION SUMMARY")
                                                self.logger.info("╠══════════════════════════════════════════════════════════════╣")
                                                self.logger.info(f"║  L1 MasterDept  : '{decoded_masterdept_preds[i]}'")
                                                self.logger.info(f"║  L2 Department  : '{department_label}'")
                                                self.logger.info(f"║  L3 QueryType   : '{querytype_label}'")
                                                self.logger.info("╚══════════════════════════════════════════════════════════════╝")
                                                
                                            except Exception as e:
                                                self.logger.error(f"❌ Error in Level 3 prediction for item {i}: {e}")
                                                self.logger.exception("Full traceback:")
                                                # Continue without Level 3 prediction
                                                
                                    except Exception as e:
                                        self.logger.error(f"❌ Error in Level 2 prediction for item {i}: {e}")
                                        self.logger.exception("Full traceback:")
                                        # Continue without Level 2 and Level 3 predictions
                                        
                            except Exception as e:
                                self.logger.error(f"❌ Error processing item {i} in batch: {e}")
                                self.logger.exception("Full traceback:")
                                # Continue to next item

                self.logger.info("=" * 80)
                self.logger.info(f"✅ INFERENCE COMPLETE - Generated {len(predictions)} predictions")
                self.logger.info("=" * 80)
                
                return predictions
                
            except Exception as e:
                self.logger.error(f"❌ Error during inference loop: {e}")
                self.logger.exception("Full traceback:")
                return []
            
        except Exception as e:
            # Catch-all for any unexpected errors
            self.logger.error("=" * 80)
            self.logger.error("❌ CRITICAL ERROR IN PREDICT METHOD")
            self.logger.error("=" * 80)
            self.logger.error(f"Error: {e}")
            self.logger.exception("Full traceback:")
            
            # Return empty list instead of crashing
            return []