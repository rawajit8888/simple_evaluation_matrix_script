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


# ========== MasterDepartmentModel Class (Level 1 Model) ==========

class MasterDepartmentModel(nn.Module):
    """
    MasterDepartment classification model.
    Level 1: Email → MasterDepartment
    """
    
    def __init__(self, modelname, num_labels):
        super(MasterDepartmentModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(modelname)
        self.dropout = nn.Dropout(0.1)
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
        
        pooled_output = outputs[1]
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
                logger.warning("⚠️  No bundled encoder found - using external encoder")
            
            self.eval()
            
        except Exception as e:
            logger.error(f"❌ Error loading MasterDepartment model: {e}")
            raise
    
    def get_encoder(self):
        """Safely retrieve encoder"""
        if self.encoder is None:
            raise ValueError("MasterDepartment encoder is None")
        return self.encoder


# ========== Training Progress Logger ==========

class TrainingLogger:
    """Helper class for clean training progress logging"""
    
    def __init__(self, logger, num_epochs, num_batches):
        self.logger = logger
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.start_time = None
        
    def start_training(self, level_name):
        self.start_time = time.time()
        self.logger.info(f"📊 Total Epochs: {self.num_epochs}")
        self.logger.info(f"📊 Batches per Epoch: {self.num_batches}")
        self.logger.info("")
        
    def start_epoch(self, epoch):
        self.logger.info(f"📅 EPOCH {epoch + 1}/{self.num_epochs}")
        
    def log_batch(self, epoch, batch_idx, loss):
        if (batch_idx + 1) % 10 == 0:
            self.logger.info(f"   Batch {batch_idx + 1}/{self.num_batches} | Loss: {loss:.4f}")
            
    def end_epoch(self, epoch, avg_loss):
        self.logger.info(f"✓ Epoch {epoch + 1} complete | Avg Loss: {avg_loss:.4f}")
        self.logger.info("")
        
    def end_training(self):
        elapsed = time.time() - self.start_time
        self.logger.info(f"⏱️  Total Time: {elapsed:.2f}s")


# ========== Main MultiTaskBertModel Class ==========

class MultiTaskBertModel:
    """
    3-Level Hierarchical Classification System
    
    Level 1: Email → MasterDepartment
    Level 2: Email + MasterDepartment → Department  
    Level 3: Email + MasterDepartment + Department → QueryType
    
    Trains 3 separate models in one training session.
    """
    
    def __init__(self, config, logger):
        self.logger = logger
        self.config = config
        
        self.baseline_model_name = config.get('BASELINE_MODEL_NAME', 'bert-base-uncased')
        self.model_dir = pathlib.Path(config.get('MODEL_DIR', './results/bert-classification-sentiment'))
        
        # Initialize tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(self.baseline_model_name)
        
        # Model paths
        self.masterdepartment_model_dir = self.model_dir / "masterdepartment_model"
        self.department_model_dir = self.model_dir / "department_model"
        self.querytype_model_dir = self.model_dir / "querytype_model"
        
        # Initialize models (will be loaded if exist)
        self.model = None  # Level 1: MasterDepartment
        self.department_model = None  # Level 2: Department
        self.querytype_model = None  # Level 3: QueryType
        
        # Will be injected from main model.py
        self.label_interface = None
        self.processed_label_encoders = None
        self.preload_task_data = None
        
        # Database for metrics
        self.metrics_db_path = self.model_dir / "metrics.db"
        
    def _get_device(self):
        """Get available device"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _tokenize_and_pad(self, texts, tokenizer, max_len):
        """Tokenize and pad texts"""
        input_ids = []
        attention_masks = []
        
        for text in texts:
            encoded = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        
        return input_ids, attention_masks
    
    def _create_dataloader(self, input_ids, attention_masks, labels=None, batch_size=16, shuffle=True):
        """Create DataLoader"""
        if labels is not None:
            labels = torch.tensor(labels)
            dataset = TensorDataset(input_ids, attention_masks, labels)
        else:
            dataset = TensorDataset(input_ids, attention_masks)
        
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        
        return dataloader
    
    def _decode_predictions(self, predictions, task_name):
        """Decode predictions using appropriate encoder"""
        if task_name == 'masterdepartment':
            if self.model and self.model.encoder:
                return self.model.encoder.inverse_transform(predictions)
            elif self.processed_label_encoders:
                encoder = self.processed_label_encoders['masterdepartment']
                if encoder:
                    return encoder.inverse_transform(predictions)
        
        elif task_name == 'department':
            if self.department_model and self.department_model.encoder:
                return self.department_model.encoder.inverse_transform(predictions)
            elif self.processed_label_encoders:
                encoder = self.processed_label_encoders['department']
                if encoder:
                    return encoder.inverse_transform(predictions)
        
        elif task_name == 'querytype':
            if self.querytype_model and self.querytype_model.encoder:
                return self.querytype_model.encoder.inverse_transform(predictions)
            elif self.processed_label_encoders:
                encoder = self.processed_label_encoders['querytype']
                if encoder:
                    return encoder.inverse_transform(predictions)
        
        return [str(p) for p in predictions]
    
    def reload_model(self):
        """Load all three models from disk"""
        self.logger.info("🔄 Reloading all models...")
        
        device = self._get_device()
        
        # Load Level 1: MasterDepartment
        if self.masterdepartment_model_dir.exists():
            try:
                self.logger.info(f"📂 Loading Level 1 model from {self.masterdepartment_model_dir}")
                
                # Load encoder first to get num_labels
                encoder_path = self.masterdepartment_model_dir / 'masterdepartment_encoder.pkl'
                if encoder_path.exists():
                    with open(encoder_path, 'rb') as f:
                        encoder = pickle.load(f)
                    num_labels = len(encoder.classes_)
                else:
                    self.logger.warning("⚠️  MasterDepartment encoder not found, using default")
                    num_labels = 100
                
                self.model = MasterDepartmentModel(self.baseline_model_name, num_labels)
                self.model.load(str(self.masterdepartment_model_dir))
                self.model.to(device)
                self.logger.info("✅ Level 1 model loaded successfully")
            except Exception as e:
                self.logger.warning(f"⚠️  Level 1 model not found or failed to load: {e}")
                self.model = None
        else:
            self.logger.warning(f"⚠️  Level 1 model not found at {self.masterdepartment_model_dir}")
        
        # Load Level 2: Department
        if self.department_model_dir.exists():
            try:
                self.logger.info(f"📂 Loading Level 2 model from {self.department_model_dir}")
                
                # Load encoder first to get num_labels
                encoder_path = self.department_model_dir / 'department_encoder.pkl'
                if encoder_path.exists():
                    with open(encoder_path, 'rb') as f:
                        encoder = pickle.load(f)
                    num_labels = len(encoder.classes_)
                else:
                    self.logger.warning("⚠️  Department encoder not found, using default")
                    num_labels = 100
                
                self.department_model = DepartmentNNModel(self.baseline_model_name, num_labels)
                self.department_model.load(str(self.department_model_dir))
                self.department_model.to(device)
                self.logger.info("✅ Level 2 model loaded successfully")
            except Exception as e:
                self.logger.warning(f"⚠️  Level 2 model not found or failed to load: {e}")
                self.department_model = None
        else:
            self.logger.warning(f"⚠️  Level 2 (Department) model not found at {self.department_model_dir} - will be created during training")
        
        # Load Level 3: QueryType
        if self.querytype_model_dir.exists():
            try:
                self.logger.info(f"📂 Loading Level 3 model from {self.querytype_model_dir}")
                
                # Load encoder first to get num_labels
                encoder_path = self.querytype_model_dir / 'querytype_encoder.pkl'
                if encoder_path.exists():
                    with open(encoder_path, 'rb') as f:
                        encoder = pickle.load(f)
                    num_labels = len(encoder.classes_)
                else:
                    self.logger.warning("⚠️  QueryType encoder not found, using default")
                    num_labels = 100
                
                self.querytype_model = QueryTypeNNModel(self.baseline_model_name, num_labels)
                self.querytype_model.load(str(self.querytype_model_dir))
                self.querytype_model.to(device)
                self.logger.info("✅ Level 3 model loaded successfully")
            except Exception as e:
                self.logger.warning(f"⚠️  Level 3 model not found or failed to load: {e}")
                self.querytype_model = None
        else:
            self.logger.warning(f"⚠️  Level 3 (QueryType) model not found at {self.querytype_model_dir} - will be created during training")
        
        self.logger.info("✓ Model reload complete")
    
    # ========== METRICS DATABASE ==========
    
    def init_metrics_db(self):
        """Initialize SQLite database for metrics storage"""
        import sqlite3
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        conn = sqlite3.connect(str(self.metrics_db_path))
        cursor = conn.cursor()
        
        # Create runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
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
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        """)
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"✓ Metrics database initialized at {self.metrics_db_path}")
    
    def save_metrics_to_sqlite(self, level, accuracy, f1, train_size, test_size):
        """Save metrics to SQLite database"""
        import sqlite3
        
        conn = sqlite3.connect(str(self.metrics_db_path))
        cursor = conn.cursor()
        
        # Insert run
        cursor.execute("""
            INSERT INTO runs (timestamp, train_samples, test_samples)
            VALUES (?, ?, ?)
        """, (datetime.now().isoformat(), train_size, test_size))
        
        run_id = cursor.lastrowid
        
        # Insert metrics
        task_name = f"level{level}"
        cursor.execute("""
            INSERT INTO metrics (run_id, task, metric, value)
            VALUES (?, ?, ?, ?)
        """, (run_id, task_name, "accuracy_overall", accuracy))
        
        cursor.execute("""
            INSERT INTO metrics (run_id, task, metric, value)
            VALUES (?, ?, ?, ?)
        """, (run_id, task_name, "f1_score", f1))
        
        conn.commit()
        conn.close()
        
        return run_id
    
    def save_full_classification_report_to_sqlite(self, run_id, task, y_true, y_pred):
        """Save per-class metrics to database"""
        import sqlite3
        from sklearn.metrics import classification_report
        
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        conn = sqlite3.connect(str(self.metrics_db_path))
        cursor = conn.cursor()
        
        for class_name, metrics in report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if metric_name in ['precision', 'recall', 'f1-score']:
                        metric_key = f"{class_name}_{metric_name.replace('-', '_')}"
                        cursor.execute("""
                            INSERT INTO metrics (run_id, task, metric, value)
                            VALUES (?, ?, ?, ?)
                        """, (run_id, task, metric_key, value))
        
        # Add weighted averages
        if 'weighted avg' in report:
            for metric_name, value in report['weighted avg'].items():
                if metric_name in ['precision', 'recall', 'f1-score']:
                    cursor.execute("""
                        INSERT INTO metrics (run_id, task, metric, value)
                        VALUES (?, ?, ?, ?)
                    """, (run_id, task, f"weighted_avg_{metric_name.replace('-', '_')}", value))
        
        conn.commit()
        conn.close()
    
    def get_latest_metrics_from_db(self):
        """Retrieve latest metrics from database"""
        import sqlite3
        
        if not self.metrics_db_path.exists():
            return {"error": "No metrics database found"}
        
        conn = sqlite3.connect(str(self.metrics_db_path))
        cursor = conn.cursor()
        
        # Get latest run
        cursor.execute("SELECT MAX(run_id) FROM runs")
        result = cursor.fetchone()
        
        if result[0] is None:
            conn.close()
            return {"error": "No training runs found"}
        
        latest_run = result[0]
        
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
        
        Handles multiple label configurations:
        1. Single taxonomy field with full path: ["Internet Banking", "Account Access", "Unblock"]
        2. Multiple taxonomy fields (one per level)
        3. Mixed configurations
        
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
        
        # Debug: Log the annotation structure
        if len(annotation_result) > 0:
            self.logger.debug(f"📝 Annotation structure: {annotation_result[0]}")
        
        # Try to extract from any taxonomy field
        for item in annotation_result:
            value = item.get('value', {})
            taxonomy = value.get('taxonomy', [[]])
            from_name = item.get('from_name', '').lower()
            
            if not taxonomy or not taxonomy[0]:
                continue
            
            # Get the full path (could be 1, 2, or 3 levels)
            full_path = taxonomy[0]
            
            self.logger.debug(f"📋 Processing taxonomy from '{from_name}': {full_path}")
            
            # Strategy 1: If this is a querytype field with 3 levels
            if len(full_path) >= 3:
                result['masterdepartment'] = full_path[0]
                result['department'] = f"{full_path[0]} > {full_path[1]}"
                result['querytype'] = f"{full_path[0]} > {full_path[1]} > {full_path[2]}"
                self.logger.debug(f"✓ Extracted all 3 levels from field '{from_name}'")
                break  # Found complete data, no need to check other fields
            
            # Strategy 2: If this is a department field with 2 levels
            elif len(full_path) >= 2 and not result['department']:
                result['masterdepartment'] = full_path[0]
                result['department'] = f"{full_path[0]} > {full_path[1]}"
                self.logger.debug(f"✓ Extracted 2 levels from field '{from_name}'")
            
            # Strategy 3: If this is a masterdepartment field with 1 level
            elif len(full_path) >= 1 and not result['masterdepartment']:
                result['masterdepartment'] = full_path[0]
                self.logger.debug(f"✓ Extracted 1 level from field '{from_name}'")
        
        # Validation - need at least masterdepartment
        if not result['masterdepartment']:
            self.logger.warning(f"⚠️  No masterdepartment found in task {task.get('id', 'unknown')}")
            self.logger.warning(f"   Annotation structure: {annotation_result}")
            return None
        
        return result
    
    # ========== TRAINING PIPELINE ==========
    
    def fit(self, event, data, tasks, **kwargs):
        """
        3-Level Hierarchical Training Pipeline
        
        Level 1: Email → MasterDepartment
        Level 2: Email + MasterDepartment → Department
        Level 3: Email + MasterDepartment + Department → QueryType
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
            extracted = self._extract_taxonomy_from_task(task)
            if extracted and extracted['text']:
                train_data.append(extracted)
        
        if len(train_data) == 0:
            self.logger.error("❌ No valid training data found")
            self.logger.error("Please check your Label Studio annotations:")
            self.logger.error("  - Ensure tasks have taxonomy labels")
            self.logger.error("  - Check that the taxonomy field contains hierarchical paths")
            self.logger.error("  - Example format: ['Internet Banking', 'Account Access', 'Unblock']")
            return
        
        self.logger.info(f"✓ Prepared {len(train_data)} training samples")
        
        # Log sample distribution
        masterdept_count = sum(1 for d in train_data if d['masterdepartment'])
        dept_count = sum(1 for d in train_data if d['department'])
        qt_count = sum(1 for d in train_data if d['querytype'])
        
        self.logger.info(f"📊 Sample distribution:")
        self.logger.info(f"   Level 1 (MasterDepartment): {masterdept_count} samples")
        self.logger.info(f"   Level 2 (Department): {dept_count} samples")
        self.logger.info(f"   Level 3 (QueryType): {qt_count} samples")
        
        # Log sample for debugging
        if train_data:
            sample = train_data[0]
            self.logger.info(f"📝 Sample annotation:")
            self.logger.info(f"   MasterDepartment: {sample['masterdepartment']}")
            self.logger.info(f"   Department: {sample['department']}")
            self.logger.info(f"   QueryType: {sample['querytype']}")
        
        # Convert to DataFrame
        df = pd.DataFrame(train_data)
        
        # CRITICAL FIX: Smart train/test split with fallback
        from sklearn.model_selection import train_test_split
        
        # Count samples per class
        class_counts = df['masterdepartment'].value_counts()
        min_samples = class_counts.min()
        
        if min_samples >= 2:
            # Can use stratified split
            self.logger.info(f"✓ Using stratified split (min class size: {min_samples})")
            try:
                train_df, test_df = train_test_split(
                    df, test_size=0.2, random_state=42, 
                    stratify=df['masterdepartment']
                )
            except ValueError as e:
                self.logger.warning(f"⚠️  Stratified split failed: {e}")
                self.logger.warning("⚠️  Falling back to random split")
                train_df, test_df = train_test_split(
                    df, test_size=0.2, random_state=42
                )
        else:
            # Use random split
            self.logger.warning(f"⚠️  Some classes have only 1 sample - using random split")
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42
            )
        
        self.logger.info(f"📊 Train: {len(train_df)} | Test: {len(test_df)}")
        
        # ========== TRAIN LEVEL 1: MasterDepartment ONLY ==========
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🎯 LEVEL 1 TRAINING: MasterDepartment ONLY")
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
            self.logger.info(f"📊 Department training samples: {len(dept_train_df)}")
            self._train_level2(dept_train_df, dept_test_df)
        else:
            self.logger.warning("⚠️  No Department labels found - skipping Level 2 training")
            self.logger.warning("⚠️  Make sure your Label Studio tasks have at least 2-level taxonomy:")
            self.logger.warning("    Example: ['Internet Banking', 'Account Access']")
        
        # ========== TRAIN LEVEL 3: QueryType ==========
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🎯 LEVEL 3 TRAINING: QueryType")
        self.logger.info("=" * 80)
        
        # Filter samples with querytype labels
        qt_train_df = train_df[train_df['querytype'].notna()].copy()
        qt_test_df = test_df[test_df['querytype'].notna()].copy()
        
        if len(qt_train_df) > 0:
            self.logger.info(f"📊 QueryType training samples: {len(qt_train_df)}")
            self._train_level3(qt_train_df, qt_test_df)
        else:
            self.logger.warning("⚠️  No QueryType labels found - skipping Level 3 training")
            self.logger.warning("⚠️  Make sure your Label Studio tasks have 3-level taxonomy:")
            self.logger.warning("    Example: ['Internet Banking', 'Account Access', 'Unblock']")
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🎉 3-LEVEL HIERARCHICAL TRAINING COMPLETE")
        self.logger.info("=" * 80)
    
    def _train_level1(self, train_df, test_df):
        """Train Level 1: MasterDepartment ONLY"""
        
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels
        masterdept_encoder = LabelEncoder()
        
        train_masterdept_encoded = masterdept_encoder.fit_transform(train_df['masterdepartment'])
        test_masterdept_encoded = masterdept_encoder.transform(test_df['masterdepartment'])
        
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
        
        train_inputs, train_masks = self._tokenize_and_pad(train_df['text'].tolist(), tokenizer, MAX_LEN)
        test_inputs, test_masks = self._tokenize_and_pad(test_df['text'].tolist(), tokenizer, MAX_LEN)
        
        # Create DataLoaders
        train_dataloader = self._create_dataloader(
            train_inputs, train_masks,
            labels=train_masterdept_encoded,
            batch_size=16,
            shuffle=True
        )
        
        # Training setup
        NUM_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 3))
        LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-5))
        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
        criterion = nn.CrossEntropyLoss()
        
        # Training logger
        training_logger = TrainingLogger(self.logger, NUM_EPOCHS, len(train_dataloader))
        training_logger.start_training("Level 1 (MasterDepartment)")
        
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
        self.logger.info("📊 Evaluating Level 1 model...")
        
        model.eval()
        
        test_dataloader = self._create_dataloader(
            test_inputs, test_masks,
            batch_size=16,
            shuffle=False
        )
        
        masterdept_preds = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, attention_mask = [t.to(device) for t in batch]
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                masterdept_preds.extend(preds.cpu().numpy())
        
        # Calculate metrics
        md_accuracy = accuracy_score(test_masterdept_encoded, masterdept_preds)
        md_f1 = f1_score(test_masterdept_encoded, masterdept_preds, average='weighted', zero_division=0)
        
        self.logger.info(f"✅ Level 1 Accuracy: {md_accuracy:.4f}")
        self.logger.info(f"✅ Level 1 F1-Score: {md_f1:.4f}")
        
        # Save model and encoder
        model.set_encoder(masterdept_encoder)
        model.save(str(self.masterdepartment_model_dir))
        
        # Update instance model
        self.model = model
        
        # Save metrics to database
        decoded_md_true = masterdept_encoder.inverse_transform(test_masterdept_encoded)
        decoded_md_pred = masterdept_encoder.inverse_transform(masterdept_preds)
        
        run_id = self.save_metrics_to_sqlite(
            1, md_accuracy, md_f1,
            len(train_df), len(test_df)
        )
        
        self.save_full_classification_report_to_sqlite(run_id, "masterdepartment", decoded_md_true, decoded_md_pred)
    
    def _train_level2(self, train_df, test_df):
        """Train Level 2: Department (conditioned on MasterDepartment)"""
        
        from sklearn.preprocessing import LabelEncoder
        
        # Encode department labels
        dept_encoder = LabelEncoder()
        
        train_dept_encoded = dept_encoder.fit_transform(train_df['department'])
        test_dept_encoded = dept_encoder.transform(test_df['department'])
        
        self.logger.info(f"📊 Department classes: {len(dept_encoder.classes_)}")
        
        # Initialize model
        num_dept = len(dept_encoder.classes_)
        device = self._get_device()
        
        dept_model = DepartmentNNModel(
            self.baseline_model_name,
            num_labels=num_dept
        )
        dept_model.to(device)
        
        # Create conditional texts (prepend MasterDepartment)
        train_conditional_texts = [
            f"MasterDepartment: {md} Email: {text}"
            for md, text in zip(train_df['masterdepartment'], train_df['text'])
        ]
        
        test_conditional_texts = [
            f"MasterDepartment: {md} Email: {text}"
            for md, text in zip(test_df['masterdepartment'], test_df['text'])
        ]
        
        # Tokenize
        tokenizer = self.tokenizer
        MAX_LEN = 256
        
        train_inputs, train_masks = self._tokenize_and_pad(train_conditional_texts, tokenizer, MAX_LEN)
        test_inputs, test_masks = self._tokenize_and_pad(test_conditional_texts, tokenizer, MAX_LEN)
        
        # Create DataLoaders
        train_dataloader = self._create_dataloader(
            train_inputs, train_masks,
            labels=train_dept_encoded,
            batch_size=16,
            shuffle=True
        )
        
        # Training setup
        NUM_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 3))
        LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-5))
        
        optimizer = AdamW(dept_model.parameters(), lr=LEARNING_RATE, eps=1e-8)
        criterion = nn.CrossEntropyLoss()
        
        # Training logger
        training_logger = TrainingLogger(self.logger, NUM_EPOCHS, len(train_dataloader))
        training_logger.start_training("Level 2 (Department)")
        
        # Training loop
        for epoch in range(NUM_EPOCHS):
            training_logger.start_epoch(epoch)
            
            dept_model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                input_ids, attention_mask, labels_batch = [t.to(device) for t in batch]
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = dept_model(input_ids, attention_mask)
                
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
        
        dept_model.eval()
        
        test_dataloader = self._create_dataloader(
            test_inputs, test_masks,
            batch_size=16,
            shuffle=False
        )
        
        dept_preds = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, attention_mask = [t.to(device) for t in batch]
                logits = dept_model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                dept_preds.extend(preds.cpu().numpy())
        
        # Calculate metrics
        dept_accuracy = accuracy_score(test_dept_encoded, dept_preds)
        dept_f1 = f1_score(test_dept_encoded, dept_preds, average='weighted', zero_division=0)
        
        self.logger.info(f"✅ Level 2 Accuracy: {dept_accuracy:.4f}")
        self.logger.info(f"✅ Level 2 F1-Score: {dept_f1:.4f}")
        
        # Save model and encoder
        dept_model.set_encoder(dept_encoder)
        dept_model.save(str(self.department_model_dir))
        
        # Update instance model
        self.department_model = dept_model
        
        # Save metrics to database
        decoded_dept_true = dept_encoder.inverse_transform(test_dept_encoded)
        decoded_dept_pred = dept_encoder.inverse_transform(dept_preds)
        
        run_id = self.save_metrics_to_sqlite(
            2, dept_accuracy, dept_f1,
            len(train_df), len(test_df)
        )
        
        self.save_full_classification_report_to_sqlite(run_id, "department", decoded_dept_true, decoded_dept_pred)
    
    def _train_level3(self, train_df, test_df):
        """Train Level 3: QueryType (conditioned on MasterDepartment + Department)"""
        
        from sklearn.preprocessing import LabelEncoder
        
        # Encode querytype labels
        qt_encoder = LabelEncoder()
        
        train_qt_encoded = qt_encoder.fit_transform(train_df['querytype'])
        test_qt_encoded = qt_encoder.transform(test_df['querytype'])
        
        self.logger.info(f"📊 QueryType classes: {len(qt_encoder.classes_)}")
        
        # Initialize model
        num_qt = len(qt_encoder.classes_)
        device = self._get_device()
        
        qt_model = QueryTypeNNModel(
            self.baseline_model_name,
            num_labels=num_qt
        )
        qt_model.to(device)
        
        # Create conditional texts (prepend MasterDepartment + Department)
        train_conditional_texts = [
            f"MasterDepartment: {md} Department: {dept} Email: {text}"
            for md, dept, text in zip(train_df['masterdepartment'], train_df['department'], train_df['text'])
        ]
        
        test_conditional_texts = [
            f"MasterDepartment: {md} Department: {dept} Email: {text}"
            for md, dept, text in zip(test_df['masterdepartment'], test_df['department'], test_df['text'])
        ]
        
        # Tokenize
        tokenizer = self.tokenizer
        MAX_LEN = 256
        
        train_inputs, train_masks = self._tokenize_and_pad(train_conditional_texts, tokenizer, MAX_LEN)
        test_inputs, test_masks = self._tokenize_and_pad(test_conditional_texts, tokenizer, MAX_LEN)
        
        # Create DataLoaders
        train_dataloader = self._create_dataloader(
            train_inputs, train_masks,
            labels=train_qt_encoded,
            batch_size=16,
            shuffle=True
        )
        
        # Training setup
        NUM_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 3))
        LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-5))
        
        optimizer = AdamW(qt_model.parameters(), lr=LEARNING_RATE, eps=1e-8)
        criterion = nn.CrossEntropyLoss()
        
        # Training logger
        training_logger = TrainingLogger(self.logger, NUM_EPOCHS, len(train_dataloader))
        training_logger.start_training("Level 3 (QueryType)")
        
        # Training loop
        for epoch in range(NUM_EPOCHS):
            training_logger.start_epoch(epoch)
            
            qt_model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                input_ids, attention_mask, labels_batch = [t.to(device) for t in batch]
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = qt_model(input_ids, attention_mask)
                
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
        
        qt_model.eval()
        
        test_dataloader = self._create_dataloader(
            test_inputs, test_masks,
            batch_size=16,
            shuffle=False
        )
        
        qt_preds = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, attention_mask = [t.to(device) for t in batch]
                logits = qt_model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                qt_preds.extend(preds.cpu().numpy())
        
        # Calculate metrics
        qt_accuracy = accuracy_score(test_qt_encoded, qt_preds)
        qt_f1 = f1_score(test_qt_encoded, qt_preds, average='weighted', zero_division=0)
        
        self.logger.info(f"✅ Level 3 Accuracy: {qt_accuracy:.4f}")
        self.logger.info(f"✅ Level 3 F1-Score: {qt_f1:.4f}")
        
        # Save model and encoder
        qt_model.set_encoder(qt_encoder)
        qt_model.save(str(self.querytype_model_dir))
        
        # Update instance model
        self.querytype_model = qt_model
        
        # Save metrics to database
        decoded_qt_true = qt_encoder.inverse_transform(test_qt_encoded)
        decoded_qt_pred = qt_encoder.inverse_transform(qt_preds)
        
        run_id = self.save_metrics_to_sqlite(
            3, qt_accuracy, qt_f1,
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
        1. Level 1: Email → MasterDepartment
        2. Level 2: Email + MasterDepartment → Department
        3. Level 3: Email + MasterDepartment + Department → QueryType
        
        Returns all predictions in a single result.
        """
        
        self.logger.info(f"🔍 Running inference on {len(texts)} email(s)...")

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

        device = self._get_device()

        if self.model is None:
            self.logger.error("❌ Level 1 model not loaded - cannot make predictions")
            return []

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
        global_text_idx = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = [t.to(device) for t in batch]
    
                # ========== LEVEL 1: MasterDepartment ==========
                masterdept_logits = self.model(input_ids, attention_mask)
                masterdept_probs = torch.softmax(masterdept_logits, dim=1)
                masterdept_preds = torch.argmax(masterdept_probs, dim=1)

                decoded_masterdept_preds = self._decode_predictions(
                    masterdept_preds, 'masterdepartment'
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

                    # ========== LEVEL 2: DEPARTMENT ==========
                    if self.department_model is not None:
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

                        if isinstance(dept_logits, (tuple, list)):
                            dept_logits = dept_logits[0]

                        dept_probs = torch.softmax(dept_logits, dim=1)
                        dept_idx = torch.argmax(dept_probs, dim=1).item()

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
                            conditional_text_qt = (
                                f"MasterDepartment: {decoded_masterdept_preds[i]} "
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

                            if isinstance(qt_logits, (tuple, list)):
                                qt_logits = qt_logits[0]

                            qt_probs = torch.softmax(qt_logits, dim=1)
                            qt_idx = torch.argmax(qt_probs, dim=1).item()

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
