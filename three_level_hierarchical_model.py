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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pickle

logger = logging.getLogger(__name__)


# ========== 3-LEVEL HIERARCHICAL MODELS WITH ENCODER BUNDLING ==========

class Level1Model(nn.Module):
    """
    Level 1: Email → MasterDepartment
    WITH BUNDLED ENCODER for deployment safety
    """
    
    def __init__(self, modelname, num_labels):
        super(Level1Model, self).__init__()
        
        self.bert = BertModel.from_pretrained(modelname)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            num_labels
        )
        
        # Bundled encoder
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
        logger.info("✅ Encoder attached to Level 1 (MasterDepartment) model")
    
    def save(self, directorypath):
        """Save model with bundled encoder"""
        try:
            os.makedirs(directorypath, exist_ok=True)
            
            # Save BERT backbone
            logger.info(f"💾 Saving Level 1 BERT to {directorypath}")
            self.bert.save_pretrained(directorypath)
            
            # Save classifier head
            classifier_path = os.path.join(directorypath, "level1_classifier.pth")
            logger.info(f"💾 Saving Level 1 classifier head to {classifier_path}")
            torch.save(self.classifier.state_dict(), classifier_path)
            
            # Save bundled encoder
            if self.encoder is not None:
                encoder_path = os.path.join(directorypath, 'level1_encoder.pkl')
                logger.info(f"💾 Saving Level 1 bundled encoder to {encoder_path}")
                with open(encoder_path, 'wb') as f:
                    pickle.dump(self.encoder, f)
                logger.info(f"   → Encoder classes: {list(self.encoder.classes_)}")
            else:
                logger.warning("⚠️ Level 1 encoder is None - not saving")
            
            logger.info("✅ Level 1 model and encoder saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving Level 1 model: {e}")
            raise
    
    def load(self, directorypath):
        """Load model with bundled encoder"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load BERT
            logger.info(f"📂 Loading Level 1 BERT from {directorypath}")
            self.bert = BertModel.from_pretrained(directorypath)
            
            # Load classifier head
            classifier_path = os.path.join(directorypath, "level1_classifier.pth")
            logger.info(f"📂 Loading Level 1 classifier head from {classifier_path}")
            self.classifier.load_state_dict(
                torch.load(classifier_path, map_location=device)
            )
            
            # Load bundled encoder
            encoder_path = os.path.join(directorypath, 'level1_encoder.pkl')
            if os.path.exists(encoder_path):
                logger.info(f"📂 Loading bundled Level 1 encoder from {encoder_path}")
                with open(encoder_path, 'rb') as f:
                    self.encoder = pickle.load(f)
                logger.info(f"   → Encoder classes: {list(self.encoder.classes_)}")
                logger.info("✅ Level 1 model and encoder loaded successfully")
            else:
                logger.warning("⚠️ No bundled encoder found for Level 1 - will use fallback")
            
            self.eval()
            
        except Exception as e:
            logger.error(f"❌ Error loading Level 1 model: {e}")
            raise
    
    def get_encoder(self):
        """Safely retrieve encoder"""
        if self.encoder is None:
            raise ValueError("Level 1 encoder is None - use fallback")
        return self.encoder


class Level2Model(nn.Module):
    """
    Level 2: Email + MasterDepartment → Department
    WITH BUNDLED ENCODER for deployment safety
    """
    
    def __init__(self, modelname, num_labels):
        super(Level2Model, self).__init__()
        
        self.bert = BertModel.from_pretrained(modelname)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            num_labels
        )
        
        # Bundled encoder
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
        logger.info("✅ Encoder attached to Level 2 (Department) model")
    
    def save(self, directorypath):
        """Save model with bundled encoder"""
        try:
            os.makedirs(directorypath, exist_ok=True)
            
            # Save BERT backbone
            logger.info(f"💾 Saving Level 2 BERT to {directorypath}")
            self.bert.save_pretrained(directorypath)
            
            # Save classifier head
            classifier_path = os.path.join(directorypath, "level2_classifier.pth")
            logger.info(f"💾 Saving Level 2 classifier head to {classifier_path}")
            torch.save(self.classifier.state_dict(), classifier_path)
            
            # Save bundled encoder
            if self.encoder is not None:
                encoder_path = os.path.join(directorypath, 'level2_encoder.pkl')
                logger.info(f"💾 Saving Level 2 bundled encoder to {encoder_path}")
                with open(encoder_path, 'wb') as f:
                    pickle.dump(self.encoder, f)
                logger.info(f"   → Encoder classes: {list(self.encoder.classes_)}")
            else:
                logger.warning("⚠️ Level 2 encoder is None - not saving")
            
            logger.info("✅ Level 2 model and encoder saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving Level 2 model: {e}")
            raise
    
    def load(self, directorypath):
        """Load model with bundled encoder"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load BERT
            logger.info(f"📂 Loading Level 2 BERT from {directorypath}")
            self.bert = BertModel.from_pretrained(directorypath)
            
            # Load classifier head
            classifier_path = os.path.join(directorypath, "level2_classifier.pth")
            logger.info(f"📂 Loading Level 2 classifier head from {classifier_path}")
            self.classifier.load_state_dict(
                torch.load(classifier_path, map_location=device)
            )
            
            # Load bundled encoder
            encoder_path = os.path.join(directorypath, 'level2_encoder.pkl')
            if os.path.exists(encoder_path):
                logger.info(f"📂 Loading bundled Level 2 encoder from {encoder_path}")
                with open(encoder_path, 'rb') as f:
                    self.encoder = pickle.load(f)
                logger.info(f"   → Encoder classes: {list(self.encoder.classes_)}")
                logger.info("✅ Level 2 model and encoder loaded successfully")
            else:
                logger.warning("⚠️ No bundled encoder found for Level 2 - will use fallback")
            
            self.eval()
            
        except Exception as e:
            logger.error(f"❌ Error loading Level 2 model: {e}")
            raise
    
    def get_encoder(self):
        """Safely retrieve encoder"""
        if self.encoder is None:
            raise ValueError("Level 2 encoder is None - use fallback")
        return self.encoder


class Level3Model(nn.Module):
    """
    Level 3: Email + MasterDepartment + Department → QueryType
    WITH BUNDLED ENCODER for deployment safety
    """
    
    def __init__(self, modelname, num_labels):
        super(Level3Model, self).__init__()
        
        self.bert = BertModel.from_pretrained(modelname)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            num_labels
        )
        
        # Bundled encoder
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
        logger.info("✅ Encoder attached to Level 3 (QueryType) model")
    
    def save(self, directorypath):
        """Save model with bundled encoder"""
        try:
            os.makedirs(directorypath, exist_ok=True)
            
            # Save BERT backbone
            logger.info(f"💾 Saving Level 3 BERT to {directorypath}")
            self.bert.save_pretrained(directorypath)
            
            # Save classifier head
            classifier_path = os.path.join(directorypath, "level3_classifier.pth")
            logger.info(f"💾 Saving Level 3 classifier head to {classifier_path}")
            torch.save(self.classifier.state_dict(), classifier_path)
            
            # Save bundled encoder
            if self.encoder is not None:
                encoder_path = os.path.join(directorypath, 'level3_encoder.pkl')
                logger.info(f"💾 Saving Level 3 bundled encoder to {encoder_path}")
                with open(encoder_path, 'wb') as f:
                    pickle.dump(self.encoder, f)
                logger.info(f"   → Encoder classes: {list(self.encoder.classes_)}")
            else:
                logger.warning("⚠️ Level 3 encoder is None - not saving")
            
            logger.info("✅ Level 3 model and encoder saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving Level 3 model: {e}")
            raise
    
    def load(self, directorypath):
        """Load model with bundled encoder"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load BERT
            logger.info(f"📂 Loading Level 3 BERT from {directorypath}")
            self.bert = BertModel.from_pretrained(directorypath)
            
            # Load classifier head
            classifier_path = os.path.join(directorypath, "level3_classifier.pth")
            logger.info(f"📂 Loading Level 3 classifier head from {classifier_path}")
            self.classifier.load_state_dict(
                torch.load(classifier_path, map_location=device)
            )
            
            # Load bundled encoder
            encoder_path = os.path.join(directorypath, 'level3_encoder.pkl')
            if os.path.exists(encoder_path):
                logger.info(f"📂 Loading bundled Level 3 encoder from {encoder_path}")
                with open(encoder_path, 'rb') as f:
                    self.encoder = pickle.load(f)
                logger.info(f"   → Encoder classes: {list(self.encoder.classes_)}")
                logger.info("✅ Level 3 model and encoder loaded successfully")
            else:
                logger.warning("⚠️ No bundled encoder found for Level 3 - will use fallback")
            
            self.eval()
            
        except Exception as e:
            logger.error(f"❌ Error loading Level 3 model: {e}")
            raise
    
    def get_encoder(self):
        """Safely retrieve encoder"""
        if self.encoder is None:
            raise ValueError("Level 3 encoder is None - use fallback")
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
        self.logger.info(f"📍 EPOCH {epoch + 1}/{self.total_epochs}")
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
        self.logger.info(f"✅ Epoch {epoch + 1} Complete")
        self.logger.info(f"⏱️  Epoch Time: {timedelta(seconds=int(epoch_time))}")
        self.logger.info(f"📉 Average Loss: {avg_loss:.4f}")
        self.logger.info(f"⏳ Estimated Time Remaining: {eta}")
        
    def end_training(self):
        """Mark training end"""
        total_time = time.time() - self.start_time
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🎉 TRAINING COMPLETED")
        self.logger.info(f"⏱️  Total Training Time: {timedelta(seconds=int(total_time))}")
        self.logger.info(f"⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)


class ThreeLevelHierarchicalModel:
    """
    3-LEVEL HIERARCHICAL BERT CLASSIFICATION MODEL
    
    Architecture:
    - Level 1: Email → MasterDepartment
    - Level 2: Email + MasterDepartment → Department
    - Level 3: Email + MasterDepartment + Department → QueryType
    
    Features:
    - Bundled encoders with models (risk-free deployment)
    - Comprehensive logging with time estimates
    - Automatic hierarchical inference routing
    - Stratified train/test split
    - SQLite metrics tracking
    """

    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', '830eb6d65978e36293a63635717da95bbbcb7a9d')
    START_TRAINING_EACH_N_UPDATES = int(os.getenv('START_TRAINING_EACH_N_UPDATES', 10))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 2e-5))
    NUM_TRAIN_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 3))
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', 0.01))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 16))
    MAX_LEN = int(os.getenv('MAX_LEN', 256))

    def __init__(self, config, logger, label_interface=None, parsed_label_config=None):
        self.config = config
        self.logger = logger
        self.label_interface = label_interface
        self.parsed_label_config = parsed_label_config
        
        # Initialize label processors
        self.processed_label_encoders = ProcessLabels(
            self.parsed_label_config,
            pathlib.Path(self.config['MODEL_DIR'])
        )
        
        # Models will be loaded on reload_model()
        self.level1_model = None
        self.level2_model = None
        self.level3_model = None
        self.tokenizer = None
        
        # Encoders (fallback from ProcessLabels)
        self.level1_encoder = None
        self.level2_encoder = None
        self.level3_encoder = None

    def reload_model(self):
        """
        Load all 3 hierarchical models with bundled encoders.
        Falls back to ProcessLabels encoders for backward compatibility.
        """
        self.logger.info("🔄 Reloading 3-level hierarchical models...")
        
        # Model paths
        level1_path = str(pathlib.Path(self.config['MODEL_DIR']) / 'level1_masterdepartment')
        level2_path = str(pathlib.Path(self.config['MODEL_DIR']) / 'level2_department')
        level3_path = str(pathlib.Path(self.config['MODEL_DIR']) / 'level3_querytype')
        
        # Tokenizer path (shared across all levels)
        tokenizer_path = level1_path if os.path.exists(level1_path) else self.config.get('BASELINE_MODEL_NAME', 'bert-base-uncased')
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.logger.info(f"✅ Tokenizer loaded from {tokenizer_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load tokenizer from {tokenizer_path}: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.logger.info("✅ Loaded default BERT tokenizer")
        
        # ========== LEVEL 1: MasterDepartment ==========
        if os.path.exists(level1_path):
            try:
                # Try to load bundled encoder to get label count
                bundled_encoder_path = os.path.join(level1_path, 'level1_encoder.pkl')
                if os.path.exists(bundled_encoder_path):
                    with open(bundled_encoder_path, 'rb') as f:
                        temp_encoder = pickle.load(f)
                    num_labels_l1 = len(temp_encoder.classes_)
                    self.logger.info(f"📊 Level 1 bundled encoder: {num_labels_l1} classes")
                else:
                    # Fallback to ProcessLabels
                    num_labels_l1 = 10  # Default
                    self.logger.info(f"📊 Level 1 default: {num_labels_l1} classes")
                
                self.logger.info(f"📂 Loading Level 1 model from {level1_path}")
                self.level1_model = Level1Model(level1_path, num_labels_l1)
                self.level1_model.load(level1_path)
                
                # Try to use bundled encoder
                try:
                    self.level1_encoder = self.level1_model.get_encoder()
                    self.logger.info("✅ Using bundled Level 1 encoder")
                except ValueError:
                    self.logger.warning("⚠️ No bundled Level 1 encoder - using fallback")
                    # Create fallback encoder if needed
                    if self.level1_encoder is None:
                        from sklearn.preprocessing import LabelEncoder
                        self.level1_encoder = LabelEncoder()
                
                self.logger.info("✅ Level 1 model loaded")
                
            except Exception as e:
                self.logger.error(f"❌ Error loading Level 1 model: {e}")
                self.level1_model = None
        else:
            self.logger.info("ℹ️  No Level 1 model found")
            self.level1_model = None
        
        # ========== LEVEL 2: Department ==========
        if os.path.exists(level2_path):
            try:
                # Try to load bundled encoder to get label count
                bundled_encoder_path = os.path.join(level2_path, 'level2_encoder.pkl')
                if os.path.exists(bundled_encoder_path):
                    with open(bundled_encoder_path, 'rb') as f:
                        temp_encoder = pickle.load(f)
                    num_labels_l2 = len(temp_encoder.classes_)
                    self.logger.info(f"📊 Level 2 bundled encoder: {num_labels_l2} classes")
                else:
                    num_labels_l2 = 20  # Default
                    self.logger.info(f"📊 Level 2 default: {num_labels_l2} classes")
                
                self.logger.info(f"📂 Loading Level 2 model from {level2_path}")
                self.level2_model = Level2Model(level2_path, num_labels_l2)
                self.level2_model.load(level2_path)
                
                # Try to use bundled encoder
                try:
                    self.level2_encoder = self.level2_model.get_encoder()
                    self.logger.info("✅ Using bundled Level 2 encoder")
                except ValueError:
                    self.logger.warning("⚠️ No bundled Level 2 encoder - using fallback")
                    if self.level2_encoder is None:
                        from sklearn.preprocessing import LabelEncoder
                        self.level2_encoder = LabelEncoder()
                
                self.logger.info("✅ Level 2 model loaded")
                
            except Exception as e:
                self.logger.error(f"❌ Error loading Level 2 model: {e}")
                self.level2_model = None
        else:
            self.logger.info("ℹ️  No Level 2 model found")
            self.level2_model = None
        
        # ========== LEVEL 3: QueryType ==========
        if os.path.exists(level3_path):
            try:
                # Try to load bundled encoder to get label count
                bundled_encoder_path = os.path.join(level3_path, 'level3_encoder.pkl')
                if os.path.exists(bundled_encoder_path):
                    with open(bundled_encoder_path, 'rb') as f:
                        temp_encoder = pickle.load(f)
                    num_labels_l3 = len(temp_encoder.classes_)
                    self.logger.info(f"📊 Level 3 bundled encoder: {num_labels_l3} classes")
                else:
                    num_labels_l3 = 50  # Default
                    self.logger.info(f"📊 Level 3 default: {num_labels_l3} classes")
                
                self.logger.info(f"📂 Loading Level 3 model from {level3_path}")
                self.level3_model = Level3Model(level3_path, num_labels_l3)
                self.level3_model.load(level3_path)
                
                # Try to use bundled encoder
                try:
                    self.level3_encoder = self.level3_model.get_encoder()
                    self.logger.info("✅ Using bundled Level 3 encoder")
                except ValueError:
                    self.logger.warning("⚠️ No bundled Level 3 encoder - using fallback")
                    if self.level3_encoder is None:
                        from sklearn.preprocessing import LabelEncoder
                        self.level3_encoder = LabelEncoder()
                
                self.logger.info("✅ Level 3 model loaded")
                
            except Exception as e:
                self.logger.error(f"❌ Error loading Level 3 model: {e}")
                self.level3_model = None
        else:
            self.logger.info("ℹ️  No Level 3 model found")
            self.level3_model = None
        
        self.logger.info("🎯 3-level model reload complete")

    def prepare_3level_data(self, train_df):
        """
        Prepare training data for all three levels
        
        Expected columns in train_df:
        - text: email content
        - masterdepartment: Level 1 target
        - department: Level 2 target
        - querytype: Level 3 target
        
        Returns:
            Prepared DataFrame with encoded labels and encoders
        """
        from sklearn.preprocessing import LabelEncoder
        
        self.logger.info("📊 Preparing 3-level training data...")
        
        # Level 1: Encode MasterDepartment
        level1_encoder = LabelEncoder()
        train_df['masterdepartment_encoded'] = level1_encoder.fit_transform(
            train_df['masterdepartment'].astype(str)
        )
        
        # Level 2: Encode Department
        level2_encoder = LabelEncoder()
        train_df['department_encoded'] = level2_encoder.fit_transform(
            train_df['department'].astype(str)
        )
        
        # Level 3: Encode QueryType
        level3_encoder = LabelEncoder()
        train_df['querytype_encoded'] = level3_encoder.fit_transform(
            train_df['querytype'].astype(str)
        )
        
        self.logger.info(f"✅ Level 1 (MasterDepartment): {len(level1_encoder.classes_)} classes")
        self.logger.info(f"   Classes: {list(level1_encoder.classes_)}")
        self.logger.info(f"✅ Level 2 (Department): {len(level2_encoder.classes_)} classes")
        self.logger.info(f"   Classes: {list(level2_encoder.classes_)}")
        self.logger.info(f"✅ Level 3 (QueryType): {len(level3_encoder.classes_)} classes")
        self.logger.info(f"   Classes: {list(level3_encoder.classes_)}")
        
        # Store encoders
        self.level1_encoder = level1_encoder
        self.level2_encoder = level2_encoder
        self.level3_encoder = level3_encoder
        
        return train_df, level1_encoder, level2_encoder, level3_encoder
    
    def create_dataloader(self, texts, labels, batch_size=16, shuffle=True):
        """Create PyTorch DataLoader for training"""
        
        # Tokenize texts
        tokenized_texts = [
            self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=self.MAX_LEN,
                truncation=True
            )
            for text in texts
        ]
        
        attention_masks = [[int(t > 0) for t in ids] for ids in tokenized_texts]
        
        # Pad sequences
        inputs = pad_sequences(
            tokenized_texts,
            maxlen=self.MAX_LEN,
            dtype='long',
            value=0,
            truncating='post',
            padding='post'
        )
        
        masks = pad_sequences(
            attention_masks,
            maxlen=self.MAX_LEN,
            dtype='long',
            value=0,
            truncating='post',
            padding='post'
        )
        
        # Create tensors
        dataset = TensorDataset(
            torch.tensor(inputs),
            torch.tensor(masks),
            torch.tensor(labels, dtype=torch.long)
        )
        
        # Create dataloader
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size
        )
        
        return dataloader
    
    def train_single_level(self, model, dataloader, level_name, epochs=3):
        """
        Train a single level model
        
        Args:
            model: The model to train (Level1Model, Level2Model, or Level3Model)
            dataloader: Training data
            level_name: Name for logging (e.g., "Level 1 - MasterDepartment")
            epochs: Number of training epochs
        
        Returns:
            Trained model
        """
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Optimizer
        optimizer = AdamW(model.parameters(), lr=self.LEARNING_RATE, eps=1e-8)
        
        # Loss function
        loss_fn = nn.CrossEntropyLoss()
        
        # Training logger
        train_logger = TrainingLogger(
            self.logger,
            total_epochs=epochs,
            total_batches=len(dataloader)
        )
        
        train_logger.start_training(model_name=level_name)
        
        for epoch in range(epochs):
            train_logger.start_epoch(epoch)
            
            model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(dataloader):
                input_ids, attention_mask, labels = [t.to(device) for t in batch]
                
                # Forward pass
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                
                # Calculate loss
                loss = loss_fn(logits, labels)
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Log progress
                train_logger.log_batch(epoch, batch_idx, loss.item())
            
            # Calculate average loss
            avg_loss = total_loss / len(dataloader)
            train_logger.end_epoch(epoch, avg_loss)
        
        train_logger.end_training()
        
        return model

    def fit(self, event, data, tasks, **kwargs):
        """
        Train all 3 hierarchical levels sequentially.
        
        Expected task data format:
        {
            'data': {
                'text': 'email content...',
                'masterdepartment': 'Sales',
                'department': 'Enterprise Sales',
                'querytype': 'Pricing Inquiry'
            }
        }
        
        Args:
            event: Training event trigger
            data: Training data
            tasks: List of labeled tasks from Label Studio
        """
        
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            self.logger.info(f"Skip training: event {event} is not supported")
            return

        self.logger.info("=" * 80)
        self.logger.info("📚 3-LEVEL HIERARCHICAL DATA PREPARATION")
        self.logger.info("=" * 80)

        # Extract training data from tasks
        ds_raw = []
        
        for task in tasks:
            try:
                text = task['data'].get('text', '')
                masterdepartment = task['data'].get('masterdepartment', 'Unknown')
                department = task['data'].get('department', 'Unknown')
                querytype = task['data'].get('querytype', 'Unknown')
                
                if text and masterdepartment and department and querytype:
                    ds_raw.append([text, masterdepartment, department, querytype])
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Skipping task due to error: {e}")
                continue
        
        if len(ds_raw) == 0:
            self.logger.error("❌ No valid training data found!")
            return
        
        # Build dataframe
        df = pd.DataFrame(
            ds_raw,
            columns=["text", "masterdepartment", "department", "querytype"]
        )
        
        self.logger.info(f"✅ Extracted {len(df)} training samples")
        
        # Prepare encoders
        df, level1_encoder, level2_encoder, level3_encoder = self.prepare_3level_data(df)
        
        # Stratified 80/20 split
        self.logger.info("📊 Performing stratified train/test split...")
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        train_indices = []
        test_indices = []
        
        rng = np.random.default_rng(42)
        
        # Stratify by querytype (finest granularity)
        for qt, group in df.groupby("querytype"):
            idx = group.index.to_list()
            rng.shuffle(idx)
            n = len(idx)
            
            if n < 2:
                train_indices.extend(idx)
                continue
            
            n_train = int(np.floor(0.8 * n))
            if n_train == 0:
                n_train = 1
            
            train_indices.extend(idx[:n_train])
            test_indices.extend(idx[n_train:])
        
        # Fallback if no test samples
        if len(test_indices) == 0 and len(df) > 1:
            all_idx = list(df.index)
            rng.shuffle(all_idx)
            split_idx = int(len(all_idx) * 0.8)
            train_indices = all_idx[:split_idx]
            test_indices = all_idx[split_idx:]
        
        train_df = df.loc[train_indices].reset_index(drop=True)
        test_df = df.loc[test_indices].reset_index(drop=True)
        
        self.logger.info(f"✅ Train: {len(train_df)}, Test: {len(test_df)}")
        
        # Save raw test data
        eval_dir = os.path.join(self.config['MODEL_DIR'], "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        test_df.to_csv(os.path.join(eval_dir, "test_raw_3level.csv"), index=False)
        
        # ========== TRAIN LEVEL 1: Email → MasterDepartment ==========
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🎯 LEVEL 1 TRAINING: Email → MasterDepartment")
        self.logger.info("=" * 80)
        
        texts_l1 = train_df['text'].tolist()
        labels_l1 = train_df['masterdepartment_encoded'].values
        dataloader_l1 = self.create_dataloader(texts_l1, labels_l1, batch_size=self.BATCH_SIZE)
        
        num_labels_l1 = len(level1_encoder.classes_)
        base_model = self.config.get('BASELINE_MODEL_NAME', 'bert-base-uncased')
        level1_model = Level1Model(base_model, num_labels_l1)
        
        level1_model = self.train_single_level(
            level1_model,
            dataloader_l1,
            level_name="Level 1 - MasterDepartment",
            epochs=self.NUM_TRAIN_EPOCHS
        )
        
        # Save Level 1 with bundled encoder
        level1_model.set_encoder(level1_encoder)
        level1_save_path = str(pathlib.Path(self.config['MODEL_DIR']) / "level1_masterdepartment")
        level1_model.save(level1_save_path)
        self.tokenizer.save_pretrained(level1_save_path)
        self.logger.info(f"✅ Level 1 model saved to {level1_save_path}")
        
        # ========== TRAIN LEVEL 2: Email + MasterDepartment → Department ==========
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🎯 LEVEL 2 TRAINING: Email + MasterDepartment → Department")
        self.logger.info("=" * 80)
        
        texts_l2 = [
            f"MasterDepartment: {row['masterdepartment']} Email: {row['text']}"
            for _, row in train_df.iterrows()
        ]
        labels_l2 = train_df['department_encoded'].values
        dataloader_l2 = self.create_dataloader(texts_l2, labels_l2, batch_size=self.BATCH_SIZE)
        
        num_labels_l2 = len(level2_encoder.classes_)
        level2_model = Level2Model(base_model, num_labels_l2)
        
        level2_model = self.train_single_level(
            level2_model,
            dataloader_l2,
            level_name="Level 2 - Department",
            epochs=self.NUM_TRAIN_EPOCHS
        )
        
        # Save Level 2 with bundled encoder
        level2_model.set_encoder(level2_encoder)
        level2_save_path = str(pathlib.Path(self.config['MODEL_DIR']) / "level2_department")
        level2_model.save(level2_save_path)
        self.tokenizer.save_pretrained(level2_save_path)
        self.logger.info(f"✅ Level 2 model saved to {level2_save_path}")
        
        # ========== TRAIN LEVEL 3: Email + MasterDepartment + Department → QueryType ==========
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("🎯 LEVEL 3 TRAINING: Email + MasterDept + Dept → QueryType")
        self.logger.info("=" * 80)
        
        texts_l3 = [
            f"MasterDepartment: {row['masterdepartment']} "
            f"Department: {row['department']} "
            f"Email: {row['text']}"
            for _, row in train_df.iterrows()
        ]
        labels_l3 = train_df['querytype_encoded'].values
        dataloader_l3 = self.create_dataloader(texts_l3, labels_l3, batch_size=self.BATCH_SIZE)
        
        num_labels_l3 = len(level3_encoder.classes_)
        level3_model = Level3Model(base_model, num_labels_l3)
        
        level3_model = self.train_single_level(
            level3_model,
            dataloader_l3,
            level_name="Level 3 - QueryType",
            epochs=self.NUM_TRAIN_EPOCHS
        )
        
        # Save Level 3 with bundled encoder
        level3_model.set_encoder(level3_encoder)
        level3_save_path = str(pathlib.Path(self.config['MODEL_DIR']) / "level3_querytype")
        level3_model.save(level3_save_path)
        self.tokenizer.save_pretrained(level3_save_path)
        self.logger.info(f"✅ Level 3 model saved to {level3_save_path}")
        
        # ========== EVALUATION ==========
        if len(test_df) > 0:
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("📊 EVALUATION ON TEST SET")
            self.logger.info("=" * 80)
            
            self.evaluate_3level_models(
                level1_model, level2_model, level3_model,
                test_df, level1_encoder, level2_encoder, level3_encoder
            )
        
        self.logger.info("=" * 80)
        self.logger.info("✅ 3-LEVEL TRAINING COMPLETE!")
        self.logger.info("=" * 80)
        self.logger.info(f"📁 Level 1: {level1_save_path}")
        self.logger.info(f"📁 Level 2: {level2_save_path}")
        self.logger.info(f"📁 Level 3: {level3_save_path}")
        self.logger.info("=" * 80)
    
    def evaluate_3level_models(self, level1_model, level2_model, level3_model, 
                                 test_df, level1_encoder, level2_encoder, level3_encoder):
        """Evaluate all 3 levels on test set"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        level1_model.eval()
        level2_model.eval()
        level3_model.eval()
        
        level1_model.to(device)
        level2_model.to(device)
        level3_model.to(device)
        
        all_l1_true = []
        all_l1_pred = []
        all_l2_true = []
        all_l2_pred = []
        all_l3_true = []
        all_l3_pred = []
        
        with torch.no_grad():
            for _, row in test_df.iterrows():
                text = row['text']
                
                # ===== LEVEL 1: Predict MasterDepartment =====
                l1_input = self.tokenizer.encode(
                    text, add_special_tokens=True, max_length=self.MAX_LEN, truncation=True
                )
                l1_mask = [int(t > 0) for t in l1_input]
                
                l1_input_padded = pad_sequences(
                    [l1_input], maxlen=self.MAX_LEN, padding='post', truncating='post'
                )
                l1_mask_padded = pad_sequences(
                    [l1_mask], maxlen=self.MAX_LEN, padding='post', truncating='post'
                )
                
                l1_logits = level1_model(
                    torch.tensor(l1_input_padded).to(device),
                    torch.tensor(l1_mask_padded).to(device)
                )
                l1_pred_idx = torch.argmax(l1_logits, dim=1).item()
                l1_pred_label = level1_encoder.inverse_transform([l1_pred_idx])[0]
                
                all_l1_true.append(row['masterdepartment'])
                all_l1_pred.append(l1_pred_label)
                
                # ===== LEVEL 2: Predict Department =====
                l2_text = f"MasterDepartment: {l1_pred_label} Email: {text}"
                l2_input = self.tokenizer.encode(
                    l2_text, add_special_tokens=True, max_length=self.MAX_LEN, truncation=True
                )
                l2_mask = [int(t > 0) for t in l2_input]
                
                l2_input_padded = pad_sequences(
                    [l2_input], maxlen=self.MAX_LEN, padding='post', truncating='post'
                )
                l2_mask_padded = pad_sequences(
                    [l2_mask], maxlen=self.MAX_LEN, padding='post', truncating='post'
                )
                
                l2_logits = level2_model(
                    torch.tensor(l2_input_padded).to(device),
                    torch.tensor(l2_mask_padded).to(device)
                )
                l2_pred_idx = torch.argmax(l2_logits, dim=1).item()
                l2_pred_label = level2_encoder.inverse_transform([l2_pred_idx])[0]
                
                all_l2_true.append(row['department'])
                all_l2_pred.append(l2_pred_label)
                
                # ===== LEVEL 3: Predict QueryType =====
                l3_text = f"MasterDepartment: {l1_pred_label} Department: {l2_pred_label} Email: {text}"
                l3_input = self.tokenizer.encode(
                    l3_text, add_special_tokens=True, max_length=self.MAX_LEN, truncation=True
                )
                l3_mask = [int(t > 0) for t in l3_input]
                
                l3_input_padded = pad_sequences(
                    [l3_input], maxlen=self.MAX_LEN, padding='post', truncating='post'
                )
                l3_mask_padded = pad_sequences(
                    [l3_mask], maxlen=self.MAX_LEN, padding='post', truncating='post'
                )
                
                l3_logits = level3_model(
                    torch.tensor(l3_input_padded).to(device),
                    torch.tensor(l3_mask_padded).to(device)
                )
                l3_pred_idx = torch.argmax(l3_logits, dim=1).item()
                l3_pred_label = level3_encoder.inverse_transform([l3_pred_idx])[0]
                
                all_l3_true.append(row['querytype'])
                all_l3_pred.append(l3_pred_label)
        
        # Calculate metrics
        l1_acc = accuracy_score(all_l1_true, all_l1_pred)
        l1_f1 = f1_score(all_l1_true, all_l1_pred, average='weighted', zero_division=0)
        
        l2_acc = accuracy_score(all_l2_true, all_l2_pred)
        l2_f1 = f1_score(all_l2_true, all_l2_pred, average='weighted', zero_division=0)
        
        l3_acc = accuracy_score(all_l3_true, all_l3_pred)
        l3_f1 = f1_score(all_l3_true, all_l3_pred, average='weighted', zero_division=0)
        
        self.logger.info(f"✅ Level 1 (MasterDepartment) - Accuracy: {l1_acc:.4f}, F1: {l1_f1:.4f}")
        self.logger.info(f"✅ Level 2 (Department) - Accuracy: {l2_acc:.4f}, F1: {l2_f1:.4f}")
        self.logger.info(f"✅ Level 3 (QueryType) - Accuracy: {l3_acc:.4f}, F1: {l3_f1:.4f}")
        
        # Save predictions
        eval_dir = os.path.join(self.config['MODEL_DIR'], "evaluation")
        eval_df = pd.DataFrame({
            'text': test_df['text'],
            'masterdepartment_true': all_l1_true,
            'masterdepartment_pred': all_l1_pred,
            'department_true': all_l2_true,
            'department_pred': all_l2_pred,
            'querytype_true': all_l3_true,
            'querytype_pred': all_l3_pred
        })
        eval_df.to_csv(os.path.join(eval_dir, "test_predictions_3level.csv"), index=False)
        self.logger.info(f"💾 Saved predictions to test_predictions_3level.csv")

    def predict(self, tasks: List[Dict], texts: List[str], context: Optional[Dict] = None, **kwargs):
        """
        HIERARCHICAL INFERENCE PIPELINE
        
        For each email:
        1. Level 1: Email → MasterDepartment
        2. Level 2: Email + predicted MasterDepartment → Department
        3. Level 3: Email + predicted MasterDepartment + Department → QueryType
        
        Returns all 3 predictions.
        """
        
        self.logger.info(f"🔮 Running 3-level inference on {len(texts)} email(s)...")
        
        if self.level1_model is None or self.level2_model is None or self.level3_model is None:
            self.logger.error("❌ Models not loaded! Call reload_model() first.")
            return []
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.level1_model.eval()
        self.level2_model.eval()
        self.level3_model.eval()
        
        self.level1_model.to(device)
        self.level2_model.to(device)
        self.level3_model.to(device)
        
        predictions = []
        
        with torch.no_grad():
            for text_obj in texts:
                text = text_obj['text'] if isinstance(text_obj, dict) else text_obj
                
                # ===== LEVEL 1: Predict MasterDepartment =====
                l1_input = self.tokenizer.encode(
                    text, add_special_tokens=True, max_length=self.MAX_LEN, truncation=True
                )
                l1_mask = [int(t > 0) for t in l1_input]
                
                l1_input_padded = pad_sequences(
                    [l1_input], maxlen=self.MAX_LEN, padding='post', truncating='post'
                )
                l1_mask_padded = pad_sequences(
                    [l1_mask], maxlen=self.MAX_LEN, padding='post', truncating='post'
                )
                
                l1_logits = self.level1_model(
                    torch.tensor(l1_input_padded).to(device),
                    torch.tensor(l1_mask_padded).to(device)
                )
                l1_probs = torch.softmax(l1_logits, dim=1)
                l1_pred_idx = torch.argmax(l1_probs, dim=1).item()
                l1_pred_label = self.level1_encoder.inverse_transform([l1_pred_idx])[0]
                l1_score = l1_probs[0][l1_pred_idx].item()
                
                # ===== LEVEL 2: Predict Department =====
                l2_text = f"MasterDepartment: {l1_pred_label} Email: {text}"
                l2_input = self.tokenizer.encode(
                    l2_text, add_special_tokens=True, max_length=self.MAX_LEN, truncation=True
                )
                l2_mask = [int(t > 0) for t in l2_input]
                
                l2_input_padded = pad_sequences(
                    [l2_input], maxlen=self.MAX_LEN, padding='post', truncating='post'
                )
                l2_mask_padded = pad_sequences(
                    [l2_mask], maxlen=self.MAX_LEN, padding='post', truncating='post'
                )
                
                l2_logits = self.level2_model(
                    torch.tensor(l2_input_padded).to(device),
                    torch.tensor(l2_mask_padded).to(device)
                )
                l2_probs = torch.softmax(l2_logits, dim=1)
                l2_pred_idx = torch.argmax(l2_probs, dim=1).item()
                l2_pred_label = self.level2_encoder.inverse_transform([l2_pred_idx])[0]
                l2_score = l2_probs[0][l2_pred_idx].item()
                
                # ===== LEVEL 3: Predict QueryType =====
                l3_text = f"MasterDepartment: {l1_pred_label} Department: {l2_pred_label} Email: {text}"
                l3_input = self.tokenizer.encode(
                    l3_text, add_special_tokens=True, max_length=self.MAX_LEN, truncation=True
                )
                l3_mask = [int(t > 0) for t in l3_input]
                
                l3_input_padded = pad_sequences(
                    [l3_input], maxlen=self.MAX_LEN, padding='post', truncating='post'
                )
                l3_mask_padded = pad_sequences(
                    [l3_mask], maxlen=self.MAX_LEN, padding='post', truncating='post'
                )
                
                l3_logits = self.level3_model(
                    torch.tensor(l3_input_padded).to(device),
                    torch.tensor(l3_mask_padded).to(device)
                )
                l3_probs = torch.softmax(l3_logits, dim=1)
                l3_pred_idx = torch.argmax(l3_probs, dim=1).item()
                l3_pred_label = self.level3_encoder.inverse_transform([l3_pred_idx])[0]
                l3_score = l3_probs[0][l3_pred_idx].item()
                
                # Return all 3 predictions
                predictions.append({
                    "masterdepartment": l1_pred_label,
                    "masterdepartment_score": l1_score,
                    "department": l2_pred_label,
                    "department_score": l2_score,
                    "querytype": l3_pred_label,
                    "querytype_score": l3_score
                })
        
        self.logger.info(f"✅ 3-level inference complete")
        return predictions
