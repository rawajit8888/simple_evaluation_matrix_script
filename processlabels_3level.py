import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import pickle
import logging

logger = logging.getLogger(__name__)


class ProcessLabels:
    """
    Label encoder manager for 3-level hierarchical learning.
    
    Handles:
    - masterdepartment (Level 1)
    - department (Level 2)
    - querytype (Level 3)
    
    Encoders are stored in a separate encoders/ directory as fallback,
    but models should bundle their own encoders for deployment safety.
    """
    
    def __init__(self, parsed_label_config, model_dir):
        self.parsed_label_config = parsed_label_config
        self.encoder_dir = os.path.join(model_dir, 'encoders')
        os.makedirs(self.encoder_dir, exist_ok=True)
        self.initEncoder()

    def initEncoder(self):
        """Initialize all label encoders for 3-level hierarchy"""
        self.labels = {
            "masterdepartment": {"encoder": None},  # Level 1
            "department": {"encoder": None},        # Level 2
            "querytype": {"encoder": None}          # Level 3
        }

        self.getLabelEncoder('masterdepartment')
        self.getLabelEncoder('department')
        self.getLabelEncoder('querytype')

    def saveLabelEncoderFile(self, labelencoder, filepath):
        """Save encoder to pickle file"""
        try:
            with open(filepath, 'wb') as file:
                pickle.dump(labelencoder, file)
            logger.info(f"💾 Saved encoder to {filepath}")
        except Exception as e:
            logger.error(f"❌ Error saving encoder to {filepath}: {e}")
            raise

    def getLabelEncoder(self, labeltype):
        """
        Get or create label encoder for a specific label type.
        
        Handles incremental learning: if new labels appear in parsed_label_config,
        they are added to the existing encoder.
        
        Args:
            labeltype: 'masterdepartment', 'department', or 'querytype'
        
        Returns:
            LabelEncoder instance
        """
        try:
            if self.labels[labeltype]["encoder"] is None:
                file_path = os.path.join(self.encoder_dir, f'{labeltype}_label_encoder.pkl')
                
                if os.path.exists(file_path):
                    # Load existing encoder
                    logger.info(f"📂 Loading existing {labeltype} encoder from {file_path}")
                    with open(file_path, 'rb') as file:
                        label_encoder = pickle.load(file)

                    # Check for new labels in config
                    if self.parsed_label_config is not None and labeltype in self.parsed_label_config:
                        config_labels = self.parsed_label_config[labeltype].get('labels', [])
                        new_labels = [
                            label for label in config_labels 
                            if label not in label_encoder.classes_.tolist()
                        ]
                        
                        if len(new_labels) > 0:
                            logger.info(f"🔧 Adding {len(new_labels)} new labels to {labeltype} encoder")
                            new_labels_np = np.array(new_labels, dtype=label_encoder.classes_.dtype)
                            updated_classes = np.concatenate((label_encoder.classes_, new_labels_np))
                            label_encoder.fit(updated_classes)
                            self.saveLabelEncoderFile(label_encoder, file_path)
                    
                else:
                    # Create new encoder
                    logger.info(f"🆕 Creating new {labeltype} encoder")
                    label_encoder = LabelEncoder()
                    
                    if self.parsed_label_config is not None and labeltype in self.parsed_label_config:
                        config_labels = self.parsed_label_config[labeltype].get('labels', [])
                        if len(config_labels) > 0:
                            label_encoder.fit(config_labels)
                            self.saveLabelEncoderFile(label_encoder, file_path)
                        else:
                            logger.warning(f"⚠️ No labels found in config for {labeltype}")
                    else:
                        logger.warning(f"⚠️ No config found for {labeltype} - encoder will be empty")
                
                self.labels[labeltype]["encoder"] = label_encoder

            return self.labels[labeltype]["encoder"]
            
        except Exception as e:
            logger.error(f"❌ Error in getLabelEncoder for {labeltype}: {e}")
            raise

    def __getitem__(self, key):
        """Make the class subscriptable - returns the encoder directly"""
        if key not in self.labels:
            raise KeyError(f"Label type '{key}' not found. Available: {list(self.labels.keys())}")
        return self.labels[key]["encoder"]
    
    def fit_encoder(self, labeltype, labels):
        """
        Fit an encoder with actual labels from data.
        
        This is called during training when labels are available.
        
        Args:
            labeltype: 'masterdepartment', 'department', or 'querytype'
            labels: list or array of labels
        """
        try:
            unique_labels = sorted(set(str(l) for l in labels))
            logger.info(f"🔧 Fitting {labeltype} encoder with {len(unique_labels)} unique labels")
            
            encoder = LabelEncoder()
            encoder.fit(unique_labels)
            
            self.labels[labeltype]["encoder"] = encoder
            
            # Save to file
            file_path = os.path.join(self.encoder_dir, f'{labeltype}_label_encoder.pkl')
            self.saveLabelEncoderFile(encoder, file_path)
            
            logger.info(f"✅ {labeltype} encoder fitted and saved")
            logger.info(f"   Classes: {list(encoder.classes_)}")
            
        except Exception as e:
            logger.error(f"❌ Error fitting {labeltype} encoder: {e}")
            raise
