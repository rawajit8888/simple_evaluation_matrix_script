import torch
import torch.nn as nn
from transformers import BertModel
import pickle
import os
import logging

logger = logging.getLogger(__name__)


class MultiTaskNNModel(nn.Module):
    """
    Multi-task BERT model with bundled label encoders for 3-level hierarchy.
    Level 1: Predicts MasterDepartment + Sentiment
    Saves encoders alongside model weights for risk-free deployment.
    """
    
    def __init__(self, modelname, classificationlabel_length=1000, sentimentlabel_length=2):
        super(MultiTaskNNModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(modelname)
        self.dropout = nn.Dropout(0.1)
        
        # Level 1 classifiers
        self.classifier = nn.Linear(self.bert.config.hidden_size, classificationlabel_length)
        self.sentiment_classifier = nn.Linear(self.bert.config.hidden_size, sentimentlabel_length)
        
        self.classifier_softmax = nn.Softmax(dim=1)
        self.sentiment_softmax = nn.Softmax(dim=1)
        
        # Store encoder info (will be set during training)
        # For 3-level hierarchy: masterdepartment (Level 1) + sentiment
        self.encoders = {
            'classification': None,  # MasterDepartment encoder
            'sentiment': None,        # Sentiment encoder
            'masterdepartment': None  # Alias for clarity
        }
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        
        classifier_logits = self.classifier(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)
        
        classifier_probs = self.classifier_softmax(classifier_logits)
        sentiment_probs = self.sentiment_softmax(sentiment_logits)
        
        return classifier_logits, sentiment_logits, classifier_probs, sentiment_probs
    
    def set_encoders(self, masterdepartment_encoder, sentiment_encoder):
        """Set label encoders before saving"""
        self.encoders['classification'] = masterdepartment_encoder
        self.encoders['masterdepartment'] = masterdepartment_encoder  # Alias
        self.encoders['sentiment'] = sentiment_encoder
        logger.info("✓ Encoders attached to model")
    
    def SaveModel(self, directorypath):
        """Save model with bundled encoders"""
        try:
            os.makedirs(directorypath, exist_ok=True)
            
            # Save BERT backbone
            logger.info(f"📦 Saving BERT backbone to {directorypath}")
            self.bert.save_pretrained(directorypath)
            
            # Save model weights
            model_path = os.path.join(directorypath, 'multitask_model.pth')
            logger.info(f"💾 Saving model weights to {model_path}")
            torch.save(self.state_dict(), model_path)
            
            # Save encoders bundled with model
            encoders_path = os.path.join(directorypath, 'label_encoders.pkl')
            logger.info(f"💾 Saving label encoders to {encoders_path}")
            with open(encoders_path, 'wb') as f:
                pickle.dump(self.encoders, f)
          
            logger.info("✅ Model and encoders saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            raise
    
    def LoadModel(self, directorypath):
        """Load model with bundled encoders"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model weights
            model_path = os.path.join(directorypath, 'multitask_model.pth')
            logger.info(f"📂 Loading model weights from {model_path}")
            
            loaded_state = torch.load(model_path, map_location=device)
            current_state = self.state_dict()
            
            # Filter compatible weights
            filtered_state = {
                k: v for k, v in loaded_state.items() 
                if k in current_state and v.size() == current_state[k].size()
            }
            
            current_state.update(filtered_state)
            self.load_state_dict(current_state, strict=False)
            self.eval()
            
            # Load bundled encoders
            encoders_path = os.path.join(directorypath, 'label_encoders.pkl')
            if os.path.exists(encoders_path):
                logger.info(f"📂 Loading bundled encoders from {encoders_path}")
                with open(encoders_path, 'rb') as f:
                    self.encoders = pickle.load(f)
                
                # Ensure both aliases are set
                if 'classification' in self.encoders and 'masterdepartment' not in self.encoders:
                    self.encoders['masterdepartment'] = self.encoders['classification']
                elif 'masterdepartment' in self.encoders and 'classification' not in self.encoders:
                    self.encoders['classification'] = self.encoders['masterdepartment']
                
                logger.info("✅ Model and encoders loaded successfully")
            else:
                logger.warning("⚠️  No bundled encoders found - using external encoders")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def get_encoder(self, task_name):
        """Safely retrieve encoder for a task"""
        if task_name not in self.encoders:
            raise ValueError(f"No encoder found for task: {task_name}")
        
        encoder = self.encoders[task_name]
        if encoder is None:
            raise ValueError(f"Encoder for {task_name} is None")
        
        return encoder
