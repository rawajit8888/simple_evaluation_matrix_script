import torch
import torch.nn as nn
from transformers import BertModel
import pickle
import os
import logging

logger = logging.getLogger(__name__)


class MultiTaskNNModel(nn.Module):
    """
    3-Level Hierarchical Multi-task BERT model with bundled label encoders.
    
    Architecture:
    - Level 1: Master Department (e.g., "Internet Banking")
    - Level 2: Sub-Category (e.g., "Internet Banking > Account Access")
    - Level 3: Specific Issue (e.g., "Internet Banking > Account Access > Unblock")
    - Sentiment: Parallel sentiment classification
    
    Saves encoders alongside model weights for risk-free deployment.
    """
    
    def __init__(self, modelname, lvl1_classes=50, lvl2_classes=100, lvl3_classes=150, sentimentlabel_length=3):
        super(MultiTaskNNModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(modelname)
        self.dropout = nn.Dropout(0.1)
        
        # ===== Hierarchical Classification Heads =====
        hidden_size = self.bert.config.hidden_size
        
        # Level 1: Master Department (Internet Banking, Credit Cards, etc.)
        self.lvl1_classifier = nn.Linear(hidden_size, lvl1_classes)
        
        # Level 2: Sub-Category (Internet Banking > Account Access, etc.)
        self.lvl2_classifier = nn.Linear(hidden_size, lvl2_classes)
        
        # Level 3: Specific Issue (Internet Banking > Account Access > Unblock, etc.)
        self.lvl3_classifier = nn.Linear(hidden_size, lvl3_classes)
        
        # ===== Sentiment Head =====
        self.sentiment_classifier = nn.Linear(hidden_size, sentimentlabel_length)
        
        # Softmax layers (optional, used for probabilities)
        self.lvl1_softmax = nn.Softmax(dim=1)
        self.lvl2_softmax = nn.Softmax(dim=1)
        self.lvl3_softmax = nn.Softmax(dim=1)
        self.sentiment_softmax = nn.Softmax(dim=1)
        
        # Store encoder info (will be set during training)
        self.encoders = {
            'lvl1': None,
            'lvl2': None,
            'lvl3': None,
            'sentiment': None
        }
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through BERT and all classification heads
        
        Returns:
            Tuple of (logits, logits, logits, logits, probs, probs, probs, probs)
            - lvl1_logits, lvl2_logits, lvl3_logits, sentiment_logits
            - lvl1_probs, lvl2_probs, lvl3_probs, sentiment_probs
        """
        # Get BERT embeddings
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # CLS token pooled output
        pooled_output = self.dropout(pooled_output)
        
        # Level 1: Master Department
        lvl1_logits = self.lvl1_classifier(pooled_output)
        lvl1_probs = self.lvl1_softmax(lvl1_logits)
        
        # Level 2: Sub-Category
        lvl2_logits = self.lvl2_classifier(pooled_output)
        lvl2_probs = self.lvl2_softmax(lvl2_logits)
        
        # Level 3: Specific Issue
        lvl3_logits = self.lvl3_classifier(pooled_output)
        lvl3_probs = self.lvl3_softmax(lvl3_logits)
        
        # Sentiment
        sentiment_logits = self.sentiment_classifier(pooled_output)
        sentiment_probs = self.sentiment_softmax(sentiment_logits)
        
        return (
            lvl1_logits, lvl2_logits, lvl3_logits, sentiment_logits,
            lvl1_probs, lvl2_probs, lvl3_probs, sentiment_probs
        )
    
    def set_encoders(self, lvl1_encoder, lvl2_encoder, lvl3_encoder, sentiment_encoder):
        """Set label encoders before saving"""
        self.encoders['lvl1'] = lvl1_encoder
        self.encoders['lvl2'] = lvl2_encoder
        self.encoders['lvl3'] = lvl3_encoder
        self.encoders['sentiment'] = sentiment_encoder
        logger.info("✅ All 4 encoders (lvl1, lvl2, lvl3, sentiment) attached to model")
    
    def SaveModel(self, directorypath):
        """Save model with bundled encoders"""
        try:
            os.makedirs(directorypath, exist_ok=True)
            
            # Save BERT backbone
            logger.info(f"💾 Saving BERT backbone to {directorypath}")
            self.bert.save_pretrained(directorypath)
            
            # Save model weights (all classifier heads)
            model_path = os.path.join(directorypath, 'multitask_model.pth')
            logger.info(f"💾 Saving model weights to {model_path}")
            torch.save(self.state_dict(), model_path)
            
            # Save encoders bundled with model
            encoders_path = os.path.join(directorypath, 'label_encoders.pkl')
            logger.info(f"💾 Saving label encoders to {encoders_path}")
            with open(encoders_path, 'wb') as f:
                pickle.dump(self.encoders, f)
            
            logger.info("✅ 3-Level Hierarchical Model and encoders saved successfully")
            
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
            
            # Filter compatible weights (important when class counts change)
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
                logger.info("✅ 3-Level Model and encoders loaded successfully")
            else:
                logger.warning("⚠️  No bundled encoders found - using external encoders")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def get_encoder(self, level_name):
        """
        Safely retrieve encoder for a level
        
        Args:
            level_name: 'lvl1', 'lvl2', 'lvl3', or 'sentiment'
        """
        if level_name not in self.encoders:
            raise ValueError(f"No encoder found for level: {level_name}")
        
        encoder = self.encoders[level_name]
        if encoder is None:
            raise ValueError(f"Encoder for {level_name} is None")
        
        return encoder
