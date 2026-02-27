import torch
import torch.nn as nn
from transformers import BertModel
import pickle
import os
import logging

logger = logging.getLogger(__name__)


class MultiTaskNNModel(nn.Module):
    """
    Multi-task BERT model with bundled label encoders.
    Saves encoders alongside model weights for risk-free deployment.
    """
    
    def __init__(self, modelname, classificationlabel_length=1000, sentimentlabel_length=2):
        super(MultiTaskNNModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(modelname)
        self.dropout = nn.Dropout(0.1)
        
        self.classifier = nn.Linear(self.bert.config.hidden_size, classificationlabel_length)
        self.sentiment_classifier = nn.Linear(self.bert.config.hidden_size, sentimentlabel_length)
        
        self.classifier_softmax = nn.Softmax(dim=1)
        self.sentiment_softmax = nn.Softmax(dim=1)
        
        # Store encoder info (will be set during training)
        self.encoders = {
            'classification': None,
            'sentiment': None
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
    
    def set_encoders(self, classification_encoder, sentiment_encoder):
        """Set label encoders before saving"""
        self.encoders['classification'] = classification_encoder
        self.encoders['sentiment'] = sentiment_encoder
        logger.info(" Encoders attached to model")
    
    def SaveModel(self, directorypath):
        """Save model with bundled encoders"""
        try:
            os.makedirs(directorypath, exist_ok=True)
            
            # Save BERT backbone
            logger.info(f" Saving BERT backbone to {directorypath}")
            self.bert.save_pretrained(directorypath)
            
            # Save model weights
            model_path = os.path.join(directorypath, 'multitask_model.pth')
            logger.info(f" Saving model weights to {model_path}")
            torch.save(self.state_dict(), model_path)
            
            # Save encoders bundled with model
            encoders_path = os.path.join(directorypath, 'label_encoders.pkl')
            logger.info(f" Saving label encoders to {encoders_path}")
            with open(encoders_path, 'wb') as f:
                pickle.dump(self.encoders, f)
          
            logger.info(" Model and encoders saved successfully")
            
        except Exception as e:
            logger.error(f" Error saving model: {e}")
            raise
    
    def LoadModel(self, directorypath):
        """Load model with bundled encoders"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model weights
            model_path = os.path.join(directorypath, 'multitask_model.pth')
            logger.info(f" Loading model weights from {model_path}")
            
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
                logger.info(f" Loading bundled encoders from {encoders_path}")
                with open(encoders_path, 'rb') as f:
                    self.encoders = pickle.load(f)
                logger.info(" Model and encoders loaded successfully")
            else:
                logger.warning("  No bundled encoders found - using external encoders")
            
        except Exception as e:
            logger.error(f" Error loading model: {e}")
            raise
    
    def get_encoder(self, task_name):
        """Safely retrieve encoder for a task"""
        if task_name not in self.encoders:
            raise ValueError(f"No encoder found for task: {task_name}")
        
        encoder = self.encoders[task_name]
        if encoder is None:
            raise ValueError(f"Encoder for {task_name} is None")
        
        return encoder

    def predict_top3(self, input_ids, attention_mask, top_k=3):
        """
        Run inference and return top-K predictions with confidence scores.

        Returns a dict with:
          - 'classification': list of top_k dicts, each with 'label' and 'confidence'
                              Label format: "internet banking > account access > unblock"
          - 'sentiment':      list of top_k dicts, each with 'label' and 'confidence'

        Example output:
        {
            'classification': [
                {'rank': 1, 'label': 'internet banking > account access > unblock', 'confidence': 0.82},
                {'rank': 2, 'label': 'internet banking > account access > login issue', 'confidence': 0.11},
                {'rank': 3, 'label': 'internet banking > password reset',             'confidence': 0.05},
            ],
            'sentiment': [
                {'rank': 1, 'label': 'negative', 'confidence': 0.91},
                {'rank': 2, 'label': 'positive', 'confidence': 0.09},
            ]
        }
        """
        self.eval()
        with torch.no_grad():
            _, _, classifier_probs, sentiment_probs = self.forward(input_ids, attention_mask)

        def _decode_top_k(probs, task_name, k):
            """Extract top-k labels and their confidence scores for a task."""
            encoder = self.get_encoder(task_name)
            k = min(k, probs.shape[1])  # guard against k > num_classes

            # topk over class dimension → shape (batch, k)
            top_probs, top_indices = torch.topk(probs, k, dim=1)

            results = []
            for rank in range(k):
                idx = top_indices[0, rank].item()          # batch index 0
                confidence = top_probs[0, rank].item()

                # LabelEncoder stores classes in .classes_ array
                label = encoder.classes_[idx]

                results.append({
                    'rank':       rank + 1,
                    'label':      label,
                    'confidence': round(confidence, 4)
                })
            return results

        return {
            'classification': _decode_top_k(classifier_probs, 'classification', top_k),
            'sentiment':      _decode_top_k(sentiment_probs,  'sentiment',      top_k),
        }