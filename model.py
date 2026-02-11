import logging

logger = logging.getLogger(__name__)
multi_classs_model = None

from flask import jsonify

import os
import pathlib
import re
import label_studio_sdk

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from multi_task_model import MultiTaskBertModel
from dataprocessing.processlabels import ProcessLabels

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv('MODEL_DIR', './results/bert-classification-sentiment')

Config = {
    "MULTI_TASK": {
        'MODEL_DIR': os.getenv('MODEL_DIR', MODEL_DIR),
        'FINETUNED_MODEL_NAME': os.getenv('FINETUNED_MULTICLASS_MODEL_NAME', 'finetuned_multitask_model'),
        'BASELINE_MODEL_NAME': os.getenv('BASELINE_MULTICLASS_MODEL_NAME', 'baseline-model')
    }
}


def reload_model():
    """Reload the global multi-class model"""
    global multi_classs_model
    if multi_classs_model is not None:
        logger.info("🔄 Reloading multi-task model...")
        multi_classs_model.reload_model()
        logger.info("✅ Model reloaded successfully")


class MultiTaskBert(LabelStudioMLBase):
    """
    Label Studio ML Backend for 3-Level Hierarchical Email Classification
    
    Features:
    - Level 1: Email → MasterDepartment + Sentiment
    - Level 2: Email + MasterDepartment → Department
    - Level 3: Email + Department → QueryType
    - Bundled encoders (risk-free deployment)
    - Comprehensive logging
    - Automatic inference routing
    """
    
    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', '830eb6d65978e36293a63635717da95bbbcb7a9d')
    START_TRAINING_EACH_N_UPDATES = int(os.getenv('START_TRAINING_EACH_N_UPDATES', 10))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-3))
    NUM_TRAIN_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 10))
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', 0.01))

    def __init__(self, **kwargs):
        super(MultiTaskBert, self).__init__(**kwargs)

        global multi_classs_model

        # ---- LAZY INITIALIZATION (CRITICAL FIX) ----
        if multi_classs_model is None:
            logger.info("🚀 Initializing MultiTaskBertModel...")
            multi_classs_model = MultiTaskBertModel(
                config=Config["MULTI_TASK"],
                logger=logger
            )
            multi_classs_model.reload_model()
            logger.info("✅ MultiTaskBertModel initialized")

        # ---- Inject Label Studio interfaces ----
        if hasattr(self, "label_interface"):
            multi_classs_model.label_interface = self.label_interface
            multi_classs_model.processed_label_encoders = ProcessLabels(
                self.parsed_label_config,
                pathlib.Path(Config["MULTI_TASK"]["MODEL_DIR"])
            )
            logger.info("✓ Label interface injected")

    def get_labels(self):
        """Get classification labels from label interface"""
        li = self.label_interface
        from_name, _, _ = li.get_first_tag_occurence('Labels', 'Text')
        tag = li.get_tag(from_name)
        return tag.labels
        
    def setup(self):
        """Configure model parameters"""
        self.set("model_version", f'{self.__class__.__name__}-v3.0.0')
        logger.info(f"✓ Model setup complete - version: {self.get('model_version')}")
    
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """
        STREAMLINED 3-LEVEL HIERARCHICAL INFERENCE
        
        Just pass email text - automatic routing through all three models:
        1. Email → MasterDepartment + Sentiment (Level 1)
        2. Email + MasterDepartment → Department (Level 2)
        3. Email + Department → QueryType (Level 3)
        
        Returns all predictions in one response.
        """
        
        li = self.label_interface
        from_name, to_name, value = li.get_first_tag_occurence('Taxonomy', 'HyperText')
        
        # Extract email texts from tasks
        texts = [self.preload_task_data(task, task['data']['html']) for task in tasks]
        
        logger.info(f"🔍 Starting inference for {len(texts)} email(s)...")
        
        # Run hierarchical inference
        results = multi_classs_model.predict(tasks, texts, context, **kwargs)

        predictions = []
        if results:
            predictions.append({
                'result': results,
                'score': sum(d["value"]["score"] for d in results) / len(results),
                'model_version': self.get('model_version')
            })
            logger.info(f"✅ Inference complete - {len(results)} predictions generated")
        else:
            logger.warning("⚠️  No predictions generated")
        
        return ModelResponse(predictions=predictions, model_version=self.get('model_version'))
        
    def predict_external(self, texts, **kwargs) -> ModelResponse:
        """
        External prediction endpoint (API usage)
        
        Args:
            texts: List of email text strings
            
        Returns:
            ModelResponse with hierarchical predictions
        """
        
        logger.info(f"🔍 External inference for {len(texts) if isinstance(texts, list) else 1} email(s)...")
        
        results = multi_classs_model.predict(None, texts, None, **kwargs)
        
        predictions = []
        if results:
            predictions.append({
                'result': results,
                'score': sum(d["value"]["score"] for d in results) / len(results),
                'model_version': self.get('model_version')
            })
            logger.info(f"✅ External inference complete")
        
        return ModelResponse(predictions=predictions, model_version=self.get('model_version'))

    def _get_tasks(self, project_id):
        """Download annotated tasks from Label Studio"""
        logger.info(f"📥 Fetching tasks from project {project_id}...")
        ls = label_studio_sdk.Client(self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project_id)
        tasks = project.get_labeled_tasks()
        logger.info(f"✓ Retrieved {len(tasks)} labeled tasks")
        return tasks

    def fit(self, event, data, **kwargs):
        """
        Train all three levels of the hierarchy
        
        Features:
        - Level 1: MasterDepartment + Sentiment
        - Level 2: Department (conditioned on MasterDepartment)
        - Level 3: QueryType (conditioned on Department)
        - Comprehensive logging with time estimates
        - Bundled encoder saving
        - Automatic evaluation on test split
        """
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            logger.info(f"Skip training: event {event} is not supported")
            return

        project_id = data['project']['id']
        tasks = self._get_tasks(project_id)
        
        if len(tasks) % self.START_TRAINING_EACH_N_UPDATES != 0 and event != 'START_TRAINING':
            logger.info(f"Skip training: {len(tasks)} tasks are not multiple of {self.START_TRAINING_EACH_N_UPDATES}")
            return

        logger.info("=" * 80)
        logger.info("🚀 3-LEVEL HIERARCHICAL TRAINING INITIATED")
        logger.info(f"📊 Total tasks: {len(tasks)}")
        logger.info(f"📋 Event: {event}")
        logger.info("=" * 80)
        
        # Inject preload_task_data for accessing task data
        multi_classs_model.preload_task_data = self.preload_task_data
        
        # Run training
        multi_classs_model.fit(event, data, tasks, **kwargs)
        
        # Reload model after training
        reload_model()
        
        logger.info("🎉 Training pipeline complete!")

    def fit_external(self, event, data, **kwargs):
        """External training wrapper"""
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            logger.info(f"Skip training: event {event} is not supported")
            return
            
        tasks = data["tasks"]
        logger.info(f"🚀 External training with {len(tasks)} tasks")
        
        multi_classs_model.fit_external(event, data, tasks, **kwargs)
        reload_model()
        
        logger.info("✅ External training complete")


# ========== Flask App Initialization ==========
from label_studio_ml.api import init_app

app = init_app(MultiTaskBert)


@app.route("/metrics/latest", methods=["GET"])
def metrics_latest():
    """
    Retrieve latest evaluation metrics
    
    Returns:
    - overall: accuracy, precision, recall, f1_score
    - classes: per-class metrics
    """
    try:
        logger.info("📊 Fetching latest metrics...")
        raw = multi_classs_model.get_latest_metrics_from_db()

        if "error" in raw:
            logger.warning(f"⚠️  {raw['error']}")
            return jsonify(raw), 404

        overall = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None
        }

        classes = {}

        for row in raw["metrics"]:
            task = row["task"]
            metric = row["metric"]
            value = row["value"]

            # ---- OVERALL METRICS (use weighted avg) ----
            if metric == "accuracy_overall":
                overall["accuracy"] = value

            elif metric == "weighted_avg_precision":
                overall["precision"] = value

            elif metric == "weighted_avg_recall":
                overall["recall"] = value

            elif metric == "weighted_avg_f1_score":
                overall["f1_score"] = value

            # ---- PER CLASS METRICS ----
            elif "_" in metric:
                class_name, metric_name = metric.rsplit("_", 1)

                if metric_name not in ["precision", "recall", "f1_score"]:
                    continue

                classes.setdefault(class_name, {})
                classes[class_name][metric_name] = value

                # accuracy per class = recall (standard definition)
                if metric_name == "recall":
                    classes[class_name]["accuracy"] = value

        logger.info("✅ Metrics retrieved successfully")
        return jsonify({
            "overall": overall,
            "classes": classes
        }), 200

    except Exception as e:
        logger.exception("❌ Failed to format metrics")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": multi_classs_model is not None,
        "model_version": "v3.0.0"
    }), 200


if __name__ == "__main__":
    logger.info("🚀 Starting ML Backend Server...")
    app.run(host='0.0.0.0', port=9090)
