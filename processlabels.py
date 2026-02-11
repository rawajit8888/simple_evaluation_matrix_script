import os
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder


class ProcessLabels:
    def __init__(self, parsed_label_config, model_dir):
        self.parsed_label_config = parsed_label_config
        self.encoder_dir = os.path.join(model_dir, "encoders")
        os.makedirs(self.encoder_dir, exist_ok=True)
        self.initEncoder()

    def initEncoder(self):
        """
        Supported label spaces for 3-level hierarchy:
        - classification    : MasterDepartment (Level 1) - also aliased as 'masterdepartment'
        - masterdepartment  : MasterDepartment (Level 1) - primary name
        - sentiment         : Sentiment (Level 1)
        - department        : Department (Level 2)
        - querytype         : QueryType (Level 3)
        - subquery          : Legacy alias for QueryType (backward compatibility)
        """
        self.labels = {
            "classification": {"encoder": None},      # MasterDepartment (alias)
            "masterdepartment": {"encoder": None},    # MasterDepartment (primary)
            "sentiment": {"encoder": None},           # Sentiment
            "department": {"encoder": None},          # Department (Level 2)
            "querytype": {"encoder": None},           # QueryType (Level 3)
            "subquery": {"encoder": None}             # Legacy alias for QueryType
        }

        # Initialize all encoders
        self.getLabelEncoder("classification")
        self.getLabelEncoder("masterdepartment")
        self.getLabelEncoder("sentiment")
        self.getLabelEncoder("department")
        self.getLabelEncoder("querytype")
        self.getLabelEncoder("subquery")

    def saveLabelEncoderFile(self, labelencoder, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(labelencoder, file)

    def getLabelEncoder(self, labeltype):
        """
        Load encoder if exists.
        Update encoder if new labels appear in Label Studio.
        Create encoder if missing.
        
        For 3-level hierarchy:
        - 'classification' and 'masterdepartment' share the same encoder
        - 'querytype' and 'subquery' share the same encoder (backward compatibility)
        """
        try:
            if labeltype not in self.labels:
                return None

            # Handle aliases - classification <-> masterdepartment
            if labeltype == "classification":
                # Check if masterdepartment encoder exists
                if self.labels["masterdepartment"]["encoder"] is not None:
                    self.labels["classification"]["encoder"] = self.labels["masterdepartment"]["encoder"]
                    return self.labels["classification"]["encoder"]
            elif labeltype == "masterdepartment":
                # Check if classification encoder exists
                if self.labels["classification"]["encoder"] is not None:
                    self.labels["masterdepartment"]["encoder"] = self.labels["classification"]["encoder"]
                    return self.labels["masterdepartment"]["encoder"]
            
            # Handle aliases - querytype <-> subquery
            if labeltype == "subquery":
                # Check if querytype encoder exists
                if self.labels["querytype"]["encoder"] is not None:
                    self.labels["subquery"]["encoder"] = self.labels["querytype"]["encoder"]
                    return self.labels["subquery"]["encoder"]
            elif labeltype == "querytype":
                # Check if subquery encoder exists
                if self.labels["subquery"]["encoder"] is not None:
                    self.labels["querytype"]["encoder"] = self.labels["subquery"]["encoder"]
                    return self.labels["querytype"]["encoder"]

            if self.labels[labeltype]["encoder"] is not None:
                return self.labels[labeltype]["encoder"]

            file_path = os.path.join(
                self.encoder_dir, f"{labeltype}_label_encoder.pkl"
            )

            # -------- CASE 1: Encoder exists on disk --------
            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    label_encoder = pickle.load(file)

                # Update encoder if LS has new labels
                if (
                    self.parsed_label_config
                    and labeltype in self.parsed_label_config
                    and "labels" in self.parsed_label_config[labeltype]
                ):
                    existing = label_encoder.classes_.tolist()
                    incoming = self.parsed_label_config[labeltype]["labels"]

                    new_labels = [l for l in incoming if l not in existing]

                    if new_labels:
                        updated_classes = np.concatenate(
                            (label_encoder.classes_, np.array(new_labels))
                        )
                        label_encoder.fit(updated_classes)
                        self.saveLabelEncoderFile(label_encoder, file_path)

            # -------- CASE 2: Encoder does NOT exist --------
            else:
                label_encoder = LabelEncoder()

                if (
                    self.parsed_label_config
                    and labeltype in self.parsed_label_config
                    and "labels" in self.parsed_label_config[labeltype]
                ):
                    label_encoder.fit(
                        self.parsed_label_config[labeltype]["labels"]
                    )
                    self.saveLabelEncoderFile(label_encoder, file_path)
                else:
                    # No labels available yet (safe empty encoder)
                    label_encoder.fit([])

            self.labels[labeltype]["encoder"] = label_encoder
            
            # Sync aliases after loading
            if labeltype == "classification" or labeltype == "masterdepartment":
                # Sync both aliases
                self.labels["classification"]["encoder"] = label_encoder
                self.labels["masterdepartment"]["encoder"] = label_encoder
            elif labeltype == "querytype" or labeltype == "subquery":
                # Sync both aliases
                self.labels["querytype"]["encoder"] = label_encoder
                self.labels["subquery"]["encoder"] = label_encoder
            
            return label_encoder

        except Exception as e:
            print(f"[ProcessLabels] Error loading encoder '{labeltype}': {e}")
            return None

    def __getitem__(self, key):
        """
        Allows usage:
        self.processed_label_encoders['classification']
        self.processed_label_encoders['masterdepartment']
        self.processed_label_encoders['sentiment']
        self.processed_label_encoders['department']
        self.processed_label_encoders['querytype']
        self.processed_label_encoders['subquery']  # legacy
        """
        if key not in self.labels:
            return None
        return self.labels[key]["encoder"]
