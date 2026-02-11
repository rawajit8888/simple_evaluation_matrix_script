import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle


class ProcessLabels:
    """
    3-Level Hierarchical Label Encoder Manager
    
    Creates and manages encoders for:
    - lvl1: Master Department (e.g., "Internet Banking")
    - lvl2: Sub-Category (e.g., "Internet Banking > Account Access")
    - lvl3: Specific Issue (e.g., "Internet Banking > Account Access > Unblock")
    - sentiment: Sentiment labels
    """

    def __init__(self, parsed_label_config, model_dir):
        self.parsed_label_config = parsed_label_config
        self.encoder_dir = f'{model_dir}\\encoders'
        
        os.makedirs(self.encoder_dir, exist_ok=True)
        
        self.labels = {
            "lvl1": {"encoder": None},
            "lvl2": {"encoder": None},
            "lvl3": {"encoder": None},
            "sentiment": {"encoder": None}
        }
        
        if self.parsed_label_config:
            print("🔧 Building hierarchical encoders from config...")
            self.build_hierarchical_encoders()
        else:
            print("📂 Loading existing encoders only...")
            self.load_existing_only()

    def load_existing_only(self):
        """Load pre-existing encoder files"""
        for name in ["lvl1", "lvl2", "lvl3", "sentiment"]:
            path = f"{self.encoder_dir}\\{name}_label_encoder.pkl"
            
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.labels[name]["encoder"] = pickle.load(f)
                print(f"✅ Loaded {name} encoder from {path}")
            else:
                raise RuntimeError(f"❌ Encoder file missing: {path}")
    
    def save_encoder(self, encoder, path):
        """Save encoder to disk"""
        with open(path, "wb") as f:
            pickle.dump(encoder, f)
        print(f"💾 Saved encoder to {path}")
    
    def load_or_create(self, name, values):
        """
        Load existing encoder or create new one.
        If new values are found, expand the encoder.
        """
        path = f"{self.encoder_dir}\\{name}_label_encoder.pkl"
        
        if os.path.exists(path):
            # Load existing encoder
            with open(path, "rb") as f:
                enc = pickle.load(f)
            
            # Check for new values
            new_vals = [v for v in values if v not in enc.classes_]
            
            if new_vals:
                print(f"🆕 Found {len(new_vals)} new values for {name}, expanding encoder...")
                updated = np.concatenate(
                    (enc.classes_, np.array(new_vals, dtype=enc.classes_.dtype))
                )
                enc.fit(updated)
                self.save_encoder(enc, path)
        else:
            # Create new encoder
            print(f"🆕 Creating new encoder for {name} with {len(values)} values...")
            enc = LabelEncoder()
            enc.fit(values)
            self.save_encoder(enc, path)
        
        self.labels[name]["encoder"] = enc
    
    def build_hierarchical_encoders(self):
        """
        Build 3-level hierarchical encoders from Label Studio taxonomy.
        
        Example taxonomy path: ["Internet Banking", "Account Access", "Unblock"]
        Creates:
        - lvl1: "Internet Banking"
        - lvl2: "Internet Banking > Account Access"
        - lvl3: "Internet Banking > Account Access > Unblock"
        """
        if not self.parsed_label_config:
            raise ValueError("parsed_label_config required to build encoders")
        
        lvl1_vals = set()
        lvl2_vals = set()
        lvl3_vals = set()
        
        # Extract taxonomy paths from Label Studio config
        taxonomy_paths = self.parsed_label_config["classification"]["labels"]
        
        print(f"📊 Processing {len(taxonomy_paths)} taxonomy paths...")
        
        for path in taxonomy_paths:
            if isinstance(path, list):
                # Level 1: First element only
                if len(path) >= 1:
                    lvl1_vals.add(path[0])
                
                # Level 2: First two elements joined
                if len(path) >= 2:
                    lvl2_vals.add(" > ".join(path[:2]))
                
                # Level 3: All three elements joined
                if len(path) >= 3:
                    lvl3_vals.add(" > ".join(path[:3]))
        
        print(f"📊 Found {len(lvl1_vals)} Level 1 categories")
        print(f"📊 Found {len(lvl2_vals)} Level 2 categories")
        print(f"📊 Found {len(lvl3_vals)} Level 3 categories")
        
        # Create/update encoders
        self.load_or_create("lvl1", list(lvl1_vals))
        self.load_or_create("lvl2", list(lvl2_vals))
        self.load_or_create("lvl3", list(lvl3_vals))
        
        # Sentiment encoder (unchanged)
        self.load_or_create(
            "sentiment",
            self.parsed_label_config["sentiment"]["labels"]
        )
        
        print("✅ All hierarchical encoders built successfully!")
    
    def split_taxonomy(self, taxonomy_path):
        """
        Split taxonomy path into 3 levels
        
        Args:
            taxonomy_path: List like ["Internet Banking", "Account Access", "Unblock"]
        
        Returns:
            Tuple of (lvl1, lvl2, lvl3) strings
        """
        lvl1 = taxonomy_path[0] if len(taxonomy_path) >= 1 else ""
        lvl2 = " > ".join(taxonomy_path[:2]) if len(taxonomy_path) >= 2 else ""
        lvl3 = " > ".join(taxonomy_path[:3]) if len(taxonomy_path) >= 3 else ""
        
        return lvl1, lvl2, lvl3
    
    def encode_levels(self, taxonomy_path):
        """
        Encode taxonomy path into integer IDs for all 3 levels
        
        Args:
            taxonomy_path: List like ["Internet Banking", "Account Access", "Unblock"]
        
        Returns:
            Tuple of (lvl1_id, lvl2_id, lvl3_id)
        """
        lvl1, lvl2, lvl3 = self.split_taxonomy(taxonomy_path)
        
        return (
            self.labels["lvl1"]["encoder"].transform([lvl1])[0],
            self.labels["lvl2"]["encoder"].transform([lvl2])[0],
            self.labels["lvl3"]["encoder"].transform([lvl3])[0]
        )
    
    def decode_levels(self, l1_id, l2_id, l3_id):
        """
        Decode integer IDs back to label strings
        
        Args:
            l1_id, l2_id, l3_id: Integer encoder indices
        
        Returns:
            Tuple of (lvl1_label, lvl2_label, lvl3_label)
        """
        return (
            self.labels["lvl1"]["encoder"].inverse_transform([l1_id])[0],
            self.labels["lvl2"]["encoder"].inverse_transform([l2_id])[0],
            self.labels["lvl3"]["encoder"].inverse_transform([l3_id])[0]
        )
    
    def __getitem__(self, key):
        """
        Access encoder by name
        
        Usage: encoder = processed_labels['lvl1']
        """
        return self.labels[key]["encoder"]
