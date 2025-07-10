# ===============================================================
# WMT-21 Critical-Error-Detection · COMETKiwi-23 XL + Llama-3
# ===============================================================

import os
import pandas as pd
from comet import load_from_checkpoint, download_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from transformers import pipeline
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    """
    Configuration class for the WMT-21 Critical Error Detection project.
    Centralizes all adjustable parameters.
    """
    HF_TOKEN = os.environ.get("HF_TOKEN", "hf_YOUR_TOKEN_HERE") # IMPORTANT: Set this securely!
    HF_MODEL = "Unbabel/wmt23-cometkiwi-da-xl"  # Gated repo
    LLAMA_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Still needed for specifying local model
    # Local data paths - update these to match your directory structure
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw") # Default to data/raw relative to script
    PAIRS = ["encs", "ende", "enja", "enzh"]
    LLAMA_PROMPT = (
        "You are a translation quality checker.\n"
        "SRC: {src}\n"
        "MT:  {mt}\n\n"
        "Answer YES if MT changes the meaning in a critical way, otherwise NO."
    )
    COMET_BATCH_SIZE = 16

class ModelLoader:
    """
    Handles loading of COMETKiwi and local Llama models.
    """
    def __init__(self, config: Config):
        self.config = config
        self.comet_model = None
        self.llama_pipeline = None

    def load_comet_model(self):
        """
        Downloads and loads the COMETKiwi-23 XL model.
        """
        logging.info("⏬ Loading COMETKiwi-23 XL from Hugging Face …")
        os.environ["HF_TOKEN"] = self.config.HF_TOKEN
        try:
            model_path = download_model(self.config.HF_MODEL)
            self.comet_model = load_from_checkpoint(model_path)
            logging.info("✅ COMETKiwi-23 XL loaded successfully.")
        except Exception as e:
            logging.error(f"❌ Error loading COMETKiwi model: {e}")
            logging.error("Please ensure your HF_TOKEN is valid and has access to the gated model.")
            raise

    def load_llama_local(self):
        """
        Loads the TinyLlama model locally using the transformers pipeline.
        Determines device (GPU/CPU) automatically.
        """
        logging.info("🦙 Loading TinyLlama locally for critical error check …")
        device = 0 if torch.cuda.is_available() else -1
        logging.info(f"Using device: {'cuda:0' if device == 0 else 'cpu'}")

        try:
            torch_dtype = None
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    torch_dtype = torch.bfloat16
                    logging.info("Using bfloat16 for Llama model.")
                else:
                    torch_dtype = torch.float16 # Fallback to float16 if bfloat16 not supported
                    logging.info("Using float16 for Llama model.")

            self.llama_pipeline = pipeline(
                "text-generation",
                model=self.config.LLAMA_ID,
                torch_dtype=torch_dtype,
                device=device
            )
            logging.info("✅ TinyLlama loaded successfully locally.")
        except Exception as e:
            logging.error(f"❌ Error loading TinyLlama locally: {e}")
            logging.error("Please ensure you have sufficient RAM/VRAM and that the 'transformers' library is installed correctly.")
            raise

class DataLoader:
    """
    Handles loading WMT-21 QE data from local TSV files.
    """
    def __init__(self, data_dir: str, pairs: list):
        self.data_dir = data_dir
        self.pairs = pairs
        self.split_map = {"train": "train", "validation": "dev", "test": "dev"}

    def load_split(self, lp: str, split: str) -> pd.DataFrame:
        """
        Load WMT21 QE data from local TSV files.
        Expected format: ID, source, target, scores, label
        Converts 'ERR' to 1 and 'NOT' to 0 for the 'label' column.
        """
        file_suffix = self.split_map.get(split, split)
        filename = f"{lp}_majority_{file_suffix}.tsv"
        filepath = os.path.join(self.data_dir, filename)

        try:
            logging.info(f"📁 Attempting to load {filepath}")
            df = pd.read_csv(filepath, sep='\t', header=None,
                             names=['id', 'source', 'target', 'scores', 'label_raw'],
                             on_bad_lines='warn')

            # Convert label to binary: 1 for ERR, 0 for NOT
            df['label'] = (df['label_raw'] == 'ERR').astype(int)
            df['lp'] = lp
            df = df.drop(columns=['label_raw'])

            logging.info(f"✅ Loaded {len(df)} samples from {filename}.")
            logging.debug(f"Current label distribution for {lp}/{split}:\n{df['label'].value_counts()}")
            return df

        except FileNotFoundError:
            logging.warning(f"❌ File not found: {filepath}. Returning empty DataFrame.")
            return pd.DataFrame(columns=['id', 'source', 'target', 'scores', 'label', 'lp'])
        except Exception as e:
            logging.error(f"❌ Error loading {filepath}: {e}. Returning empty DataFrame.")
            return pd.DataFrame(columns=['id', 'source', 'target', 'scores', 'label', 'lp'])

    def load_all_splits(self):
        """
        Loads all specified language pairs for validation and test splits.
        Handles cases where dev data might be empty or single-class.
        """
        logging.info("📥 Loading WMT-21 QE data from local files …")

        dev_df_parts = [self.load_split(p, "validation") for p in self.pairs]
        test_df_parts = [self.load_split(p, "test") for p in self.pairs]

        dev_df = pd.concat(dev_df_parts, ignore_index=True)
        test_df = pd.concat(test_df_parts, ignore_index=True)

        # Robust check for dev data having both classes before proceeding to training
        if dev_df.empty or dev_df['label'].nunique() < 2:
            logging.warning("📋 No dev data found or dev data has only one class. Attempting to use train data as dev split.")
            train_df_parts = [self.load_split(p, "train") for p in self.pairs]
            temp_dev_df = pd.concat(train_df_parts, ignore_index=True)
            if temp_dev_df['label'].nunique() >= 2:
                dev_df = temp_dev_df
                logging.info("✅ Successfully used train data as dev split with both classes.")
            else:
                logging.error("❌ Train data also contains only one class. Cannot proceed with training or evaluation.")
                logging.error(f"📊 Label distribution - Train (attempted dev): {temp_dev_df['label'].value_counts().to_dict()}")
                raise ValueError("Insufficient data for training: Dev/Train data must contain both classes.")
        else:
            logging.info(f"✅ Dev data successfully loaded with both classes: {dev_df['label'].value_counts().to_dict()}")

        if test_df.empty:
            logging.warning("⚠️ Test data is empty. Evaluation will be skipped.")
        elif test_df['label'].nunique() < 2:
            logging.warning("⚠️ Test data has only one class. Evaluation metrics requiring multiple classes will be affected.")

        logging.info(f"✅ Final dataset sizes: {len(dev_df)} dev, {len(test_df)} test samples")
        logging.debug(f"📊 Dev DataFrame columns: {list(dev_df.columns)}")
        logging.debug(f"📊 Sample dev data:\n{dev_df.head()}")
        logging.info(f"📊 Label distribution - Dev: {dev_df['label'].value_counts().to_dict()}")
        if not test_df.empty:
            logging.info(f"📊 Label distribution - Test: {test_df['label'].value_counts().to_dict()}")

        return dev_df, test_df

class FeatureExtractor:
    """
    Calculates COMETKiwi scores and Llama-3 critical error flags.
    """
    def __init__(self, comet_model, llama_pipeline, prompt_template: str, comet_batch_size: int):
        self.comet_model = comet_model
        self.llama_pipeline = llama_pipeline
        self.prompt_template = prompt_template
        self.comet_batch_size = comet_batch_size

    def get_comet_scores(self, df: pd.DataFrame) -> list:
        """
        Calculates COMET scores for source-target pairs in a DataFrame.
        """
        if df.empty:
            return []
        logging.info("⚙️ Scoring with COMETKiwi XL …")
        batch = [{"src": s, "mt": t} for s, t in zip(df.source, df.target)]
        return self.comet_model.predict(batch, batch_size=self.comet_batch_size)["scores"]

    def get_llama_critical_flag(self, df: pd.DataFrame) -> list:
        """
        Runs Llama-3 to check for critical errors and returns binary flags.
        """
        if df.empty:
            return []
        logging.info("🦙 Running Llama-3 verifier locally …")
        llama_flags = []
        # Process in batches or iteratively, depending on performance
        for src, mt in zip(df.source, df.target):
            full_prompt = self.prompt_template.format(src=src, mt=mt)
            try:
                output = self.llama_pipeline(
                    full_prompt,
                    max_new_tokens=5,
                    temperature=0.0,
                    do_sample=False,
                    return_full_text=False
                )[0]['generated_text']
                llama_flags.append("YES" in output.strip().upper())
            except Exception as e:
                logging.error(f"Error during Llama inference for (src='{src[:50]}...', mt='{mt[:50]}...'): {e}")
                llama_flags.append(False) # Default to False on error
        return llama_flags

class FusionModel:
    """
    Trains and evaluates a Logistic Regression model for fusion.
    """
    def __init__(self):
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the logistic regression model.
        """
        logging.info("🏋️ Training Logistic Regression model …")
        if y_train.nunique() < 2:
            logging.error("\n⚠️ Cannot train Logistic Regression: Development set contains only one class.")
            logging.error("Please ensure your dev/train data files contain both 'NOT' (0) and 'ERR' (1) labels.")
            raise ValueError("Training data must contain at least two classes.")
        self.model.fit(X_train, y_train)
        logging.info("✅ Logistic Regression model trained successfully.")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, test_df_original: pd.DataFrame):
        """
        Evaluates the model and prints overall and per-language-pair MCC.
        """
        logging.info("\n📊 Evaluating model on Test set …")
        if X_test.empty or y_test.empty or y_test.nunique() < 2:
            logging.warning("\n⚠️ Cannot evaluate on Test set: Test set is empty or contains only one class for evaluation.")
            logging.warning("Please ensure your test data files contain both 'NOT' (0) and 'ERR' (1) labels.")
            return

        pred = self.model.predict(X_test)
        overall_mcc = matthews_corrcoef(y_test, pred)

        logging.info("\n============ RESULTS (COMET XL + Llama-3) ============")
        logging.info(f"OVERALL MCC (All LPs): {overall_mcc:0.3f}")

        # Add predictions back to a temporary DataFrame for easy grouping
        temp_test_df = test_df_original.copy()
        temp_test_df['pred'] = pred

        for lp, sub in temp_test_df.groupby("lp"):
            if sub['label'].nunique() >= 2:
                lp_mcc = matthews_corrcoef(sub.label, sub.pred)
                logging.info(f"{lp:5s} MCC = {lp_mcc:0.3f}")
            else:
                logging.warning(f"{lp:5s} MCC = N/A (Test subset for {lp} has only one class for MCC calculation)")


def main():
    """
    Main function to run the WMT-21 Critical Error Detection pipeline.
    """
    config = Config()

    # 1. Load Models
    model_loader = ModelLoader(config)
    try:
        model_loader.load_comet_model()
        model_loader.load_llama_local()
    except Exception:
        logging.error("Exiting due to model loading failure.")
        return

    # 2. Load Data
    data_loader = DataLoader(config.DATA_DIR, config.PAIRS)
    try:
        dev_df, test_df = data_loader.load_all_splits()
    except ValueError as e:
        logging.error(f"Exiting due to data loading failure: {e}")
        return

    # 3. Feature Extraction
    feature_extractor = FeatureExtractor(
        model_loader.comet_model,
        model_loader.llama_pipeline,
        config.LLAMA_PROMPT,
        config.COMET_BATCH_SIZE
    )

    dev_df["comet"] = feature_extractor.get_comet_scores(dev_df)
    test_df["comet"] = feature_extractor.get_comet_scores(test_df)

    dev_df["llama_bin"] = feature_extractor.get_llama_critical_flag(dev_df)
    test_df["llama_bin"] = feature_extractor.get_llama_critical_flag(test_df)

    # Convert boolean llama_bin to int for Logistic Regression
    dev_df['llama_bin'] = dev_df['llama_bin'].astype(int)
    test_df['llama_bin'] = test_df['llama_bin'].astype(int)

    # 4. Train Fusion Model
    fusion_model = FusionModel()
    try:
        X_dev = dev_df[["comet", "llama_bin"]]
        y_dev = dev_df["label"]
        fusion_model.train(X_dev, y_dev)
    except ValueError as e:
        logging.error(f"Exiting due to training failure: {e}")
        return

    # 5. Evaluate on Test Set
    X_test = test_df[["comet", "llama_bin"]]
    y_test = test_df["label"]
    fusion_model.evaluate(X_test, y_test, test_df)


if __name__ == "__main__":
    main()