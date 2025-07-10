# WMT-21 Critical-Error-Detection: COMETKiwi-23 XL + TinyLLama Fusion [[Dashboard](https://muskaan712.github.io/error-detection-llm/)]

This repository contains the code for a critical error detection system, combining the strengths of a COMETKiwi-23 XL quality estimation model and a local TinyLlama (TinyLlama) model. The system identifies "critical errors" in machine translation outputs by fusing the scores from both models using a logistic regression classifier.



## 🎯 Approach

The core idea is a two-stage approach for translation quality assessment, specifically focusing on critical errors:

1.  **COMETKiwi-23 XL for Overall Quality:** A state-of-the-art machine translation quality estimation model (COMETKiwi-23 XL) provides a general quality score for each translation segment. This model is adept at capturing various quality aspects but might not be specifically fine-tuned for critical errors.

2.  **TinyLlama for Critical Error Verification:** A smaller, locally-run TinyLlama model (TinyLlama in this implementation) acts as a specialized "critical error verifier." It's prompted to explicitly identify if a translation critically changes the meaning of the source text. This provides a binary (YES/NO) signal for critical errors.

3.  **Logistic Regression Fusion:** The scores from COMETKiwi-23 XL and the binary flag from TinyLlama are then combined as features for a simple logistic regression classifier. This classifier is trained on human-annotated critical error labels (ERR/NOT) to learn how to best combine these signals to predict critical errors. The use of `class_weight="balanced"` helps handle potential class imbalance in critical error datasets.

## 🚀 Pipeline

The execution pipeline follows these steps:

1.  **Configuration Loading:** All parameters (Hugging Face token, model IDs, data paths, prompts) are loaded from a central `Config` class.
2.  **Model Loading:**
    * **COMETKiwi-23 XL:** The pre-trained COMETKiwi-23 XL model is downloaded and loaded from Hugging Face. Access to a gated repository might require specific `HF_TOKEN` permissions.
    * **TinyLlama (TinyLlama):** The TinyLlama model is loaded locally using the `transformers` library's `pipeline`. The system automatically detects and utilizes a GPU if available, falling back to CPU.
3.  **Data Loading:**
    * WMT-21 Quality Estimation (QE) data is loaded from local TSV files. The `load_split` function parses the data, converting raw labels ('ERR', 'NOT') into binary (1, 0) for critical error detection.
    * A robust check ensures that the development (or training, if dev is insufficient) dataset contains both critical error and non-error examples, which is crucial for training the classifier.
4.  **Feature Extraction:**
    * **COMET Scores:** The loaded COMETKiwi model is used to predict quality scores for all source-target pairs in both the development and test sets.
    * **TinyLlama Critical Flags:** The TinyLlama model is prompted for each source-target pair to determine if a critical error is present. The output is parsed to yield a binary flag (`True` for YES, `False` for NO).
5.  **Fusion Model Training:**
    * A Logistic Regression classifier is trained using the COMET scores and TinyLlama binary flags as input features, and the human-annotated critical error labels as the target variable. `class_weight="balanced"` is applied to account for potential class imbalance.
6.  **Evaluation:**
    * The trained fusion model is evaluated on the test set using the Matthews Correlation Coefficient (MCC), a suitable metric for imbalanced classification tasks.
    * Both an overall MCC and per-language-pair MCC scores are reported.

## 📁 Code Structure

The code is organized into modular classes and functions for better readability and maintainability:

* `Config`: Stores all configuration parameters for the project.
* `ModelLoader`: Handles the loading of both the COMET and Llama models.
* `DataLoader`: Manages the loading and initial processing of the WMT-21 QE datasets.
* `FeatureExtractor`: Responsible for calculating COMET scores and TinyLlama critical error flags.
* `FusionModel`: Encapsulates the training and evaluation logic for the Logistic Regression classifier.
* `main()`: Orchestrates the entire pipeline, calling methods from the respective classes.

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/wmt21-critical-error-detection.git](https://github.com/your-username/wmt21-critical-error-detection.git)
    cd wmt21-critical-error-detection
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (You will need to create a `requirements.txt` file with the following packages: `pandas`, `unbabel-comet`, `scikit-learn`, `transformers`, `torch`)

4.  **Hugging Face Token:**
    * You need a Hugging Face token with access to gated models (specifically `Unbabel/wmt23-cometkiwi-da-xl`).
    * **Set your Hugging Face token as an environment variable before running the script**:
        ```bash
        export HF_TOKEN="hf_YOUR_TOKEN_HERE"
        ```
        (Replace `hf_YOUR_TOKEN_HERE` with your actual token).
        Alternatively, you can modify the `HF_TOKEN` in `Config` class directly (not recommended for sensitive tokens).

## 📊 Data

The script expects WMT-21 QE data in TSV format, located in the `data/raw` directory (relative to the script). The file naming convention is `[language_pair]_majority_[split].tsv`, e.g., `encs_majority_dev.tsv`.

The TSV files should have the following columns (without header):
`ID`, `source`, `target`, `scores` (can be dummy if not used directly), `label_raw` (e.g., 'ERR' or 'NOT').

## 🚀 Usage

To run the pipeline, simply execute the `main.py` script:

```bash
python main.py
```

The script will output progress messages, model loading status, data statistics, and finally, the evaluation results (MCC scores).

## 📈  Results
- ✅ Final dataset sizes: 4000 dev samples
- 📊 Label distribution - Dev: {0: 3322, 1: 678}
- ⚙️ Scoring dev with COMETKiwi XL …
- 🦙 Running TinyLLamma verifier locally …


### Here are the Matthews Correlation Coefficient (MCC) results obtained from running the pipeline:

| Metric                 | Value |
| :--------------------- | :---- |
| **Overall MCC (4 LPs)** | 0.282 |
| encs                | 0.290 |
| ende                | 0.288 |
| enja                | 0.215 |
| enzh                | 0.182 |