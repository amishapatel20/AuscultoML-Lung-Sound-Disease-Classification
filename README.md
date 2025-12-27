# Lung Sound Disease Classifier – Research Summary and System Design
#Dataset  link --- https://drive.google.com/file/d/1TmrdhbssDlx0NOeb8vH08RvTuJK-ncoj/view?usp=sharing
This repository implements a complete, end‑to‑end pipeline for **automatic classification of lung sounds** (Normal, Wheeze, Fine Crackle, etc.) from auscultation audio. It includes:

- Data ingestion and cleaning from a clinical lung‑sound dataset
- Event‑level clipping of long recordings into labeled segments
- FFT‑based band‑energy feature engineering in a medically relevant band (100–2000 Hz)
- Supervised learning with a Random Forest classifier
- Model persistence (`lung_model.joblib`) and a FastAPI service for real‑time inference

The goal is to explore whether simple, interpretable frequency‑domain features can separate common lung sound patterns and to build a minimal, deployable prototype suitable for further research.

---

## 1. Key Outcomes (At a Glance)

### 1.1 Classification Performance (segment‑level)

| Metric                          | Value              |
|---------------------------------|--------------------|
| Train accuracy                  | **1.00**           |
| Test accuracy                   | **0.91**           |
| Number of test segments         | 35                 |
| Feature type                    | Single band energy |
| Frequency band used             | 100–2000 Hz        |

### 1.2 Per‑Class Test Performance

| Class         | Precision | Recall | F1‑score | Support |
|---------------|-----------|--------|----------|---------|
| Fine Crackle  | 0.67      | 0.50   | 0.57     | 4       |
| Normal        | 0.94      | 0.97   | 0.95     | 30      |
| Wheeze        | 1.00      | 1.00   | 1.00     | 1       |
| **Overall**   | **0.91**  | **0.91** | **0.91** | **35** |

These numbers come directly from the training notebook [Model.ipynb](Model.ipynb), using the internal `train_test_split` on the engineered segment‑level dataset (`features_all`).

### 1.3 Dataset Label Distribution (subset used)

From a scan of 200 JSON annotation files in the dataset, the following label counts were observed:

| Label           | Count |
|-----------------|-------|
| Normal          | 628   |
| Fine Crackle    | 83    |
| Wheeze          | 39    |
| DAS             | 31    |
| CAS             | 12    |
| Poor Quality    | 17    |
| CAS & DAS       | 4     |
| Coarse Crackle  | 2     |
| Wheeze+Crackle  | 1     |

This imbalance (e.g. far more Normal than Fine Crackle/Wheeze) is important context when interpreting the metrics above and motivates future work on class‑imbalance handling.

---

## 2. Project Structure (Conceptual)

Typical layout after running the notebook once:

- **[Model.ipynb](Model.ipynb)** – Main notebook containing:
  - Dataset ZIP handling and extraction
  - Directory inspection and validation
  - Clipping of recordings into labeled segments
  - FFT‑based feature extraction
  - Model training, evaluation, and saving
- **requirements.txt** – List of Python dependencies needed for the notebook and API.
- **lung_model.joblib** – Saved model artifact (RandomForestClassifier + LabelEncoder) produced by the notebook.
- **classification_folder/** – Extraction target for the dataset ZIP, containing:
  - `Classification/`
    - `train_classification_json/`
    - `test_classification_json/`
    - `train_classification_wav/`
    - `test_classification_wav/`
- **clips/** – Automatically created by the notebook; holds clipped WAV segments organized by label:
  - `clips/Normal/…wav`
  - `clips/Fine_Crackle/…wav`
  - `clips/Wheeze/…wav`
  - etc.
- **(Optional) lung_pipeline.py** – Inference helper that:
  - Loads `lung_model.joblib`
  - Computes band energy for a new WAV
  - Returns predicted label
- **(Optional) main.py** – FastAPI app exposing a `/predict` endpoint for model inference.

---

## 3. End-to-End System Design

### 3.1 Short Interview Answer

If an interviewer asks, **"What is the system design of your project?"**, you can summarize it as:

- A **data processing pipeline** that ingests a lung-sound dataset (WAV + JSON), validates the folder structure, and clips long recordings into **short, labeled audio segments** based on event annotations.
- A **feature-extraction component** that converts each segment into a simple, interpretable **FFT band‑energy feature** in the 100–2000 Hz band and stores everything in a pandas DataFrame.
- A **model-training module** that uses this feature table to train and evaluate a **RandomForest classifier**, reports metrics (accuracy, confusion matrix, per‑class precision/recall/F1), and saves the trained model + label encoder as `lung_model.joblib`.
- An **inference pipeline** (lung_pipeline.py) that loads the saved model, computes the same band‑energy feature for any new WAV file, and outputs the predicted lung sound class.
- A **FastAPI web service** (main.py) that exposes a `/predict` endpoint so external clients or a UI can upload audio and receive real‑time predictions.

In short: it is a modular system with clear stages (data ingestion → segmentation → feature engineering → model training → persisted model → web API) connected by simple, well‑defined interfaces (files, DataFrames, and function calls).

### 3.2 High-Level Data Flow

1. **Raw Data (ZIP)**  
   A ZIP file (e.g. `Lung_classification.zip`) contains the dataset provided by the authors.

2. **Extraction & Folder Setup**  
   The notebook:
   - Determines the base project directory (`base_dir = Path().resolve()`)
   - Locates the ZIP file (with a robust fallback asking the user for a path)
   - Extracts all files to `classification_folder/`

3. **Dataset Structure Validation**  
   The notebook checks that `classification_folder/Classification` and its subfolders (train/test JSON/WAV) exist and lists their contents.

4. **Linking WAV and JSON by ID**  
   Helper function `_find_pair_paths(target_id, split="train")`:
   - Given an ID like `1684` and a split (`"train"` or `"test"`), it searches recursively under:
     - `train_classification_json/` or `test_classification_json/`
     - `train_classification_wav/` or `test_classification_wav/`
   - Matches filenames ending with that ID (e.g. `..._1684.json`, `..._1684.wav`)
   - Returns the pair `(json_path, wav_path)` for that recording.

5. **Clipping Recordings into Segments**  
   Function `clip_segments_for_id(target_id, split="train", output_root=None)`:

   - Loads the WAV (`librosa.load`) and corresponding JSON metadata.
   - Reads `event_annotation` from JSON: a list of events, each with `start`, `end`, and `type`.
   - Uses `_detect_time_unit` to infer if event times are in **samples** or **milliseconds**:
     - Compares the maximum event `end` to the actual audio duration using both assumptions.
   - Converts each event’s `start`/`end` into **sample indices**.
   - Clips `y[start_sample:end_sample]` for each event.
   - Determines a label per segment:
     - Prefer `event["type"]` (e.g. "Fine Crackle", "Wheeze")
     - Fall back to `record_annotation` if event type is missing
     - Lastly use `"Unknown"` if nothing is available
   - Saves each segment as a new WAV file under `clips/<Label>/...segX_Label.wav` using `soundfile.write`.
   - Returns `clips_info`: a list of dictionaries with:
     - Original id
     - Segment index
     - Label
     - Start/end sample indices
     - Output file path

6. **FFT-Based Feature Extraction**  
   For each clipped WAV file, the notebook uses:

   - `compute_band_energy_for_clip(path, low_f=100.0, high_f=2000.0)`:
     - Loads the clip with `librosa.load`.
     - Computes real FFT using `np.fft.rfft`.
     - Computes frequency bins with `np.fft.rfftfreq`.
     - Selects frequencies between `low_f` and `high_f` (default 100–2000 Hz).
     - Calculates **band energy** as the sum of squared magnitudes in that band.
   - `build_features_from_clips(clips, low_f, high_f)`:
     - Iterates over `clips_info` entries.
     - For each clip, computes band energy using the function above.
     - Builds a pandas DataFrame, where each row represents one segment with columns:
       - `id`, `segment_index`, `label`, `low_f`, `high_f`, `band_energy`.

7. **Aggregating Features Across Many IDs**  
   Function `build_features_for_many_ids(max_ids=50, low_f=100.0, high_f=2000.0)`:

   - Scans `train_classification_json/` for up to `max_ids` different recording IDs.
   - For each ID, calls `clip_segments_for_id` to generate segments.
   - Concatenates all `clips_info` into a single list.
   - Calls `build_features_from_clips` to compute features for all segments.
   - Prints label distribution and returns a DataFrame `features_all`.

8. **Model Training & Evaluation**  
   The notebook then:

   - Chooses the feature DataFrame (`features_all` if available, otherwise `features_1684`).
   - Extracts input and target:
     - `X = df[["band_energy"]].values` (a single numeric feature per segment)
     - `y = df["label"].values` (string labels: Normal, Wheeze, etc.)
   - Encodes labels numerically using `LabelEncoder`.
   - Splits data into train and test sets via `train_test_split` (with stratification):
     - This is a **validation split inside the training data**, separate from the dataset’s own train/test folders.
   - Trains a `RandomForestClassifier` on `X_train, y_train`.
   - Evaluates accuracy on both train and test sets.
   - Prints:
     - Train and test accuracy
     - Confusion matrix
     - Label mapping (encoded index → original label)
     - Full classification report (precision, recall, F1-score per class)

9. **Model Saving (Persistence)**  
   After training, the notebook:

   - Packs the trained model and label encoder into a dictionary:
     - `artifact = {"model": clf, "label_encoder": le}`
   - Saves it to `lung_model.joblib` using `joblib.dump`.

10. **Inference Pipeline (Deployed Model)**  
    Outside the notebook, a typical inference helper (`lung_pipeline.py`) would:

    - Load `lung_model.joblib` at import time.
    - Re-implement `compute_band_energy_for_clip` logic for arbitrary WAV inputs.
    - Provide a function `predict_lung_sound(path)` that:
      - Loads a new WAV file.
      - Computes its band energy in the 100–2000 Hz band.
      - Applies the loaded RandomForest model to predict a label.
      - Converts the numeric prediction back to a class name via the loaded `LabelEncoder`.

11. **FastAPI Web Service**  
    A typical FastAPI app (`main.py`) is structured as:

    - Creates a `FastAPI` instance with a title, e.g. `FastAPI(title="Lung Sound Disease Classifier")`.
    - Adds CORS middleware (optional, for local web UIs).
    - Defines endpoints:
      - `GET /` – health check returning a simple JSON message.
      - `POST /predict` – accepts a WAV file upload using `UploadFile = File(...)`, saves it temporarily, calls `predict_lung_sound` from the inference module, and returns the predicted label and band energy.
    - Run using Uvicorn:
      - `uvicorn main:app --reload`

---

## 4. Libraries and Their Roles

### 4.1 Core and Utility Libraries

- **pathlib (Path)**
  - Used to handle filesystem paths in an OS-independent, object-oriented way.
  - Example: `base_dir = Path().resolve()` gives the absolute project directory.
  - `base_dir / "classification_folder" / "Classification"` builds paths safely using `/` operator instead of string concatenation.

- **json**
  - Used to load event annotations from JSON files.
  - Example: `meta = json.load(open(json_path))` to read `record_annotation` and `event_annotation`.

- **zipfile**
  - Used to extract the dataset ZIP into a local folder.
  - Example: `zipfile.ZipFile(zip_file_path, "r").extractall(extract_folder_path)`.

- **soundfile (imported as sf)**
  - Used for writing audio data (numpy arrays) back to disk as WAV files.
  - Example: `sf.write(out_path, y[start:end], sr)` in `clip_segments_for_id` to save each clipped segment.

### 4.2 Audio and Signal Processing

- **librosa**
  - High-level audio analysis library.
  - Used to load WAV files with `librosa.load(path, sr=None)`:
    - Returns a numpy array of samples and the sampling rate.
  - Handles mono conversion and different sampling rates automatically.

- **numpy** (including `np.fft`)
  - Fundamental numerical library used for array operations.
  - FFT functions:
    - `np.fft.rfft(y_clip)` – real-valued FFT to transform from time domain to frequency domain.
    - `np.fft.rfftfreq(len(y_clip), d=1.0/sr)` – compute the corresponding frequency bins.
  - Used for computing magnitudes and band energies:
    - `mag = np.abs(Y)`
    - `band_energy = np.sum(mag[band_mask] ** 2)`

### 4.3 Data Handling and (Optional) Visualization

- **pandas**
  - Used to build structured datasets (`DataFrame`) from features and labels.
  - Each row corresponds to one clipped segment; columns include `id`, `segment_index`, `label`, `low_f`, `high_f`, `band_energy`.

- **matplotlib / seaborn** (in requirements.txt)
  - Can be used inside the notebook to visualize distributions, confusion matrices, or feature histograms.
  - Not strictly required by the core pipeline, but helpful for EDA and plots.

### 4.4 Machine Learning

- **scikit-learn**
  - Provides standard ML primitives:
    - **LabelEncoder**: converts string labels (e.g. "Normal", "Wheeze") into integer codes.
    - **train_test_split**: splits data into training and validation sets.</n    - **RandomForestClassifier**: ensemble learning model used for classification.
    - **Metrics**: `confusion_matrix`, `classification_report` for evaluation.
  - Chosen here because Random Forests are:
    - Robust to noisy features
    - Work well on low-dimensional feature spaces
    - Easy to interpret via feature importance (even though we currently use a single feature).

### 4.5 Model Serialization

- **joblib**
  - Used to efficiently save and load Python objects that contain large numpy arrays.
  - Saves the entire model pipeline (classifier + label encoder) into `lung_model.joblib`.

### 4.6 Web API and Deployment

- **FastAPI**
  - High-performance Python web framework for building APIs.
  - Automatically generates interactive API docs at `/docs` (Swagger UI).
  - Handles request parsing, file uploads, and response validation.
  - In this project:
    - Defines `/predict` endpoint that accepts an uploaded WAV and returns the predicted class.

- **Uvicorn**
  - ASGI server used to run the FastAPI app.
  - Command example: `uvicorn main:app --reload`.

- **python-multipart**
  - Required dependency for FastAPI to handle `multipart/form-data` file uploads (used by `UploadFile = File(...)`).

---

## 5. Model Details

### 5.1 Input Feature

- For each clipped lung sound segment, the model uses **one main feature**:
  - **Band-limited energy** of the audio signal in the frequency range [100 Hz, 2000 Hz].
- This is computed using the FFT magnitude spectrum:
  - Transform the clip from time domain → frequency domain.
  - Select frequencies between 100 and 2000 Hz.
  - Sum the squared magnitudes (energy) across these frequencies.
- This choice is motivated by:
  - Many lung sound characteristics (crackles, wheezes) are prominent in that frequency band.
  - A single, interpretable scalar feature simplifies the initial model design.

### 5.2 Target Labels

- The target labels come from JSON:
  - Primarily from `event_annotation[i]["type"]`
  - Optionally from `record_annotation` when event-level labels are missing.
- Typical labels include:
  - **Normal**
  - **Fine Crackle**
  - **Wheeze**
  - (others depending on dataset)

### 5.3 Model Architecture

- **RandomForestClassifier** from scikit-learn:
  - Ensemble of decision trees.
  - Each tree sees a bootstrap sample of the data and a subset of features.
  - Final prediction is based on majority vote across trees.
- Advantages for this project:
  - Handles non-linear decision boundaries.
  - Works reasonably well even with a small number of features.
  - Generally robust to outliers and noise.

### 5.4 Training Procedure

1. **Dataset Preparation**
   - Build `features_all` (or `features_1684` for a single-id experiment).
   - Ensure multiple classes are present in the data.

2. **Encoding & Splitting**
   - Convert string labels to integers using `LabelEncoder`.
   - Use `train_test_split` with stratification to create training and validation subsets.

3. **Model Fitting**
   - Initialize `RandomForestClassifier` (e.g. `n_estimators=50`, `random_state=42`).
   - Call `.fit(X_train, y_train)`.

4. **Evaluation**
   - Compute training and test accuracy.
   - Compute confusion matrix and classification report.
   - Check for class imbalance and misclassification patterns.

5. **Saving**
   - Save the trained model and encoder to `lung_model.joblib` via `joblib.dump`.

---

## 6. How to Run the System

### 6.1 Environment Setup

1. Create and activate a virtual environment (or use an existing one).
2. Install dependencies from [requirements.txt](requirements.txt):
   - `pip install -r requirements.txt`

### 6.2 Training Pipeline in the Notebook

1. Open [Model.ipynb](Model.ipynb) in VS Code or Jupyter.
2. Run the cells in order:
   - **Cell 1** – Locate the ZIP file (either in the project root or via user input).
   - **Cell 2** – Extract ZIP contents into `classification_folder/`.
   - **Cells 3–5** – Inspect and validate folder structure, list JSON and WAV files.
   - **Clipping cells** – Generate `clips/` folder with labeled segment WAV files.
   - **Feature extraction cells** – Build `features_1684` (single id) and `features_all` (multiple ids).
   - **Model training cell** – Train RandomForest, evaluate, and print metrics.
   - **Model saving cell** – Save `lung_model.joblib`.

After this, the trained model is stored on disk and is ready for deployment.

### 6.3 Running the FastAPI Service (if `main.py` and `lung_pipeline.py` are present)

1. Ensure `lung_model.joblib` exists in the project directory.
2. Make sure the environment has FastAPI, Uvicorn, and python-multipart installed (all in [requirements.txt](requirements.txt)).
3. Start the server from the project directory:
   - `uvicorn main:app --reload`
4. Open your browser at:
   - `http://127.0.0.1:8000/docs`
   - Use the `/predict` endpoint, upload a WAV file, and view the predicted label.

---



