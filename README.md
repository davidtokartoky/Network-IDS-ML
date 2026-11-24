# Network-IDS-ML

An experimental network attack detection system using machine learning. It performs binary classification of network flows (normal traffic vs. attack) on the CIC-IDS2017 dataset.

**Project status:** A study / experimental project created with the help of AI tools. It is not intended for production deployment. Important evaluation limitations are described in the **Limitations** section.

## What the Project Does

The `livecybersecurity.py` script loads CSV files from the CIC-IDS2017 dataset, preprocesses them, trains a classifier, saves the model and scaler, and then evaluates the model on multiple files. It processes data offline from CSV files and does not capture or analyze network traffic in real time.

## Dataset

- CIC-IDS2017 from the Canadian Institute for Cybersecurity (`MachineLearningCVE` files, CSVs with approximately 78 network flow features).
- The data was generated in a laboratory testbed with simulated traffic and attacks, not from a deployed production network.
- The dataset is not included in the repository and must be downloaded separately.

### Files Used

| Purpose | File |
| :--- | :--- |
| Training and test | `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` |
| Training only | `Wednesday-workingHours.pcap_ISCX.csv` |
| Training and test | `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` |
| Training and test | `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` |
| Test only | `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` |

## Data Pipeline

1. **Chunked loading** (chunks of 100,000 rows) to avoid memory issues.
2. **Cleaning:** `inf` values are converted to `NaN`, columns with more than 30% missing values are dropped, and remaining missing values are filled with the mean (numerical) or mode (categorical).
3. **Labels:** binary — `BENIGN = 0`, all attacks `= 1`.
4. **Feature selection:** dropped columns `Flow ID`, `Timestamp`, `Src IP`, and `Dst IP`.
5. **Normalization:** `StandardScaler`.
6. **Class balancing:** SMOTE (oversampling) on the training data.

## Model

- `XGBClassifier` (XGBoost), 100 trees, `random_state=42`.
- The trained model is saved to `trained_model.pkl` and the scaler to `scaler.pkl` (via `joblib`), so they can be reused without retraining.

## Limitations and Honest Evaluation

Before using the results, read these caveats:

- Test data overlaps with training data. Three of the four tested files (`DDoS`, `WebAttacks`, `Infiltration`) are also in the training set. Metrics on them therefore do not measure performance on unseen data and are optimistic.
- Only the `PortScan` file is genuinely new. In the code, it is listed in the test dictionary under the key `"Botnet"`, even though it is the PortScan attack file — the naming is incorrect.
- The results in `test_results.json` are therefore not an independent estimate of performance on unknown network traffic.
- A previous version of this README mentioned Random Forest and 99.98% accuracy. Those figures did not match the current code (which uses XGBoost) and did not come from an independent test, so they have been removed.
- The project has not been tested on real-world network traffic, and conclusions about practical usability cannot be drawn from it.

## Running the Project

1. Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn joblib matplotlib seaborn
