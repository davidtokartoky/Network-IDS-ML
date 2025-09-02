# Import necessary libraries
# You may need to install imblearn and xgboost: pip install imblearn xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
import os
import json
warnings.filterwarnings('ignore')

# Import XGBoost and the SMOTE library for over-sampling
from imblearn.over_sampling import SMOTE
import xgboost as xgb

def check_file_paths(file_paths):
    """
    Checks if all required files exist before starting the process.
    """
    all_exist = True
    for path in file_paths:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}. Please check your file paths and directory structure.")
            all_exist = False
    return all_exist

def preprocess_data(file_path):
    """
    Preprocess the CICIDS2017 dataset for anomaly detection.
    This version loads the data in chunks to handle large files and memory issues.
    """
    full_path = file_path
    
    print(f"Loading dataset from: {full_path}")
    try:
        # Load the file in chunks to avoid memory errors
        # Chunksize is set to 100,000, which is a good starting point for large datasets
        chunk_list = []
        i = 0
        for chunk in pd.read_csv(full_path, chunksize=100000):
            chunk_list.append(chunk)
            i += 1
            print(f"Loading chunk {i}...")
            
        print("Concatenating chunks...")
        df = pd.concat(chunk_list, ignore_index=True)
        print(f"Successfully loaded {len(df)} rows.")

    except FileNotFoundError:
        print(f"ERROR: File not found at {full_path}. Please check the file path and name.")
        return None, None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading the file: {e}")
        return None, None
        
    df.columns = df.columns.str.strip()  # Removes leading/trailing spaces from column names
    
    # Basic cleaning
    print("Cleaning data...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    missing_values = df.isnull().sum()
    print(f"Total missing values: {missing_values.sum()}")
    
    # Drop columns with too many missing values (>30%)
    threshold = 0.3 * len(df)
    df = df.dropna(axis=1, thresh=threshold)
    
    # Fill remaining NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Handle the Label column
    label_col = None
    for col in df.columns:
        if 'label' in col.lower():
            label_col = col
            break
    
    X = None
    y = None
    
    if label_col:
        print(f"Found label column: {label_col}")
        le = LabelEncoder()
        if df[label_col].dtype == 'object':
            df[label_col] = le.fit_transform(df[label_col].astype(str))
            try:
                benign_code = le.transform(['BENIGN'])[0]
                df[label_col] = df[label_col].apply(lambda x: 0 if x == benign_code else 1)
            except:
                normal_label = df[label_col].value_counts().idxmax()
                df[label_col] = df[label_col].apply(lambda x: 0 if x == normal_label else 1)
        else:
            unique_labels = df[label_col].unique()
            if len(unique_labels) > 2:
                normal_label = df[label_col].value_counts().idxmax()
                df[label_col] = df[label_col].apply(lambda x: 0 if x == normal_label else 1)
    
        y = df[label_col]
        X = df.drop(label_col, axis=1)
    else:
        print("Warning: No label column found. Creating a dummy label column.")
        y = pd.Series([0] * len(df))  # All samples considered normal
        X = df
    
    # Drop irrelevant columns
    cols_to_drop = [col for col in X.columns if 'Flow ID' in col or 'Timestamp' in col or 'Src IP' in col or 'Dst IP' in col]
    X = X.drop(columns=cols_to_drop, errors='ignore')
    
    # Encode any remaining categorical columns
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"Found non-numeric columns: {list(non_numeric_cols)}")
        for col in non_numeric_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y

def analyze_errors(X_test, y_test, y_pred, X_test_original, model, X_train):
    """
    Analyzes misclassified cases
    """
    y_pred_series = pd.Series(y_pred, index=y_test.index)
    
    wrong_indices = y_test != y_pred_series
    X_wrong = X_test_original.loc[wrong_indices]
    y_wrong_true = y_test.loc[wrong_indices]
    y_wrong_pred = y_pred_series.loc[wrong_indices]
    
    print(f"Počet chybne klasifikovaných prípadov: {len(X_wrong)}")
    print("\n" + "="*60)
    print("ANALÝZA CHYBNÝCH PRÍPADOV")
    print("="*60)
    
    if len(X_wrong) > 0:
        print("\n1. ŠTATISTIKY CHYBNÝCH PRÍPADOV:")
        print("-" * 40)
        
        feature_importance = model.feature_importances_
        top_features_indices = np.argsort(feature_importance)[-5:]
        top_features = X_train.columns[top_features_indices]
        
        print("Hodnoty top features v chybných prípadoch:")
        for feature in top_features:
            print(f"{feature}: {X_wrong[feature].mean():.4f} (priemer)")
        
        print("\nPorovnanie s celým testovacím setom:")
        for feature in top_features:
            print(f"{feature}: {X_test_original[feature].mean():.4f} (priemer v test set)")
        
        print("\nRozdiely v štandardných odchýlkach:")
        for feature in top_features:
            wrong_std = X_wrong[feature].std()
            test_std = X_test_original[feature].std()
            print(f"{feature}: {wrong_std:.4f} vs {test_std:.4f}")
        
        error_analysis_df = X_wrong.copy()
        error_analysis_df['true_label'] = y_wrong_true
        error_analysis_df['predicted_label'] = y_wrong_pred
        error_analysis_df.to_csv('error_analysis.csv', index=False)
        print("\nChybné prípady uložené do 'error_analysis.csv'")
        
        return X_wrong, top_features
    else:
        print("Žiadne chybné prípady na analýzu!")
        return None, None

def save_results_to_file(results, filename='test_results.json'):
    """
    Saves the test results to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nVýsledky testovania uložené do '{filename}'.")


# Main execution
if __name__ == "__main__":
    
    # Check if this script is being executed from the correct location
    if not os.path.basename(os.getcwd()) == "ids2017-project":
        print("ERROR: This script must be run from the 'ids2017-project' directory.")
        print(f"Current directory is: {os.getcwd()}")
        print("Please use the 'cd' command to navigate to the correct directory before running the script.")
        exit()

    print("Štartujem skript...")
    
    # === FÁZA 1: TRÉNOVANIE MODELU NA VIACERÝCH DÁTACH ===
    print("="*50)
    print("FÁZA 1: TRÉNOVANIE MODELU")
    print("Trénujem model na viacerých datasetoch, aby sa naučil detegovať rôzne útoky.")
    print("==================================================")
    
    # Cesty k datasetom, ktoré chceme použiť na trénovanie
    # Pridané aj ďalšie dáta na tréning, aby bol model robustnejší
    train_file_paths = [
        "MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv",
        "MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv" 
    ]
    
    # Check if the training files exist
    if not check_file_paths(train_file_paths):
        print("Trénovanie zrušené, pretože neboli nájdené všetky trénovacie dáta.")
        exit()

    all_train_data = []
    all_train_labels = []
    
    for file_path in train_file_paths:
        X_raw, y = preprocess_data(file_path)
        if X_raw is not None:
            all_train_data.append(X_raw)
            all_train_labels.append(y)
            
    if not all_train_data:
        print("Trénovanie zrušené, pretože neboli načítané žiadne trénovacie dáta.")
    else:
        # Konkatenácia všetkých trénovacích dát do jedného DataFrame
        combined_X_train = pd.concat(all_train_data, ignore_index=True)
        combined_y_train = pd.concat(all_train_labels, ignore_index=True)
        
        # Normalizácia
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(combined_X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=combined_X_train.columns)
        
        # === APLIKÁCIA SMOTE NA TRÉNOVACÍCH DÁTACH ===
        print("\nPoužívam SMOTE na vyváženie trénovacích dát...")
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train_scaled, combined_y_train)
        print(f"Pôvodný počet vzoriek: {len(X_train_scaled)}")
        print(f"Počet vzoriek po SMOTE: {len(X_res)}")
        
        # Zmena modelu z RandomForest na XGBoost
        model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        print("\nTrénujem model na vyvážených kombinovaných dátach...")
        model.fit(X_res, y_res)
        
        # Save the trained model and scaler
        joblib.dump(model, 'trained_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        print("\nModel a scaler úspešne uložené pre budúce použitie.")
        
        
        # === FÁZA 2: TESTOVANIE ULOŽENÉHO MODELU NA NOVÝCH DÁTACH ===
        print("\n" + "="*50)
        print("FÁZA 2: TESTOVANIE MODELU")
        print("Načítavam model a testujem ho na viacerých nových datasetoch.")
        print("="*50)
        
        test_file_paths = {
            "DDOS": "MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "WebAttacks": "MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Infiltration": "MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Botnet": "MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
        }
        
        # Check if the testing files exist
        if not check_file_paths(test_file_paths.values()):
            print("Testovanie zrušené, pretože neboli nájdené všetky testovacie dáta.")
            exit()
            
        all_results = {}
        
        # Testovanie modelu na každom testovacom datasete
        for attack_name, test_path in test_file_paths.items():
            print(f"\nTestujem na datasete: {attack_name}")
            
            X_test_raw, y_test = preprocess_data(test_path)
            
            if X_test_raw is not None:
                # Reorder columns to match the training set order
                X_test_raw = X_test_raw.reindex(columns=X_train_scaled.columns, fill_value=0)
                
                # Load the trained model and scaler
                loaded_model = joblib.load('trained_model.pkl')
                loaded_scaler = joblib.load('scaler.pkl')
                
                # Scale the new test data using the *loaded* scaler
                X_test_scaled = loaded_scaler.transform(X_test_raw)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_raw.columns)
                
                # Make predictions with the loaded model
                y_pred = loaded_model.predict(X_test_scaled)
                
                # Evaluate model on the new data
                print("\nVÝSLEDKY VYHODNOTENIA NA NOVÝCH DÁTACH:")
                print("="*50)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                cm = confusion_matrix(y_test, y_pred).tolist()
                
                print(f"Presnosť (Accuracy): {accuracy:.4f}")
                print("\nDetailný report:")
                print(classification_report(y_test, y_pred))
                print("Matica zámen (Confusion Matrix):")
                print(cm)
                
                # Ulozenie vysledkov
                all_results[attack_name] = {
                    "accuracy": accuracy,
                    "report": report,
                    "confusion_matrix": cm
                }
                
                # Analyze errors on the new data
                X_wrong, top_features = analyze_errors(X_test_scaled, y_test, y_pred, X_test_raw, loaded_model, X_train_scaled)
        
        # Ulozenie vsetkych vysledkov do JSON suboru
        save_results_to_file(all_results)
