# Network-IDS-ML
# Network Intrusion Detection System (IDS) – ML Project

**A machine learning-based IDS for detecting network attacks with high accuracy and modular design**

## Overview
This project is a robust system for network attack detection using machine learning.

### 1. Datasets
- **CIC-IDS2017** dataset – modern cybersecurity benchmark  
- Focused on DDoS attacks  
- Real network data from university environment  

### 2. Data Pipeline
- **Loading & cleaning**: automatic handling of missing values  
- **Feature engineering**: removal of unnecessary columns (IP addresses, timestamps)  
- **Label processing**: binary labels (0 = normal, 1 = attack)  
- **Normalization**: StandardScaler for optimal performance  

### 3. Model & Architecture
- **Random Forest Classifier** with 100 trees  
- **99.98% accuracy** on test data  
- Only **13 false negatives** out of 67,724 cases  
- **Modular design**: model and scaler saved for future use  

### 4. Advanced Features
- **Error analysis**: detailed metrics on misclassified samples  
- **Cross-dataset testing**: evaluated on unseen attack types  
- **Automated reports**: confusion matrix, classification reports, feature importance  

## Tech Specs
| Component | Specification |
|-----------|---------------|
| Language | Python 3.11+ |
| Libraries | scikit-learn, pandas, numpy, matplotlib, seaborn, joblib |
| Model | Random Forest (100 trees) |
| Accuracy | 99.98% |
| False Negatives | 13 / 67,724 |
| Features | 78 network features |

## Key Advantages
1. High accuracy – suitable even for commercial-grade solutions  
2. Low false positive rate – critical in cybersecurity  
3. Modular – easy retraining for new data  
4. Transparent – detailed performance reports  
5. Real-world applicability – tested on real network traffic  

## Potential Applications
- Real-time IDS systems  
- Security Operations Center automation  
- Research on new attack types  
- Educational purposes – ML in cybersecurity  
