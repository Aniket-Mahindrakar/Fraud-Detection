# Fraud Detection
## Project Overview
This project implements a scalable fraud detection system capable of analyzing 2 billion transactions with a 0.38% fraud rate (~7.6M cases). The system achieved 87% detection accuracy (~6.61M cases caught), delivering significant business value.

## Business Impact
### I. Fraud Loss Prevention
* Total fraud exposure: 7.6M × $500 = $3.8B
* Detected fraud (87%): $3.3B protected
* Remaining undetected: $0.5B

### II. Customer Trust & Retention
* ~66K customers retained (1% of prevented fraud cases)
* Lifetime value per customer: $2,000
* Revenue retained: $132M

### III. Regulatory & Compliance
* Reduces exposure to AML/ATF fines ($10M–$100M per incident)
* Provides auditable, explainable fraud controls

## System Architecture
<img width="951" height="656" alt="Fraud Detection" src="https://github.com/user-attachments/assets/02108ed4-bae4-4cc8-8608-ad7fc0e2077f" />

## Technical Details
### I. Feature Engineering
To improve fraud detection performance, we engineered features that capture behavioral patterns, temporal signals, and contextual risks:

| Feature               | Description                                 | Fraud Relevance                                      |
|-----------------------|---------------------------------------------|-----------------------------------------------------|
| Transaction Amount    | Absolute value of transaction               | Large amounts often correlate with fraud           |
| Is night              | Flags late-night transactions               | Fraudsters target low-activity hours              |
| Is weekend            | Flags weekend transactions                  | Unusual spending times may indicate risk          |
| Transaction day       | Day-of-week effects                          | Fraud may spike on certain days                    |
| Past 24h user activity| Number of transactions in past 24 hours    | Sudden spikes indicate potential fraud            |
| Amount to avg ratio   | Ratio of current transaction to historical average | Detects deviations from normal behavior       |
| Merchant risk         | Risk score based on merchant history        | High-risk merchants more likely for fraud         |

### II. Model Development
The fraud detection model uses a robust machine learning pipeline designed to handle class imbalance and prioritize recall.
* SMOTE addresses class imbalance (0.38% fraudulent records) by generating synthetic fraud samples, helping the model learn patterns from rare fraudulent transactions.
* XGBoost serves as the core model, capturing complex patterns in transactional and behavioral features to distinguish fraud from legitimate activity.
* Hyperparameter tuning with Randomized Search and Stratified K-Fold cross-validation ensures the model generalizes well and performs reliably on rare fraud cases.

## Detected Fraudulent Transaction Preview
<img width="565" height="338" alt="image" src="https://github.com/user-attachments/assets/a6d94138-0208-4148-bc0e-2ec551f9116e" />

## License
This project is licensed under the MIT License.
