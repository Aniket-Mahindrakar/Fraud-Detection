# Fraud Detection
## Value Proposition
This project delivers a machine learning–based fraud detection system designed to catch fraudulent transactions early and minimize financial losses. By prioritizing high recall (87%), the model ensures most fraud attempts are flagged, helping organizations:
* Prevent revenue loss from undetected fraud.
* Adapt to evolving fraud patterns beyond static rule-based systems.
* Protect customer trust through proactive risk management.

## System Architecture
<img width="951" height="656" alt="Fraud Detection" src="https://github.com/user-attachments/assets/02108ed4-bae4-4cc8-8608-ad7fc0e2077f" />

## Feature Engineering
To improve fraud detection performance, we engineered features that capture behavioral patterns, temporal signals, and contextual risks:
* <b>transaction_amount:</b> Absolute value of a transaction; unusually large amounts often correlate with fraud.
* <b>is_night:</b> Flags if a transaction occurred during late-night hours; fraudsters often operate when user activity is low.
* <b>is_weekend:</b> Indicates weekend activity; abnormal spending patterns at off-business times can suggest risk.
* <b>transaction_day:</b> Captures day-of-week effects; fraud may spike on specific days when monitoring is weaker.
* <b>user_activity_24h:</b> Number of transactions by the same user in the past 24 hours; sudden spikes in activity can indicate compromised accounts.
* <b>amount_to_avg_ratio:</b> Ratio of current transaction amount to user’s historical average; highlights deviations from normal spending behavior.
* <b>merchant_risk:</b> Risk score based on merchant history; transactions at high-risk merchants are more likely fraudulent.

These features help the model distinguish between normal user behavior and potentially fraudulent patterns, improving recall on rare fraud cases.

## Model Development
The fraud detection model uses a robust machine learning pipeline designed to handle class imbalance and prioritize recall.
* SMOTE addresses class imbalance (0.38% fraudulant records) by generating synthetic fraud samples, helping the model learn patterns from rare fraudulent transactions.
* XGBoost serves as the core model, capturing complex patterns in transactional and behavioral features to distinguish fraud from legitimate activity.
* Hyperparameter tuning with Randomized Search and Stratified K-Fold cross-validation ensures the model generalizes well and performs reliably on rare fraud cases.
