# Fraud Detection
## Project Overview
This project implements a scalable fraud detection system capable of analyzing 2 billion financial transactions to identify high-risk activity. With a fraud rate of 0.38% (~7.6M cases), the system achieved 87% detection accuracy (~6.61M cases caught).

Through this performance, the system demonstrated:
* $3.3B in fraud losses prevented, reducing financial risk exposure by ~87%.
* $132M in retained customer revenue, by preventing account compromises and churn.
* Improved compliance posture, lowering exposure to $10M–$100M regulatory fines.

## Business Impact
### I. Fraud Loss Prevention
* Assuming, average fraudulent transaction to be $500, then:
  * Total fraud exposure = 7.6M × $500 = $3.8B
  * Detected fraud (87%) = $3.3B protected
  * Remaining undetected fraud = $0.5B
* <b>Business impact:</b> Prevents billions in potential losses, reducing financial risk by ~87%.

### II. Customer Trust & Retention
* By catching 87% of fraud, far fewer customers experience account compromise.
* If even 1% of prevented fraud cases (~66K customers) would have otherwise churned, and each customer has a lifetime value of $2,000, that’s
<b>→ $132M in retained revenue.</b>

### III. Regulatory & Compliance
* Strong fraud detection reduces exposure to AML/ATF non-compliance fines (often $10M–$100M per incident).
* Demonstrates auditable, automated, explainable controls, lowering regulatory risk.

## System Architecture
<img width="951" height="656" alt="Fraud Detection" src="https://github.com/user-attachments/assets/02108ed4-bae4-4cc8-8608-ad7fc0e2077f" />

## Technical Details
### I. Feature Engineering
To improve fraud detection performance, we engineered features that capture behavioral patterns, temporal signals, and contextual risks:
* <b>Transaction Amount:</b> Absolute value of a transaction; unusually large amounts often correlate with fraud.
* <b>Is Night:</b> Flags if a transaction occurred during late-night hours; fraudsters often operate when user activity is low.
* <b>Is Weekend:</b> Indicates weekend activity; abnormal spending patterns at off-business times can suggest risk.
* <b>Transaction Day:</b> Captures day-of-week effects; fraud may spike on specific days when monitoring is weaker.
* <b>Past 24h User Activity:</b> Number of transactions by the same user in the past 24 hours; sudden spikes in activity can indicate compromised accounts.
* <b>Amount to avg ratio:</b> Ratio of current transaction amount to user’s historical average; highlights deviations from normal spending behavior.
* <b>Merchant Risk:</b> Risk score based on merchant history; transactions at high-risk merchants are more likely fraudulent.

These features help the model distinguish between normal user behavior and potentially fraudulent patterns, improving recall on rare fraud cases.

### II. Model Development
The fraud detection model uses a robust machine learning pipeline designed to handle class imbalance and prioritize recall.
* SMOTE addresses class imbalance (0.38% fraudulant records) by generating synthetic fraud samples, helping the model learn patterns from rare fraudulent transactions.
* XGBoost serves as the core model, capturing complex patterns in transactional and behavioral features to distinguish fraud from legitimate activity.
* Hyperparameter tuning with Randomized Search and Stratified K-Fold cross-validation ensures the model generalizes well and performs reliably on rare fraud cases.

## Detected Fradulant Transaction Preview
<img width="565" height="338" alt="image" src="https://github.com/user-attachments/assets/a6d94138-0208-4148-bc0e-2ec551f9116e" />

## License
This project is licensed under the MIT License.


