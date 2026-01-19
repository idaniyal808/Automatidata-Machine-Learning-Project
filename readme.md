Project Title: NYC Taxi Tip Prediction ModelClient: New York City Taxi & Limousine Commission (TLC)Objective: Develop a machine learning solution to increase driver revenue by identifying "generous tippers" (tips $\ge$ 20%).MethodologyData Source: 2017 Yellow Taxi Trip Data (~22k records).Feature Engineering: Engineered 300+ features, including pickup/dropoff time bins (am_rush, pm_rush), trip duration means, and predicted fare amounts.Modeling: Evaluated Random Forest and XGBoost classifiers.Selection Metric: F1-Score was chosen to balance the cost of false positives (driver disappointment) and false negatives (missed revenue).Key FindingsThe Model: The Random Forest model achieved an F1 score of 0.72 and an accuracy of 68.7%.Impact: The model correctly identifies ~78% of generous tippers (Recall), providing drivers with a significant advantage over random chance.Top Predictors: VendorID, predicted_fare, and mean_duration were the strongest indicators of whether a passenger would tip generously.


# 🚕 NYC Taxi Revenue Optimization: Generous Tipper Prediction
**Machine Learning Case Study: Predicting High-Value Outcomes for Taxi Drivers**

## 📌 Business Problem
Taxi drivers in New York City rely heavily on tips for their livelihood. The New York City Taxi & Limousine Commission (TLC) requested a model to help drivers identify passengers likely to be "Generous Tippers" (tips ≥ 20%). 

Initially, the request was to predict "Non-Tippers," but I adjusted the objective to focus on **incentivizing generosity** rather than discriminating against service users, ensuring an ethical approach to data science.

## 🛠️ Methodology & Tech Stack
I followed the **PACE** (Plan, Analyze, Construct, Execute) framework to ensure a structured delivery.

- **Tools:** Python (Pandas, NumPy), Scikit-Learn, XGBoost, Matplotlib/Seaborn.
- **Feature Engineering:** - Engineered 300+ binary features using One-Hot Encoding.
    - Created time-of-day bins (`am_rush`, `daytime`, `pm_rush`, `nighttime`).
    - Handled floating-point arithmetic errors for target labeling.
- **Models Evaluated:** Random Forest (Champion) and XGBoost.

## 📊 Results & Key Findings
The Random Forest model was selected as the champion due to its superior F1-score and ability to balance precision and recall.

| Metric | Random Forest Score | Significance |
| :--- | :--- | :--- |
| **F1-Score** | **0.7235** | High balance between finding tippers and accuracy. |
| **Recall** | **0.7791** | Corrected identifies ~78% of all generous tippers. |
| **Accuracy** | **0.6865** | Outperforms random chance by nearly 50%. |

### Key Predictors of Generosity
The model identified that **VendorID**, **predicted_fare**, and **mean_duration** are the strongest indicators of whether a passenger will tip 20% or more.



## 💡 Ethical Conclusion
By shifting the model’s focus to "Generous Tippers," we created a tool that helps drivers maximize their income without compromising the TLC’s commitment to providing equal service access to all New Yorkers regardless of their tipping predicted behavior.

## 🚀 Reproduction
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run `nyc_taxi_model.py` to see the full modeling logic and evaluation.
