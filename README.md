# Sampling Assignment

## Objective
To study the effect of different sampling techniques on machine learning model performance for an imbalanced credit card dataset.

## Dataset
Creditcard_data.csv  
Original class distribution:
Class 0: 763  
Class 1: 9  

## Sampling Techniques Used
1. Random Under Sampling  
2. Random Over Sampling  
3. SMOTE  
4. ADASYN  
5. SMOTEENN  

## Models Used
M1 - Logistic Regression  
M2 - Decision Tree  
M3 - Random Forest  
M4 - Support Vector Machine  
M5 - Gradient Boosting  

## Results
Dataset Shape: (772, 31)

Class Distribution:
Class
0    763
1      9
Name: count, dtype: int64

Final Accuracy Table:
<img width="868" height="132" alt="Screenshot 2026-01-21 at 12 04 19 AM" src="https://github.com/user-attachments/assets/66745b92-f2fd-438c-8f60-dfaa0f934fb8" />

Accuracy Graph: 
<img width="1000" height="600" alt="Result_Graph" src="https://github.com/user-attachments/assets/c15426bb-f98e-4f38-88ec-5f7524d63298" />

Best Sampling Technique for Each Model:

M1_LogisticRegression --> Sampling5_SMOTEENN with Accuracy = 94.53%

M2_DecisionTree --> Sampling2_ROS with Accuracy = 98.47%

M3_RandomForest --> Sampling2_ROS with Accuracy = 99.78%

M4_SVM --> Sampling5_SMOTEENN with Accuracy = 98.86%

M5_GradientBoosting --> Sampling5_SMOTEENN with Accuracy = 99.77%

##Methodology

The given credit card dataset was first analyzed and found to be highly imbalanced, with very few fraud cases compared to normal transactions. Since imbalanced data negatively affects machine learning model performance, the dataset was standardized and then balanced using five different sampling techniques: Random Under Sampling, Random Over Sampling, SMOTE, ADASYN, and SMOTEENN.

After applying each sampling technique, the balanced dataset was split into training and testing sets. Five different machine learning models — Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, and Gradient Boosting — were trained on each sampled dataset. The performance of every model-sampling pair was evaluated using accuracy as the metric.

Finally, the accuracy scores were compiled into a result table and visualized using a graph to compare how different sampling techniques influence model performance. The best sampling technique for each model was identified based on the highest accuracy achieved.

## Conclusion
SMOTEENN sampling provided the best overall performance across all models by generating synthetic minority samples and removing noisy majority samples.
