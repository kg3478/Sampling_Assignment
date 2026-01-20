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

                        Sampling1_RUS   Sampling2_ROS   Sampling3_SMOTE  Sampling4_ADASYN   Sampling5_SMOTEENN
M1_LogisticRegression          50.0          91.7            91.7            91.29              94.53

M2_DecisionTree               66.67         98.47           98.25            98.04              97.95

M3_RandomForest                50.0         99.78           99.13            99.35              99.54

M4_SVM                         50.0         97.82           98.25            98.26              98.86

M5_GradientBoosting            50.0         99.56           99.34            98.69              99.77


Best Sampling Technique for Each Model:

M1_LogisticRegression --> Sampling5_SMOTEENN with Accuracy = 94.53%

M2_DecisionTree --> Sampling2_ROS with Accuracy = 98.47%

M3_RandomForest --> Sampling2_ROS with Accuracy = 99.78%

M4_SVM --> Sampling5_SMOTEENN with Accuracy = 98.86%

M5_GradientBoosting --> Sampling5_SMOTEENN with Accuracy = 99.77%


## Conclusion
SMOTEENN sampling provided the best overall performance across all models by generating synthetic minority samples and removing noisy majority samples.
