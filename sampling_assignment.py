import pandas as pd
import numpy as np

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Sampling Techniques
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTEENN

df = pd.read_csv("Creditcard_data.csv")

print("Dataset Shape:", df.shape)
print("\nClass Distribution:")
print(df['Class'].value_counts())


X = df.drop('Class', axis=1)
y = df['Class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


samplers = {
    "Sampling1_RUS": RandomUnderSampler(random_state=42),
    "Sampling2_ROS": RandomOverSampler(random_state=42),
    "Sampling3_SMOTE": SMOTE(random_state=42),
    "Sampling4_ADASYN": ADASYN(random_state=42),
    "Sampling5_SMOTEENN": SMOTEENN(random_state=42)
}


models = {
    "M1_LogisticRegression": LogisticRegression(max_iter=1000),
    "M2_DecisionTree": DecisionTreeClassifier(),
    "M3_RandomForest": RandomForestClassifier(),
    "M4_SVM": SVC(),
    "M5_GradientBoosting": GradientBoostingClassifier()
}


results = pd.DataFrame(index=models.keys(), columns=samplers.keys())

for s_name, sampler in samplers.items():
    

    X_resampled, y_resampled = sampler.fit_resample(X_scaled, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, 
        test_size=0.3, 
        random_state=42, 
        stratify=y_resampled
    )
    
   
    for m_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        results.loc[m_name, s_name] = round(acc, 2)


print("\nFinal Accuracy Table:\n")
print(results)

#Accuracy_Graph
mport matplotlib.pyplot as plt
results_numeric = results.astype(float)

plt.figure(figsize=(10,6))

for model in results_numeric.index:
    plt.plot(results_numeric.columns, results_numeric.loc[model], marker='o')

plt.xlabel("Sampling Techniques")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison Across Sampling Techniques")
plt.xticks(rotation=45)
plt.legend(results_numeric.index)

plt.tight_layout()
plt.show()

print("\nBest Sampling Technique for Each Model:\n")

for model in results.index:
    best_sampling = results.loc[model].astype(float).idxmax()
    best_accuracy = results.loc[model].astype(float).max()
    print(f"{model} --> {best_sampling} with Accuracy = {best_accuracy}%")
