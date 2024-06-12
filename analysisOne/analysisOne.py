# Assessment-of-Retrospective-Engagement-Scenarios System
# Analysis One: Most influential factors to victory or defeat

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt


# Load data
battles_df = pd.read_csv("data/battles.csv").iloc[:625]
copy_battles_df = battles_df.copy()

# Data preprocessing
battles_df = battles_df.loc[:, "postype":]
battles_df = battles_df.drop(battles_df.iloc[:, 34:44], axis=1)
battles_df = battles_df.drop("crit", axis=1)

# Data preprocessing
battlesCategoricalInfo = [col for col in battles_df if battles_df[col].dtype == object]
battlesNumericalInfo = [
    col
    for col in battles_df
    if (battles_df[col].dtype == np.float64 or battles_df[col].dtype == np.int64)
]

# Handle missing values
battles_df[battlesCategoricalInfo] = battles_df[battlesCategoricalInfo].fillna(
    "missing"
)

# Fill NaN values in wina column with -1 based on research
battles_df["wina"] = battles_df["wina"].fillna(-1)

selected_features = battlesNumericalInfo.copy()
selected_features.remove("wina")

X = battles_df[selected_features + battlesCategoricalInfo]
y = battles_df["wina"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing pipelines for numerical and categorical data
numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])

categorical_pipeline = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, selected_features),
        ("cat", categorical_pipeline, battlesCategoricalInfo),
    ]
)

# Model training with class weights to handle imbalance
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(class_weight="balanced")),
    ]
)

model.fit(X_train, y_train)

# Feature importance
classifier = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocessor"]

# Get feature importances
feature_importance = classifier.feature_importances_

# Get feature names from preprocessor
num_features = preprocessor.transformers_[0][2]
cat_features = (
    preprocessor.transformers_[1][1]
    .named_steps["onehot"]
    .get_feature_names_out(battlesCategoricalInfo)
)
all_feature_names = np.concatenate([num_features, cat_features])


def modelEval():
    # Model evaluation

    # Check class distribution
    print("Class distribution in 'wina':", Counter(y))

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def featureWeights(sortedForm=True, percentForm=False, plot=False):
    # Calculate weight of each feature
    total_importance = sum(feature_importance)
    weights = feature_importance / total_importance

    if sortedForm:
        featureWeights = dict()

    if not sortedForm:
        print("Feature Weights:")

    for feature, weight in zip(all_feature_names, weights):
        if sortedForm:
            featureWeights.update({feature: weight})

        if not sortedForm:
            print(
                f"{feature}: {round(weight * 100) if percentForm else weight}"
                + ("%" if percentForm else "")
            )

    if sortedForm:
        sortedWeights = {
            k: v
            for k, v in sorted(
                featureWeights.items(), key=lambda item: item[1], reverse=True
            )
        }
        print("Feature Weights (Sorted):")
        for feature in sortedWeights:
            print(
                f"{feature}: {round(sortedWeights[feature] * 100) if percentForm else sortedWeights[feature]}"
                + ("%" if percentForm else "")
            )
    if plot:
        # Plot Barplot
        feature_weights_df = pd.DataFrame.from_dict(
            sortedWeights, orient="index", columns=["weight"]
        )
        plt.figure(figsize=(12, 8))
        sns.barplot(
            x=feature_weights_df["weight"],
            y=feature_weights_df.index,
            palette="viridis",
        )
        plt.title("Feature Weights Bar Plot")
        plt.xlabel("Weight")
        plt.ylabel("Feature")
        plt.show()


if __name__ == "__main__":
    modelEval()
    featureWeights(percentForm=True, plot=True)
