# Assessment-of-Retrospective-Engagement-Scenarios System
# Analysis Two: Focusing more in depth on weather, terrain,
# element of surprise, and fortifiation

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
battle = pd.read_csv("data/battles.csv").iloc[:625]
terrain = pd.read_csv("data/terrain.csv")
weather = pd.read_csv("data/weather.csv")

df = pd.merge(battle, terrain, on="isqno")
df = pd.merge(df, weather, on="isqno")

# Fill NaN values in wina column with -1 based on research
df["wina"] = df["wina"].fillna(-1)
df = df[
    [
        "surpa",
        "post1",
        "wx1",
        "wx2",
        "wx3",
        "wx4",
        "wx5",
        "terra1",
        "terra2",
        "terra3",
        "aeroa",
        "wina",
    ]
]

# Extracting relevant features
df_combined = df[
    [
        "surpa",
        "post1",
        "wx1",
        "wx2",
        "wx3",
        "wx4",
        "wx5",
        "terra1",
        "terra2",
        "aeroa",
        "wina",
    ]
]


# Handling missing values and encoding categorical variables
def preprocess_data(df):
    df.loc[:, "wina"] = df["wina"].apply(lambda x: x + 1 if x == -1 else x)
    df = df.dropna()  # Drop rows with missing values
    df = df.astype({"wina": "int"})  # Convert 'wina' to integer type

    # Dummy encode 'post1'
    df["post1"] = df["post1"].apply(lambda x: x[0])
    df = pd.get_dummies(df, columns=["post1"])

    df["surpa"] = df["surpa"].map(
        {
            0: "Neither",
            1: "Minor (A)",
            2: "Substantial (A)",
            3: "Complete (A)",
            -1: "Minor (D)",
            -2: "Substantial (D)",
            -3: "Comlete (D)",
        }
    )
    df = pd.get_dummies(df, columns=["surpa"])

    df["wx1"] = df["wx1"].map({"W": "Wet", "D": "Dry"})
    df = pd.get_dummies(df, columns=["wx1"])
    df["wx2"] = df["wx2"].map(
        {"S": "Sunny", "L": "Light Precipitation", "H": "Heavy Precipitation"}
    )
    df = pd.get_dummies(df, columns=["wx2"])
    df["wx3"] = df["wx3"].map({"C": "Cold", "H": "Hot", "T": "Temperate"})
    df = pd.get_dummies(df, columns=["wx3"])
    df["wx4"] = df["wx4"].map(
        {"S": "Summer", "$": "Spring", "W": "Winter", "F": "Fall"}
    )
    df = pd.get_dummies(df, columns=["wx4"])
    df["wx5"] = df["wx5"].map({"E": "Tropical", "D": "Desert", "T": "Temperate"})
    df = pd.get_dummies(df, columns=["wx5"])
    df["terra1"] = df["terra1"].map({"R": "Rolling", "G": "Rugged", "F": "Flat"})
    df = pd.get_dummies(df, columns=["terra1"])
    df["terra2"] = df["terra2"].map(
        {"B": "Bare", "M": "Mixed", "D": "Desert", "W": "Heavily Wooded"}
    )
    df = pd.get_dummies(df, columns=["terra2"])

    return df


df_combined = preprocess_data(df_combined)

# Splitting data into features (X) and target (y)
X = df_combined.drop(columns=["wina"])
y = df_combined["wina"]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a Random Forest Classifier
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Feature importance
feature_importance = model.feature_importances_
features = X.columns


def modelEval():
    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Overall Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    print("Overall Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def featureWeights():
    # Grouping feature names
    grouped_features = {
        "surpa_Neither": "surpa",
        "surpa_Minor (A)": "surpa",
        "surpa_Substantial (A)": "surpa",
        "surpa_Complete (A)": "surpa",
        "surpa_Minor (D)": "surpa",
        "surpa_Substantial (D)": "surpa",
        "surpa_Complete (D)": "surpa",
        "post1_A": "post1",
        "post1_D": "post1",
        "wx1_Dry": "wx1",
        "wx1_Wet": "wx1",
        "wx2_Heavy Precipitation": "wx2",
        "wx2_Sunny": "wx2",
        "wx2_Light Precipitation": "wx2",
        "wx3_Cold": "wx3",
        "wx3_Hot": "wx3",
        "wx3_Temperate": "wx3",
        "wx4_Fall": "wx4",
        "wx4_Spring": "wx4",
        "wx4_Summer": "wx4",
        "wx4_Winter": "wx4",
        "wx5_Desert": "wx5",
        "wx5_Temperate": "wx5",
        "wx5_Tropical": "wx5",
        "terra1_Flat": "terra1",
        "terra1_Rolling": "terra1",
        "terra1_Rugged": "terra1",
        "terra2_Bare": "terra2",
        "terra2_Desert": "terra2",
        "terra2_Heavily Wooded": "terra2",
        "terra2_Mixed": "terra2",
    }

    # Print feature importances and accuracy for each feature
    printed_features = set()
    for feature in grouped_features.values():
        if feature in printed_features:
            continue
        print(f"\nFeature Weights for {feature}:")
        printed_features.add(feature)
        # Train a Random Forest Classifier for this feature only
        feature_columns = [col for col in X.columns if col.startswith(feature)]
        feature_importance_feature = []
        for col in feature_columns:
            feature_index = X.columns.get_loc(col)
            feature_importance_feature.append(feature_importance[feature_index])
            print(f"{col}: {round(feature_importance[feature_index] * 100)}%")


if __name__ == "__main__":
    modelEval()
    featureWeights()
