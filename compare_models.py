import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


df = pd.read_csv("Student_Performance.csv")
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

X = df[['Hours Studied', 'Previous Scores', 'Extracurricular Activities',
        'Sleep Hours', 'Sample Question Papers Practiced']]
y = df['Performance Index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    "Linear Regression":   LinearRegression(),
    "Decision Tree":       DecisionTreeRegressor(random_state=42),
    "Random Forest":       RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingRegressor(n_estimators=100, random_state=42)
}

print("\n📊 MODEL COMPARISON RESULTS")
print("=" * 45)

results = []
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    results.append({"Model": name, "MAE": round(mae, 3), "R² Score": round(r2, 4)})
    trained_models[name] = (model, preds)
    print(f"  {name:<25} MAE: {mae:.3f}   R²: {r2:.4f}")

print("=" * 45)
best = max(results, key=lambda x: x["R² Score"])
print(f"\n✅ Best Model: {best['Model']}  (R² = {best['R² Score']})")


names  = [r["Model"] for r in results]
scores = [r["R² Score"] for r in results]

plt.figure(figsize=(9, 5))
bars = plt.bar(names, scores, color=['#4C72B0','#DD8452','#55A868','#C44E52'], edgecolor='black')
plt.ylim(0, 1.05)
plt.title("Model Comparison — R² Score", fontsize=14)
plt.ylabel("R² Score")
plt.xticks(rotation=15)
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             str(score), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()
print("📁 Saved: model_comparison.png")


rf_model, rf_preds = trained_models["Random Forest"]

plt.figure(figsize=(7, 6))
plt.scatter(y_test, rf_preds, alpha=0.4, color='green')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Performance Index")
plt.ylabel("Predicted Performance Index")
plt.title("Random Forest: Actual vs Predicted")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()
print("📁 Saved: actual_vs_predicted.png")


plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()
print("📁 Saved: correlation_heatmap.png")