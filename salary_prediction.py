import pandas as pd 
# Load dataset
data = pd.read_csv("data/salary_dataset.csv")
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.scatter(data["Experience"], data["Salary"])
plt.title("Experience vs Salary")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.grid(True)
plt.savefig("experience_vs_salary.png")
print("Graph saved successfully!")
# Features (Input)
X = data[["Experience", "TestScore", "InterviewScore"]]
# Target (Output)
y = data["Salary"]
from sklearn.model_selection import train_test_split
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
from sklearn.linear_model import LinearRegression
# Create model
model = LinearRegression()
# Train model
model.fit(X_train, y_train)
print("\nModel trained successfully!")
# Make predictions on test data
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
# Evaluate Model
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)
print("\nModel Evaluation")
print("RMSE:", rmse)
print("R2 Score:", r2)
# -----------------------------
# Single Feature Model
# -----------------------------
X_single = data[["Experience"]]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_single,
    y,
    test_size=0.2,
    random_state=42
)
single_model = LinearRegression()
single_model.fit(X_train_s, y_train_s)
single_predictions = single_model.predict(X_test_s)
single_rmse = mean_squared_error(
    y_test_s,
    single_predictions
) ** 0.5
single_r2 = r2_score(            
    y_test_s,
    single_predictions
)
print("\nSingle Feature Model")
print("RMSE:", single_rmse)
print("R2 Score:", single_r2)
# -----------------------------
# Multiple Feature Model
# -----------------------------
print("\nMultiple Feature Model")
print("RMSE:", rmse)
print("R2 Score:", r2)
import pickle
# Save the trained model
with open("models/salary_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("\nModel saved successfully as salary_model.pkl")
# ---------------------------------
# User Salary Prediction
# ---------------------------------
print("\n----- Salary Predictor -----")
experience = float(input("Enter Experience (Years): "))
test_score = float(input("Enter Test Score: "))
interview_score = float(input("Enter Interview Score: "))
new_data = pd.DataFrame({
    "Experience": [experience],
    "TestScore": [test_score],
    "InterviewScore": [interview_score]
})
predicted_salary = model.predict(new_data)
print(f"\nPredicted Salary: ₹{predicted_salary[0]:,.2f}")