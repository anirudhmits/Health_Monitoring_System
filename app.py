import tkinter as tk
from tkinter import messagebox
import pickle
import pandas as pd

# Load the trained model from the .pkl file
with open('logistic_regression_model.pkl', 'rb') as f:
    logistic_classifier = pickle.load(f)

# Load the dataset used for training
data = pd.read_csv("D:\\heart disease\\dataset_heart.csv")

# Get column names after one-hot encoding
columns = data.columns.tolist()

def predict_input():
    try:
        # Prompt user for input
        age = int(age_entry.get())
        sex = int(sex_entry.get())
        chestpaintype = int( chestpaintype_entry.get())
        restingbloodpressure = int(restingbloodpressure_entry.get())
        serumcholestoral = int(serumcholestoral_entry.get())
        fastingbloodsugar = int(fastingbloodsugar_entry.get())
        restingelectrocardiographicresults = int(restingelectrocardiographicresults_entry.get())
        maxheartrate = int(maxheartrate_entry.get())
        exerciseinducedangina = int(exerciseinducedangina_entry.get())
        oldpeak = float(oldpeak_entry.get())
        STsegment = int(STsegment_entry.get())
        majorvessels = int(majorvessels_entry.get())
        thal = int(thal_entry.get())

        # Combine all features into a single feature vector
        input_features = [age,sex, chestpaintype, restingbloodpressure, serumcholestoral, fastingbloodsugar, restingelectrocardiographicresults, maxheartrate,
                          exerciseinducedangina, oldpeak, STsegment, majorvessels, thal] 

        # Make prediction using the model
        prediction = logistic_classifier.predict([input_features])

        # Display prediction result using a message box
        if prediction == 2:
            messagebox.showinfo("Prediction Result", "High risk of heart disease")
        else:
            messagebox.showinfo("Prediction Result", "Low risk of heart disease")
    except ValueError as e:
        messagebox.showerror("Error", str(e))

# Create GUI window
window = tk.Tk()
window.title("Heart Disease Prediction")

# Labels and Entry fields for user inputs
tk.Label(window, text="age:").grid(row=0, column=0)
age_entry = tk.Entry(window)
age_entry.grid(row=0, column=1)

tk.Label(window, text="sex (0 for female, 1 for male):").grid(row=1, column=0)
sex_entry = tk.Entry(window)
sex_entry.grid(row=1, column=1)

tk.Label(window, text="chestpaintype:").grid(row=2, column=0)
chestpaintype_entry = tk.Entry(window)
chestpaintype_entry.grid(row=2, column=1)

tk.Label(window, text="restingbloodpressure:").grid(row=3, column=0)
restingbloodpressure_entry = tk.Entry(window)
restingbloodpressure_entry.grid(row=3, column=1)

tk.Label(window, text="serumcholestoral:").grid(row=4, column=0)
serumcholestoral_entry = tk.Entry(window)
serumcholestoral_entry.grid(row=4, column=1)

tk.Label(window, text="fastingbloodsugar:").grid(row=5, column=0)
fastingbloodsugar_entry = tk.Entry(window)
fastingbloodsugar_entry.grid(row=5, column=1)

tk.Label(window, text="restingelectrocardiographicresults:").grid(row=6, column=0)
restingelectrocardiographicresults_entry = tk.Entry(window)
restingelectrocardiographicresults_entry.grid(row=6, column=1)

tk.Label(window, text="maxheartrate:").grid(row=7, column=0)
maxheartrate_entry = tk.Entry(window)
maxheartrate_entry.grid(row=7, column=1)

tk.Label(window, text="exerciseinducedangina:").grid(row=8, column=0)
exerciseinducedangina_entry = tk.Entry(window)
exerciseinducedangina_entry.grid(row=8, column=1)

tk.Label(window, text="oldpeak:").grid(row=9, column=0)
oldpeak_entry= tk.Entry(window)
oldpeak_entry.grid(row=9, column=1)

tk.Label(window, text="STsegment:").grid(row=10, column=0)
STsegment_entry = tk.Entry(window)
STsegment_entry.grid(row=10, column=1)

tk.Label(window, text="majorvessels:").grid(row=11, column=0)
majorvessels_entry = tk.Entry(window)
majorvessels_entry.grid(row=11, column=1)

tk.Label(window, text="thal:").grid(row=12, column=0)
thal_entry = tk.Entry(window)
thal_entry.grid(row=12, column=1)

# Button for making predictions
predict_button = tk.Button(window, text="Predict", command=predict_input)
predict_button.grid(row=13, column=0, columnspan=2)

# Run the GUI event loop
window.mainloop()
