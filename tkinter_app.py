import joblib
import numpy as np
from tkinter import *
from tkinter import ttk, messagebox

# Load components
model = joblib.load('artifacts/RFModel.joblib')
le_Weather = joblib.load('artifacts/LE_Weather.joblib')
le_Day = joblib.load('artifacts/LE_Day.joblib')
le_ProductName = joblib.load('artifacts/LE_ProductName.joblib')

amt_scaler = joblib.load('artifacts/Amt_scaler.joblib')

features = ['ProductName', 'Day', 'weather']

label_encoders = {'weather':le_Weather, 'Day':le_Day, 'ProductName':le_ProductName}

root = Tk()
root.title("Amt Prediction")
root.geometry("400x400")

input_vars = {}
row = 0

for feature in features:
    Label(root, text=feature + ":").grid(row=row, column=0, padx=10, pady=8, sticky=W)
    
    # Getting classes for this feature
    options = label_encoders[feature].classes_.tolist()
    var = StringVar()
    var.set(options[0]) 

    dropdown = ttk.OptionMenu(root, var, options[0], *options)
    dropdown.grid(row=row, column=1, padx=10, pady=8)
    
    input_vars[feature] = var
    row += 1


def predict_amt():
    try:
        # Collect and encode inputs
        encoded_input = []
        for feature in features:
            value = input_vars[feature].get()
            le = label_encoders[feature]
            encoded = le.transform([value])[0]
            encoded_input.append(encoded)

        input_array = np.array(encoded_input).reshape(1, -1)

        # Predict using model
        final_prediction = model.predict(input_array)
        #final_prediction = amt_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0][0]

        messagebox.showinfo("Prediction", f"Predicted Amt: ₹{final_prediction:.2f}")
    
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")

# Predict button
Button(root, text="Predict Amt", command=predict_amt, bg='green', fg='white', font=('Arial', 12, 'bold')).grid(
    row=row, column=0, columnspan=2, pady=20
)

root.mainloop()

