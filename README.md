# 🏠 Premium House Price Predictor Dashboard

An interactive, dark-themed desktop application built using Python, Scikit-Learn, and Tkinter. This project demonstrates an end-to-end Machine Learning pipeline—from synthetic data generation and feature scaling to live, dynamic data visualization.

---

## 🚀 Key Features

* **Real-Time Predictions:** Seamlessly adjust property attributes (Square Footage, Bedrooms, and Neighborhood Rating) using interactive sliders to instantly see updated valuation estimates.
* **Live Matplotlib Visualizations:** Includes two side-by-side dynamic charts:
  * *Price vs. Square Footage:* Displays actual sales data alongside a real-time updating model trend line and a custom marker tracking "Your House".
  * *Actual vs. Predicted Valuation:* A diagnostic scatter plot featuring a perfect fit reference line and real-time crosshairs pinpointing where your inputs fall in the test distribution.
* **Production-Ready Pipeline:** Features robust training/testing data splits (80/20) and automated input feature normalization using Scikit-Learn's `StandardScaler`.
* **Transparent Model Metrics:** Displays live performance statistics including R² accuracy, Root Mean Squared Error (RMSE), and the raw mathematical formula weights.

---

## 🛠️ Installation & Setup

Ensure you have Python installed on your system. Then, follow these steps to run the application locally:

1. **Download the project files** (`housepredct.py` and `requirements.txt`) into the same folder.
2. Open your terminal or command prompt inside that folder and **install the required libraries**:
   ```bash
   pip install -r requirements.txt
