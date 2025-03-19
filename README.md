#  Customer Churn Prediction

This project is a **Customer Churn Prediction** web application built using **Streamlit** and a trained deep learning model in **TensorFlow/Keras**. The app allows users to input customer details and predicts the likelihood of churn.


## Features

✅ User-friendly **Streamlit UI**  
✅ **Predicts churn probability** based on customer details  
✅ Utilizes a **deep learning model (TensorFlow/Keras)** for accurate predictions  
✅ **Preprocessing includes** one-hot encoding, label encoding, and feature scaling  
✅ **Interactive inputs** for real-time experimentation  


## Project Structure


 Customer-Churn-Prediction
│-- experiments.ipynb        # Jupyter Notebook for initial experiments
│-- prediction.ipynb         # Model training and evaluation notebook
│-- app.py                   # Streamlit app for prediction
│-- model.h5                 # Trained deep learning model
│-- onehot_encoder_geo.pkl   # OneHotEncoder for Geography
│-- label_encoder_gender.pkl # LabelEncoder for Gender
│-- scaler.pkl               # StandardScaler for feature normalization
│-- README.md                # Project Documentation
```


 Setup Instructions

1️⃣ Clone the Repository


git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Application

streamlit run app.py


How It Works

1️⃣ Enter customer details such as:  
   - **Age, Credit Score, Balance, Geography, Gender, Tenure, etc.**  
2️⃣ The app processes the input using **pre-trained encoders and scalers**.  
3️⃣ The model predicts **Churn Probability**.  
4️⃣ If the probability **> 0.5**, the customer is **likely to churn**; otherwise, they are **not likely to churn**.  


Technologies Used

- Python
- Streamlit (for UI)
- TensorFlow/Keras (for ML model)
- Scikit-learn (for data preprocessing)
- Pandas & NumPy (for data handling)
- Pickle (for saving encoders and scalers)

Model Performance
The model was trained on a customer churn dataset and evaluated using various performance metrics.
The training and evaluation were conducted using Jupyter Notebooks (prediction.ipynb).
The final model was saved as a TensorFlow Keras model (model.h5).


Conclusion
Customer churn is a crucial challenge for businesses, and predictive analytics can help reduce customer attrition by identifying at-risk customers.
This project provides a simple yet powerful web application that leverages deep learning to predict churn probability. By integrating advanced machine learning techniques, 
businesses can take proactive measures to retain valuable customers.



