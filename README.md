# 🧠 Federated Learning-Based Diabetes Risk Prediction

This project implements a **privacy-aware neural network** using **Federated Learning (FL)** to predict diabetes risk from patient data. FL allows multiple clients to collaboratively train a model without sharing sensitive health information.

---

## 🔍 Overview

Healthcare data is sensitive and often distributed. This project uses **Federated Learning** to build a diabetes prediction model using locally stored data from simulated clients. Rather than collecting all data on a central server, only model weights are shared and averaged to update a global model.

---

## 📥 Input Features

The model uses 8 key medical inputs:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age

Output: A probability score indicating **diabetes risk** (0 or 1).

---

## 🧪 Dataset

- **Pima Indians Diabetes Dataset**  
- Source: [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- Format: CSV with 768 samples and 9 columns (8 features + 1 label)

---

## ⚙️ How It Works

1. **Data Preprocessing**  
   Data is normalized using `StandardScaler`.

2. **Client Simulation**  
   Training data is split into 3 local datasets, simulating 3 different clients.

3. **Local Training**  
   Each client trains a model locally for 5 epochs.

4. **Model Aggregation**  
   The weights are averaged and used to update the global model.

5. **Federated Rounds**  
   The above process repeats for 20 rounds to improve accuracy.

6. **Evaluation**  
   Accuracy is compared between non-federated and federated approaches.

---

## 📊 Results Visualization

A bar graph is generated to compare:

- **Non-Federated Accuracy**
- **Federated Accuracy**

Both models are evaluated on a separate test set.

---

## 🧾 Usage

To run the project:

```bash
python diabetes_federated.py
```

You will be prompted to enter new patient data:

```bash
Pregnancies: 3
Glucose: 140
Blood Pressure: 72
...
```

Output will show:

- Sigmoid prediction value
- Rounded binary result (0 or 1)

---

## 💾 Model Saving

The final global model is saved in:

```
models/federated_diabetes_model.keras
```

You can load it using:

```python
from tensorflow.keras.models import load_model
model = load_model("models/federated_diabetes_model.keras")
```

---

## 🧰 Built With

- Python 3.x
- TensorFlow / Keras
- scikit-learn
- NumPy / Pandas
- Matplotlib

---

## 🔐 Why Federated Learning?

- Preserves data privacy
- Simulates decentralized medical environments
- Reduces data transfer and central storage risks

---

## 📬 Author

**Sameer Raj** – Developer, researcher, and AI enthusiast.

---

## 📄 License

This project is licensed under the MIT License.
