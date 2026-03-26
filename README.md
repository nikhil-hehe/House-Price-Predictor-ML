# 🏠 House Price Predictor

A comprehensive machine learning project that predicts median house values in California based on 1990 Census data. This project features a full data science pipeline—from Exploratory Data Analysis (EDA) to model tuning—and a user-friendly GUI for real-time inference.

**Dataset Source:** [Kaggle - California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

---

## 🚀 Tech Stack

- **Language:** Python 3.10+
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn (Pipeline, GridSearchCV, HistGradientBoostingRegressor)
- **GUI Framework:** Gradio
- **Model Persistence:** Joblib

---

## 📊 Project Overview

The model is trained on the California Housing dataset, focusing on the following key features:

- **Location:** Longitude and Latitude coordinates  
- **Housing Characteristics:** Median age, total rooms, and total bedrooms  
- **Demographics:** Population and total households in a block  
- **Economy:** **Median Income** (identified as the strongest predictor of house value)  
- **Proximity:** Categorical distance to the ocean (e.g., INLAND, NEAR BAY)

The integrated pipeline automatically handles:

- Missing value imputation  
- Feature scaling  
- One-hot encoding for categorical variables  

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/nikhil-hehe/house-price-predictor-ml.git
cd house-price-predictor
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Mac/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install pandas numpy scikit-learn gradio joblib
```

### 4. Prepare the Model
Ensure house_model.pkl is in the root directory. If the file is missing:
- Open 15_4_house_price_prediction.ipynb.
- Run all cells to retrain the model.
- The notebook will automatically export the best_estimator_ via joblib.

---

### 🖥️ Usage

Running the GUI
Launch the interactive web interface by running:

```bash
python gui.py
```

After running, open the local URL provided in your terminal (usually http://127.0.0.1:7860). In the interface, you can:
- Adjust sliders for neighborhood statistics.
- Select the Ocean Proximity from the dropdown.
- Click Submit to receive an instant price estimation.

---

### 📂 File Structure
# File	Description
- **housing.csv** -	The raw California housing dataset.
- **15_4_house_price_prediction.ipynb** -	The primary notebook for EDA, cleaning, and training.
- **house_model.pkl** -	The serialized Scikit-Learn pipeline (Best Estimator).
- **gui.py** - Python script for the Gradio-based user interface.
- **README.md**	- Project documentation.
