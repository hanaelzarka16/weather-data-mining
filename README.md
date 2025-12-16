# Weather Data Mining and Rain Prediction 

## Project Overview
This project explores weather data using **data mining and machine learning techniques** to discover weather patterns and predict whether it will rain.  
The workflow combines **unsupervised learning** (clustering) and **supervised learning** (classification) to both understand and predict weather behavior.

The project was developed as part of a Data Mining course and focuses on:
- Identifying meaningful weather regimes
- Engineering meteorologically relevant features
- Comparing multiple machine learning models for rain prediction

---

## Dataset
- **Source:** Kaggle – Weather Forecast Dataset  
- **Features include:** Temperature, Humidity, Wind Speed, Cloud Cover, Pressure, and derived features
- **Target variable:** `Rain` (binary: Rain / No Rain)

Two datasets are used:
- `weather_forecast_data.csv` – original dataset
- `weather_with_advanced_clusters.csv` – enriched dataset with clustering labels

---
## Project Structure

```text
weather-data-mining/
├── data/
│ └── weather_mining.csv
│
├── notebooks/
│ └── Weather_mining.ipynb
│
├── report/
│ └── DataMining_Final_Report.pdf
│
├── README.md
```
---

## Methodology

### 1. Data Preprocessing
- Inspected dataset structure and missing values
- Visualized outliers using boxplots
- Removed extreme outliers using Z-score filtering
- Standardized all numeric features using `StandardScaler`

---

### 2. Feature Engineering
New features were created to capture physical weather relationships:
- `Wind_Power`
- `Humidity_Cloud`
- `Pressure_Change`
- `Humidity_to_Pressure`
- `Cloud_to_Pressure`

All features were rescaled after engineering.

---

### 3. Unsupervised Learning (Clustering)
- **PCA** used for dimensionality reduction and visualization
- **K-Means clustering**
  - Optimal k selected using Elbow Method and Silhouette Score
  - k = 2 chosen as the best balance between compactness and separation
- **DBSCAN**
  - Tested and rejected due to lack of density-based structure
- **Advanced clustering**
  - Gaussian Mixture Models (GMM) with BIC/AIC selection
  - Hierarchical (Agglomerative) clustering for structure validation

---

### 4. Supervised Learning (Classification)

Models trained to predict **Rain vs No Rain**:
- Logistic Regression
- Decision Tree
- Random Forest
- Fine-tuned Multi-Layer Perceptron (MLP)

Evaluation metrics:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve & AUC (where applicable)

The MLP was fine-tuned using **RandomizedSearchCV** with stratified cross-validation.

---

## Results Summary
- Tree-based models (Decision Tree, Random Forest) achieved very high accuracy
- Tuned MLP achieved strong generalization with excellent ROC AUC
- Humidity-related and pressure-based features were consistently the most important
- Clustering helped reveal interpretable weather regimes

---

## Tools & Libraries
- Python, Google Colab
- pandas, numpy
- scikit-learn
- matplotlib, seaborn, plotly
- joblib

---

## How to Run
1. Clone the repository
2. Open `notebooks/Weather_mining.ipynb` in Google Colab or Jupyter
3. Upload datasets if required
4. Run cells sequentially

---

## Authors
- Hana ELzarka
- Hania Amr
- Nada Abdelsalam
- Nada Allam

---

## License
This project is for academic and educational purposes.
