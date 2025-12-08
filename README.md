🔥 Estimating Fire Radiative Power (FRP) in Forest Fires Using Satellite Data
A Machine Learning Project using Linear Regression & Satellite-Based Remote Sensing Data

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?style=flat-square&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)


🚀 Project Overview
Forest fires cause devastating ecological, economic, and environmental damage.
Fire Radiative Power (FRP) is a crucial indicator of fire intensity, spread, and total emissions.
This project uses MODIS satellite data (2000–2021) and applies Linear Regression to estimate FRP quickly and effectively.

The model helps in:

🌲 Wildfire monitoring

🚒 Disaster management

🌍 Emission assessment

🔬 Environmental protection
This project uses Satellite-based Remote Sensing Data and applies Linear Regression & Logistic Regression to estimate FRP for faster emergency response, ecological monitoring, and climate modelling.

🎯 Problem Statement

“Predict Fire Radiative Power (FRP) accurately using satellite features like brightness, temperature, geographic location, and acquisition time.”

📂 Dataset
The dataset consists of forest fire data from **Turkey (2000-2021)** detected by the MODIS instrument.
* **Source:** MODIS Collection 61
* **Rows:** ~211,000+ data points
* **Key Features:**
    * `latitude`, `longitude`: Location coordinates.
    * `brightness`: Brightness temperature of the fire pixel (Kelvin).
    * `scan`, `track`: Pixel resolution details.
    * `acq_date`, `acq_time`: Date and time of acquisition.
    * `confidence`: Quality flag of the detection (0-100).
    * `bright_t31`: Channel 31 brightness temperature.
    * `daynight`: Day (D) or Night (N) detection.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Linear Regression, Logistic Regression)

## 📊 Methodology

### 1. Data Preprocessing
* Converted `acq_date` to datetime objects to extract `Month` and `Year`.
* Encoded categorical data: `DayNight` mapped to binary (1/0).
* Handled missing values and dropped nulls for clean analysis.
* Feature Selection: Focused on `brightness`, `bright_t31`, `confidence`, `scan`, `track`, and temporal features.

### 2. Exploratory Data Analysis (EDA)
We analyzed the distribution of FRP and its correlation with other variables.
* **Correlation:** A strong correlation was observed between `brightness` and `FRP`.
* **Seasonality:** Analyzed fire intensity trends across months and years.

### 3. Machine Learning Models
We implemented two approaches:

#### **A. Regression (Predicting Exact FRP)**
Used **Linear Regression** to predict the specific numeric value of Fire Radiative Power.
* **MAE:** 14.45
* **RMSE:** 28.44
* **R² Score:** 0.66 (Explains 66% of the variance in fire intensity)

#### **B. Classification (Predicting Intensity Levels)**
Grouped FRP into three categories: **Low (<15), Medium (15-50), High (>50)** and used **Logistic Regression**.
* **Accuracy:** **92%**
* **Precision/Recall:** High precision (0.94) for "Low" intensity fires.

## 📉 Results Visualized

*(Note: These are placeholders for your actual plots. Add screenshots of your notebook output here)*

| Actual vs Predicted | Confusion Matrix |
| :---: | :---: |
| ![Actual vs Pred](path_to_your_plot_image.png) | ![Confusion Matrix](path_to_your_cm_image.png) |

## 🚀 How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run the Notebook: Open`FRP_Estimation.ipynb`(or your script name) in Jupyter Notebook or Google Colab.

🔮 Future Scope

Deep Learning: Implementing Neural Networks (ANN/LSTM) to capture non-linear complex patterns.

Real-time Forecasting: Integrating live API feeds from NASA FIRMS.

Spatial Analysis: Using Geospatial libraries (GeoPandas) to map fire hotspots visually on a map.

📝 License

This project is open-source and available for educational purposes.

🤝 Contributions

Contributions, issues, and feature requests are welcome!
Feel free to fork this repository and submit a pull request.

🧑‍💻 Author

Arijit Mandal

⭐ If you like this project, don’t forget to star ⭐ the repository!
