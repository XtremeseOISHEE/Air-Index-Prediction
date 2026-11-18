# AirSense Dhaka

**Hourly PM2.5 prediction and pollution pattern clustering for Dhaka, Bangladesh, using machine learning.**

This project predicts the next hour's PM2.5 level in Dhaka using four models: Linear Regression, Random Forest, a Neural Network (MLP), and an LSTM. It also uses K-means clustering and PCA to find and visualise pollution risk patterns. It was built as a Machine Learning course project at Jahangirnagar University.

---

## Why this project

Dhaka is often ranked among the most polluted cities in the world. PM2.5 (fine particles smaller than 2.5 micrometres) is the main health risk. The goals here are:

- Predict the next hour's PM2.5 from recent history and time features.
- Compare classic machine learning with neural networks on the same task.
- Group hours into pollution risk levels without using labels (unsupervised learning).

---

## Dataset

| Item | Details |
|---|---|
| Name | Dhaka Hourly Air Quality (2016-2022) |
| Source | Kaggle, uploaded by `shawkatsujon` |
| Link | https://www.kaggle.com/datasets/shawkatsujon/dhaka-bangladesh-hourly-air-quality-20162022 |
| Content | Hourly PM2.5 readings for Dhaka, one CSV file per year (2016 to 2022) |
| Period used | February 2016 to June 2022 (real readings start on 2016-02-29) |
| Pollutant | PM2.5 only (no weather or other pollutants) |

> **TODO before publishing:** open the Kaggle page and confirm (1) the original data provider and (2) the dataset licence. Then add both here.

### Raw columns

| Column | Meaning |
|---|---|
| `Date (LT)` | Date and time in local (Dhaka) time |
| `Hour` | Hour of the day (0 to 23) |
| `NowCast Conc.` | Smoothed PM2.5 value (weighted average of recent hours) |
| `Raw Conc.` | Raw PM2.5 reading in ug/m3. **Used as the prediction target.** |
| `Conc. Unit` | Unit of measurement (always ug/m3) |
| `AQI` | Air Quality Index calculated from PM2.5 |
| `AQI Category` | Text label for the AQI value (Good, Moderate, and so on) |
| `QC Name` | Quality flag. `Missing` marks hours with no reading (stored as `-999`) |

---

## Method

### 1. Merge and clean

- Merged the 7 yearly files, sorted by time, and removed duplicate timestamps.
- Replaced the `-999` placeholder with a real missing value (`NaN`).
- Built a continuous hourly timeline so skipped hours also show up as missing.
- Filled **short gaps only** (24 hours or less) with linear interpolation.
- Left longer gaps as missing instead of guessing.

**Result:** 56,233 hourly rows. About 8.9% were missing at first. About 4.4% remained missing after interpolation and were dropped before modelling (52,262 usable rows).

### 2. Features

| Feature | Description |
|---|---|
| `month`, `day_of_week`, `hour` | Time features |
| `is_weekend` | 1 for Friday and Saturday (Bangladesh weekend) |
| `pm25_lag_1h` | PM2.5 one hour earlier |
| `pm25_lag_24h` | PM2.5 at the same hour yesterday |

Target: `pm25_raw` (PM2.5 in ug/m3).

### 3. Train and test split

The split is **chronological**, not random. Random splitting would let the model see the future.

- Train: 2016 to 2021 (48,692 rows)
- Test: 2022, unseen data (3,570 rows)

### 4. Models

| Model | Setup |
|---|---|
| Linear Regression | Scaled features |
| Random Forest | 200 trees, max depth 12 |
| Neural Network (MLP) | Two hidden layers (64 and 32 neurons), ReLU, Adam, early stopping |
| LSTM | 50 units, input is the previous 24 hours of PM2.5, 20 epochs |

### 5. Unsupervised analysis

- **K-means** (4 clusters) on `pm25_raw`, `aqi`, `hour`, and `month`.
- **PCA** (2 components) on 7 features, used to plot the clusters in 2D.

---

## Results

Evaluated on the 2022 test set. Lower RMSE and MAE are better. Higher R2 is better.

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Neural Network (MLP) | 30.22 | 17.92 | 0.8239 |
| LSTM | 30.33 | 18.22 | 0.8226 |
| Random Forest | 31.21 | 18.50 | 0.8123 |
| Linear Regression | 31.59 | 19.20 | 0.8077 |

**Main finding:** all four models perform very close to each other. The lag and time features seem to carry most of the predictive power. A more complex model (LSTM) did not beat the simpler MLP.

### K-means clusters

| Cluster | Avg PM2.5 (ug/m3) | Avg AQI | Hours | Pollution level |
|---|---|---|---|---|
| 0 | 204.9 | 246.7 | 9,096 | Highest |
| 2 | 137.5 | 201.4 | 8,875 | High |
| 1 | 49.6 | 119.9 | 18,429 | Medium |
| 3 | 41.8 | 110.6 | 15,862 | Lower |

### PCA

Two components keep about 61.9% of the variance (PC1 47.6%, PC2 14.3%). In the 2D plot, the four clusters separate clearly.

### Seasonal pattern

Average PM2.5 is highest in winter (about 170 ug/m3) and lowest in the monsoon (about 74 ug/m3).

---

## Demo dashboard

`airsense_demo.html` is a standalone page. Open it in any browser. Pick a month and an hour to see the average PM2.5, the AQI category, and the risk cluster, plus a 24-hour chart.

**Important:** this dashboard shows **historical averages** for each month and hour (2016 to 2022). It does **not** run the trained models live.

---

## Limitations

- Only one pollutant (PM2.5). No weather data such as temperature, humidity, or wind.
- Data comes from a single monitoring site, so it may not represent all of Dhaka.
- The test set covers January to June 2022 only, so it does not cover a full year.
- Predictions are one hour ahead. There is no persistence baseline ("next hour = this hour") for comparison yet.
- Long data gaps (over 24 hours) were dropped, not filled.

## Future work

- Add weather features (temperature, humidity, wind speed, rainfall).
- Compare against a persistence baseline and a longer forecast horizon.
- Try multiple stations and hyperlocal prediction.
- Serve the trained model live (for example with Streamlit) with health alerts.

---

## Repository structure

```
.
├── AirSense_Dhaka.ipynb       # full pipeline: cleaning, models, clustering, PCA
├── merge_clean.py             # merge and clean the 7 yearly CSV files
├── modeling.py                # regression, MLP, K-means, PCA
├── airsense_demo.html         # standalone demo dashboard
├── data/                      # place the yearly CSV files here (see Dataset)
└── README.md
```

## How to run

1. Download the dataset from Kaggle (link above) and put the 7 CSV files in `data/`.
2. Install the libraries:
   ```bash
   pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
   ```
3. Run `merge_clean.py` to create `dhaka_aqi_master_2016_2022.csv`.
4. Open `AirSense_Dhaka.ipynb` (Google Colab works well) and run the cells in order.

Tested on Google Colab with Python 3 and TensorFlow 2.20.

## Tech stack

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, Chart.js (dashboard).

## Author

**Asma-Ul-Husna Oishee**
Department of Computer Science and Engineering, Jahangirnagar University

## Acknowledgements

Dataset: "Dhaka Hourly Air Quality (2016-2022)" by `shawkatsujon` on Kaggle. Please credit the original data provider as well (see the TODO above).
