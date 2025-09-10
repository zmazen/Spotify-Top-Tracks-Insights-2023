# Spotify 2023 Data Analysis – Track Popularity & Audio Features

## Live Streamlit App

You can view the interactive Streamlit application online [here](https://spotify-top-tracks-insights-2023-8artgipqbqdrtawxunmero.streamlit.app/)

## Project Structure
```
Spotify-Top-Tracks-Insights-2023/
│
├── screenshots                            # Folder containing visualizations and charts
├── README.md                              # Project documentation
├── Spotify Top Tracks Insights 2023.ipynb # Main analysis notebook
├── app.py                                 # Streamlit dashboard 
└── spotify-2023.csv                       # Dataset
```

## Tools
- Python
- Pandas
- NumPy
- Plotly Express
- Matplotlib
- Streamlit
- Scikit-Learn
    - RandomForestRegressor
    - LinearRegression

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Project Overview
This project analyzes **Spotify 2023 streaming data** to explore trends in track popularity, artist performance, and the influence of audio features (Danceability, Valence, Energy, Acousticness) on streams. The goal is to uncover insights into what drives listener engagement and evaluate how well machine learning models can predict track streams based solely on audio characteristics.

## Key Objectives
- Explore trends in **track streams, artist performance, and track age**.  
- Analyze the **relationship between audio features and streaming popularity**.  
- Evaluate **predictive modeling** using Random Forest and Linear Regression to forecast Spotify streams.  
- Summarize key insights for **music producers, playlist curators, and data enthusiasts**.

## Exploratory Data Analysis (EDA)

### Streams vs Audio Features
We analyzed the relationship between track streams and four key audio features.

#### 1. Streams vs Danceability %
- Tracks with **moderate to high danceability** tend to accumulate **higher streams**.  
- Extremely low danceability tracks are less represented in high-stream tracks.  
- Peaks indicate **common listener preferences** for rhythm and groove.  

#### 2. Streams vs Energy %
- Higher-energy tracks generally attract **more streams**, while mid-energy tracks also perform well.  
- Low-energy tracks tend to appeal to a **niche audience**.  
- Histogram peaks identify **preferred energy ranges**.

#### 3. Streams vs Valence %
- Tracks with higher valence (positive, happy mood) often get **more streams**.  
- Mid-range valence tracks also perform well.  
- Low-valence tracks are less popular, appealing to a **smaller audience**.

#### 4. Streams vs Acousticness %
- Tracks with **low to moderate acousticness** get the **most streams**, indicating listener preference for electronic or produced music.  
- High acousticness tracks tend to attract fewer streams, suggesting a **smaller, niche audience**.

### Correlation Analysis
- Weak correlations exist between streams and audio features:  
  - Danceability %: slightly negative  
  - Valence %: minimal positive  
  - Energy % and Acousticness %: negligible to negative  
- Stronger correlations observed between features themselves, e.g., **Energy vs Acousticness** (≈-0.58).  
- Insights indicate that **streams are driven by a combination of factors**, not audio features alone.

## Machine Learning Modeling

### Objective
Predict Spotify streams based on audio features using **Random Forest Regressor** and **Linear Regression**.

### Data Preparation
- Features: `Danceability %`, `Valence %`, `Energy %`, `Acousticness %`  
- Target: `Streams`  
- Log-transformation applied to `Streams` (`np.log1p`) to reduce skewness  
- Train-test split: 80% train, 20% test  
- Feature scaling with `StandardScaler`

### Random Forest Regressor
- **n_estimators=100**, **max_depth=None**, **random_state=42**  
- Captures **non-linear relationships** between features and streams  
- **Performance:**  
  - R²: -0.152  
  - RMSE: 1.088  
- Feature importance:  
  - **Valence % > Danceability % > Energy % > Acousticness %**  
- **Insight:** Audio features alone are poor predictors of streaming success.

### Linear Regression
- Assumes **linear relationships** between features and streams  
- **Performance:**  
  - R²: 0.016  
  - RMSE: 1.006  
- Coefficients:  
  - Danceability %: negative effect  
  - Valence %: small positive effect  
  - Energy % & Acousticness %: minor negative contributions  
- **Insight:** Linear assumptions fail to capture stream dynamics; predictions are weak.

### Actual vs Predicted Visualizations
- Scatter plots for both models show **wide dispersion**, confirming low predictive power.  
- High-stream tracks (viral hits) are notably underestimated.  

## Key Findings
1. **Track Age & Streaming**: Most streams are from **recent releases**, older tracks are exceptions.  
2. **Solo vs Collaboration**: Collaborations often achieve higher streams due to multiple fanbases.  
3. **Top Artists & Tracks**: Popular artists dominate streams; virality and playlists influence success.  
4. **Audio Features**: Danceability, Valence, Energy, and Acousticness have **limited correlation** with streams.  
5. **Machine Learning Models**: Both Random Forest and Linear Regression show **very low predictive power**.  
6. **External Factors**: Marketing, social media trends, playlist placements, and seasonality are likely the **main drivers of streaming success**.

## Visualizations Included
The project includes the following visualizations to illustrate trends in Spotify 2023 data:

- **Track Popularity & Feature Analysis**
  - Top 10 Tracks by Acousticness, Danceability, Energy, Valence  
  - Streams distribution by **Track Age**  
  - Streams by **Solo vs Collaboration**  
  - Top Artists by cumulative streams  
  - Top Tracks by streams  

- **Audio Feature Relationships**
  - Histograms: Streams vs Danceability, Energy, Valence, Acousticness  
  - Correlation Heatmap: Streams & Audio Features  

- **Machine Learning Visualizations**
  - Random Forest: Actual vs Predicted Streams  
  - Random Forest: Feature Importance Bar Chart  
  - Linear Regression: Actual vs Predicted Streams  
  - Linear Regression: Feature Coefficients Bar Chart

## Conclusion
- Spotify stream success is **multi-faceted**; audio features alone do **not predict popularity**.  
- Combining **audio analysis with marketing, social media engagement, and cultural trends** is essential for accurate predictions.  
- This project demonstrates the **limitations of predictive modeling** in complex, real-world digital music ecosystems.
