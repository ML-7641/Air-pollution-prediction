# Air Quality and Pollution Assessment Project Proposal
**Group 38: Zundong Wu, Xingjian Bi, Weiyu Sun, Yishu Li**

## Introdcution
Air quality classification aims to categorize air quality into distinct levels based on pollutant concentrations, assisting the public in taking protective measures and providing data support for environmental management. The World Health Organization (WHO) estimates that air pollution causes approximately 7 million premature deaths annually, underscoring the critical need for accurate classification. Machine learning techniques, due to their ability to handle complex data, have emerged as a cornerstone in this field.

### Literature Review
Related studies demonstrate the superior performance of machine learning algorithms such as Random Forest (RF) and LightGBM in classification tasks. For instance, Sanjeev et al. (2021) achieved 100% accuracy using RF ([1]), while Li et al. (2024) reported a 97.5% accuracy with LightGBM for air quality classification in Jinan ([2]). Additionally, Houdou et al. (2024) conducted a systematic review of interpretable machine learning models in air pollution prediction, emphasizing the importance of model transparency for decision-making, covering 56 relevant studies ([4]). Han et al. (2023) surveyed general frameworks for urban air quality analysis, addressing challenges like multimodal data learning and uncertainty quantification in predictions ([5]). These studies provide a theoretical foundation for our classification task.

### Dataset Description and Dataset Link
We plan to utilize the "Air Quality and Pollution Assessment" dataset from Kaggle (https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment?resource=download ). This dataset focuses on air quality assessment across various regions, containing 5,000 samples and capturing critical environmental and demographic factors influencing pollution levels. It is well-suited for machine learning tasks, particularly air quality classification or regression analysis, enabling researchers to explore air pollution patterns and their contributing factors.

Key Features:
Temperature (°C): Average temperature of the region.
Humidity (%): Relative humidity recorded in the region.
PM2.5 Concentration (µg/m³): Fine particulate matter levels.
PM10 Concentration (µg/m³): Coarse particulate matter levels.
NO2 Concentration (ppb): Nitrogen dioxide levels.
SO2 Concentration (ppb): Sulfur dioxide levels.
CO Concentration (ppm): Carbon monoxide levels.
Proximity to Industrial Areas (km): Distance to the nearest industrial zone.
Population Density (people/km²): Number of people per square kilometer in the region.

Target Variable: Air Quality Levels
Good: Clean air with low pollution levels.
Moderate: Acceptable air quality with some pollutants present.
Poor: Noticeable pollution that may affect sensitive groups.
Hazardous: Highly polluted air posing serious health risks to the population.

## Problem Definition
### Problem
The problem we aim to address is the classification of air quality into distinct categories (Good, Moderate, Poor, Hazardous) using machine learning algorithms based on historical and real-time data. The specific task involves training a classification model, leveraging both supervised and unsupervised learning techniques, using dataset features (e.g., pollutant concentrations and meteorological conditions) to output discrete air quality categories.

### Motivation
Accurate air quality classification is of multifaceted importance:

Public Health: Timely classification enables health warnings, allowing the public to take protective actions and reduce exposure to harmful pollutants.

Environmental Management: Understanding factors affecting air quality aids in developing emission reduction strategies to improve urban air quality.

Policy Development: Data-driven classification models provide evidence for environmental policies, supporting governments in formulating air quality improvement plans.

Given the profound impact of air pollution on human health (e.g., respiratory diseases) and the environment (e.g., climate change), there is an urgent need to develop efficient and accurate air quality classification methods. Traditional monitoring systems, limited by sensor distribution, struggle to deliver real-time, localized classification results, whereas machine learning techniques can significantly enhance predictive capabilities.

## Methods

### Data Preprossing

#### Label Encoding
If the air quality categories have a natural order (i.e., Good < Moderate < Poor < Hazardous), assign integer values:
Good → 0
Moderate → 1
Poor → 2
Hazardous → 3
This method is useful when you want the model to understand increasing severity especially for tree-based models like XGBoost

#### Data Splitting
We will split the dataset into an 80% training set and a 20% test set to ensure a proper evaluation of the model’s performance. The training set will be used to train the machine learning model, allowing it to learn patterns from the data, while the test set will be kept separate to assess the model’s generalization ability on unseen data. For unsupervised learning cases, we use all dataset to fit the model and compare the cluster result with the ground truth label.

### Model
#### XGBoost
XGBoost (Extreme Gradient Boosting) is a powerful tree-based ensemble learning method that excels in structured datasets.
Implementation: xgboost.XGBClassifier()
Why it's Effective:
Handles non-linearity well.
Performs built-in feature selection, which helps in dealing with correlated variables.
Efficiently deals with missing values and imbalanced data.
Provides feature importance scores, aiding interpretability.

#### Random Forest
Random Forest is a robust ensemble learning method that constructs multiple decision trees and combines their outputs for improved classification performance.
Implementation: RandomForestClassifier()
Why it's Effective:
Reduces overfitting by averaging multiple decision trees.
Handles both linear and non-linear relationships effectively.
Works well with high-dimensional data and correlated features.
Provides feature importance scores, enhancing interpretability.

#### GMM
Gaussian Mixture Model (GMM) is a widely-used unsupervised clustering method, it assumes each cluster derived from a certain Gaussian distribution and then assigns data points to clusters based on the probability of belonging to each Gaussian component.
Implementation: sklearn.mixture.GaussianMixture, we assume the number of clusters is the same with kinds of label types in the dataset.
Why use GMM?
Compared with those supervised learning methods mentioned above, unsupervised learning lacks label information. So seemingly there is no need for us to use unsupervised learning for research. However, we think using such an unsupervised learning method may potentially help us find the fresh relationship hidden among the dataset that can not be reflected from the given labels.

### Results and Discussion
#### Metric
F1-Score (for balanced classification)
Accuracy (overall correctness)
AUC-ROC (for robustness in classification)
Project Goal
Develop a machine learning model to classify air quality levels (Good, Moderate, Poor, Hazardous) using environmental and demographic data. The model will support pollution management, public health protection, and policy decision-making, with a focus on sustainability, fairness, and transparency.
Expected Result
The model is expected to achieve an F1-score above 80%, ensuring balanced classification performance across air quality categories. It should also maintain high accuracy (>85%) and an AUC-ROC score above 0.85, indicating strong predictive capability.

## Reference
[1] M. M. Rahman, M. E. H. Nayeem, M. S. Ahmed, K. A. Tanha, M. S. A. Sakib, K. M. M. Uddin, and H. M. H. Babu, "AirNet: predictive machine learning model for air quality forecasting using web interface," Environmental Systems Research, vol. 13, no. 44, Oct. 2024. [Online]. Available: https://doi.org/10.1186/s40068-024-00378-z
[2] Q. Liu, B. Cui, and Z. Liu, "Air Quality Class Prediction Using Machine Learning Methods Based on Monitoring Data and Secondary Modeling," Atmosphere, vol. 15, no. 5, p. 553, Apr. 2024. [Online]. Available: https://doi.org/10.3390/atmos15050553
[3] A. Kumar and P. Goyal, "Forecasting of daily air quality index in Delhi," Science of the Total Environment, vol. 409, no. 24, pp. 5517–5523, Dec. 2011. [Online]. Available: https://doi.org/10.1016/j.scitotenv.2011.08.069
[4] K. P. Singh, S. Gupta, A. Kumar, and S. P. Shukla, "Linear and nonlinear modeling approaches for urban air quality prediction," Science of the Total Environment, vol. 426, pp. 244–255, May 2012. [Online]. Available: https://doi.org/10.1016/j.scitotenv.2012.03.076

