# Restaurant Recommendation: Clustering and Classification

This project groups restaurants scraped from Swiggy into recommendation clusters, then trains supervised models to predict the cluster for new restaurants. A Streamlit app serves the final model.

## Overview

The dataset had no target column, so KMeans clustering created three labeled groups (0, 1, 2). These labels were then used to train classifiers. Logistic Regression performed best.

## Features

- Cleans and preprocesses scraped Swiggy data.
- Clusters restaurants with KMeans. The Elbow and Silhouette plots support k = 3.
- Compares Logistic Regression, Random Forest, Gradient Boosting, and Decision Tree classifiers.
- Serves predictions through an interactive Streamlit web app.

## Dataset

Source: restaurant data scraped from Swiggy (fewer than 2,000 records).

| Column | Description |
| --- | --- |
| Hotel Name | Restaurant name (dropped before modeling) |
| Rating | Restaurant rating |
| Delivery Time | Delivery time in minutes |
| Food Type | Type of food served |
| Location | Restaurant location |
| Offer Above Price | Minimum order for an offer (dropped; mostly missing) |
| Offer in Percentage | Offer discount (dropped; mostly missing) |

## Preprocessing

1. Dropped `hotel_name`.
2. Cleaned `rating` and filled missing values with the median.
3. Converted `time_minutes` ranges and filled missing values with the median.
4. Label-encoded `location`.
5. Replaced `not_available` with NaN in `offer_percentage`.
6. Dropped `offer_percentage` and `offer_above` because most values were missing.
7. Frequency-encoded `food_type` to avoid adding many columns.
8. Removed 19 duplicate rows.
9. Applied scaling. Standard scaling was used, and power transformation was applied to `food_type`, `location`, and `rating` to reduce skew. This gave the best results. RobustScaler did not help.

## Results

| Metric | Logistic Regression (best) |
| --- | --- |
| Accuracy | 0.9845 |
| Precision | 0.9847 |
| Recall | 0.9845 |
| F1-Score | 0.9845 |
| ROC-AUC | 0.9997 |

Decision Tree performed close behind. Training and validation scores were similar, which suggests good generalization. With fewer than 2,000 records, the results may still be optimistic.

## Project Structure

Update this section to match your repository. For example:

| File | Purpose |
| --- | --- |
| `<notebook or script>` | Data cleaning, clustering, and model training |
| `<app file>.py` | Streamlit app |
| `<model file>` | Saved Logistic Regression model and preprocessing objects |





## Running the App

```bash
streamlit run <app file>.py
```

Then open the URL that Streamlit prints, usually http://localhost:8501.

Enter the rating, delivery time (minutes), food type, and location. Optional fields such as hotel name and offers are accepted but not used in prediction. The app returns the predicted cluster (0, 1, or 2).

## Limitations

- The dataset is small, which increases the risk of overfitting or underfitting.
- Many optional fields were marked "Not Available," so offer information was dropped.
- Cluster labels come from KMeans, not from a ground-truth label. The classifiers learn to reproduce those clusters, so high accuracy means they match the clustering, not that the clusters are correct.

## Author

Tarek Muhammad

## License

Specify a license here, or remove this section.
