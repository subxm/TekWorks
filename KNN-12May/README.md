# KNN - K-Nearest Neighbors

K-Nearest Neighbors implementations using the Iris dataset.

## Contents

- `streamlit.py` - Streamlit web application for KNN regression on Iris dataset
- `knn-classification.ipynb` - KNN classification tutorial and examples
- `knn-regressor.ipynb` - KNN regression tutorial and examples
- `new.ipynb` - Additional KNN experiments

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Web App

Launch the Streamlit application:

```bash
streamlit run streamlit.py
```

Open your browser to `http://localhost:8501`

## Requirements

- Python 3.7+
- streamlit
- pandas
- scikit-learn

## Features

The Streamlit app allows you to:

- Adjust iris flower features using interactive sliders
- Get KNN regression predictions for the target variable
- Explore different numbers of neighbors using the Iris dataset
