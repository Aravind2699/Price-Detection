# Taxi Fare Prediction Project

This project predicts taxi fares using machine learning models trained on NYC Yellow Taxi trip data.

## Project Structure

```
ML_product/
├── data/
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data files
├── models/                 # Trained model files
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code (data loading, training, utils, etc.)
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Data

- The main dataset is NYC Yellow Taxi trip data for 2023.
- Download the data from [Google Drive](https://drive.google.com/file/d/1QSHzPGDlaqkfEtmTY48tUcFwHqt7CbeM/view?usp=sharing) and place it in `data/raw/` as `2023_Yellow_Taxi_Trip_Data.csv`.

## Setup

1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset as described above.

## Usage

- **Training the model:**
  ```bash
  python src/train.py
  ```
- **Making predictions:**
  Use the `predict_new_data` or `predict_single_sample` functions in `src/train.py`.

## Notebooks

- See `notebooks/exploration.ipynb` for data exploration and prototyping.

## Testing

- Run unit tests with:
  ```bash
  pytest tests/
  ```

## Authors
- Your Name

## License
- MIT License (or specify your license)
