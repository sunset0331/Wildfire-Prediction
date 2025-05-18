
# Wildfire Prediction using PyTorch Lightning

This project builds a binary classification model to predict the likelihood of a wildfire based on satellite observation data from the Central African Republic. It leverages PyTorch Lightning for clean, modular training code and provides insightful evaluation visualizations.

## 🔍 Overview

The model classifies satellite fire events into two categories:
- **Fire (High Confidence ≥ 80)**
- **No Fire (Low Confidence < 80)**

### Data Source
MODIS satellite fire detection dataset (`modis_2023_Central_African_Republic.csv`).

## 🛠 Features Used
- `brightness`, `scan`, `track`, `bright_t31`, `frp`, `fire_type`, `daynight`

## 📊 Target Variable
- `fire_class`: 1 (Fire), 0 (No Fire)

## 📦 Libraries Used
- `pandas`, `numpy`, `torch`, `pytorch_lightning`
- `sklearn`, `matplotlib`, `seaborn`

## 🧪 Model Architecture

A simple feedforward neural network:
```text
Input Layer -> Linear(64) -> ReLU -> Dropout(0.2)
-> Linear(32) -> ReLU -> Linear(2)
```
Uses `CrossEntropyLoss` and `Adam` optimizer.

## 🔁 Workflow

1. **Data Preprocessing**:
   - Datetime conversion from `acq_date` and `acq_time`
   - Label creation from `confidence`
   - Feature normalization via `StandardScaler`

2. **Dataset Preparation**:
   - `WildfireDataset` built using PyTorch `Dataset`
   - `DataLoader` with batch size 64

3. **Model Training**:
   - 20 epochs with validation tracking
   - Training and validation loss/accuracy logged via PyTorch Lightning

4. **Evaluation Metrics**:
   - Confusion Matrix
   - Fire confidence probability distribution
   - Calibration curve for model reliability

## 📈 Evaluation Results

- **Confusion Matrix**: Indicates classification performance
- **Probability Histogram**: Shows how confident the model is about predictions
- **Calibration Plot**: Visualizes how well predicted probabilities match actual outcomes

## 🧠 Usage

To run training and evaluation:
```python
# Run model training
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Evaluate on validation set
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        logits = model(x)
        ...
```

## 📌 Improvements & Future Work

- Add temporal features (e.g., time of day, seasonality)
- Handle class imbalance with weighted loss or resampling
- Experiment with more complex architectures
- Deploy as a web service for real-time predictions

## 📄 License

This project is for academic and research purposes. Feel free to extend or modify.

---

**Author:** Utkarsh Gaur  
**Institution:** IIIT Nagpur
