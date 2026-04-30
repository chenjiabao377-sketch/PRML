# ============================================================
# LSTM Multivariate Air Quality (PM2.5) Forecasting
# ============================================================
# Requirements:
#   pip install pandas numpy matplotlib scikit-learn tensorflow
# ============================================================

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibility ──────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 0. PATHS  ← edit here if needed
# ============================================================
TRAIN_PATH = r"D:\学习资料\大三下\prml\archive\LSTM-Multivariate_pollution.csv"
TEST_PATH  = r"D:\学习资料\大三下\prml\archive\pollution_test_data1.csv"
OUT_DIR    = r"D:\学习资料\大三下\prml\results"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD & PARSE
# ============================================================
def load_and_parse(path):
    df = pd.read_csv(path)
    print(f"\n[INFO] Loaded '{path}'  shape={df.shape}")
    print(df.head(3))

    # Build datetime index from separate columns when present
    if {'year','month','day','hour'}.issubset(df.columns):
        df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
        df.set_index('datetime', inplace=True)
        df.drop(columns=['No','year','month','day','hour'], errors='ignore', inplace=True)
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
        df.set_index('datetime', inplace=True)
        df.drop(columns=['date'], errors='ignore', inplace=True)

    # Rename common aliases
    df.rename(columns={
        'pm2.5':'pm25','PM2.5':'pm25','pollution':'pm25',
        'DEWP':'dewp','TEMP':'temp','PRES':'pres',
        'cbwd':'cbwd','Iws':'iws','Is':'snow','Ir':'rain'
    }, inplace=True)

    # One-hot encode wind direction
    if 'cbwd' in df.columns:
        dummies = pd.get_dummies(df['cbwd'], prefix='wd')
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=['cbwd'], inplace=True)

    # Drop any remaining non-numeric columns
    df = df.select_dtypes(include=[np.number])

    # Drop rows where target is NaN (first row in original dataset)
    df.dropna(subset=['pm25'], inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    return df

train_df = load_and_parse(TRAIN_PATH)
test_df  = load_and_parse(TEST_PATH)

# Align columns between train and test
common_cols = [c for c in train_df.columns if c in test_df.columns]
train_df = train_df[common_cols]
test_df  = test_df[common_cols]

print(f"\n[INFO] Features used ({len(common_cols)}): {common_cols}")
TARGET = 'pm25'

# ============================================================
# 2. SCALE
# ============================================================
feature_cols = common_cols          # all numeric cols including pm25
n_features   = len(feature_cols)
target_idx   = feature_cols.index(TARGET)

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df[feature_cols].values)
test_scaled  = scaler.transform(test_df[feature_cols].values)

# ============================================================
# 3. SLIDING-WINDOW SEQUENCES
# ============================================================
N_STEPS = 24   # look-back window (hours)

def make_sequences(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i, :])   # all features
        y.append(data[i, target_idx])       # pm25 only
    return np.array(X), np.array(y)

X_train, y_train = make_sequences(train_scaled, N_STEPS)
X_test,  y_test  = make_sequences(test_scaled,  N_STEPS)

print(f"\n[INFO] X_train={X_train.shape}  y_train={y_train.shape}")
print(f"[INFO] X_test ={X_test.shape}   y_test ={y_test.shape}")

# ============================================================
# 4. BUILD LSTM MODEL
# ============================================================
def build_model(n_steps, n_features):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(n_steps, n_features)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse', metrics=['mae'])
    model.summary()
    return model

model = build_model(N_STEPS, n_features)

# ============================================================
# 5. TRAIN
# ============================================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 6. PREDICT & INVERSE-TRANSFORM
# ============================================================
def inverse_pm25(scaled_values, scaler, n_features, target_idx):
    """Inverse transform only the pm25 column."""
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, target_idx] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]

y_pred_scaled = model.predict(X_test).flatten()
y_pred = inverse_pm25(y_pred_scaled, scaler, n_features, target_idx)
y_true = inverse_pm25(y_test,        scaler, n_features, target_idx)

# Clamp negatives
y_pred = np.clip(y_pred, 0, None)

# ============================================================
# 7. METRICS
# ============================================================
rmse = math.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

print("\n" + "="*50)
print("          EVALUATION METRICS (Test Set)")
print("="*50)
print(f"  RMSE : {rmse:.4f}  μg/m³")
print(f"  MAE  : {mae:.4f}  μg/m³")
print(f"  R²   : {r2:.4f}")
print(f"  MAPE : {mape:.2f} %")
print("="*50)

# Save metrics to txt
metrics_path = os.path.join(OUT_DIR, "metrics.txt")
with open(metrics_path, 'w') as f:
    f.write(f"RMSE : {rmse:.4f}\nMAE  : {mae:.4f}\nR2   : {r2:.4f}\nMAPE : {mape:.2f}%\n")
print(f"[INFO] Metrics saved → {metrics_path}")

# ============================================================
# 8. PLOTS
# ============================================================

# ── 8a. Training loss curve ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history['loss'],     label='Train Loss (MSE)', color='steelblue')
ax.plot(history.history['val_loss'], label='Val Loss (MSE)',   color='orange')
ax.set_title('LSTM Training & Validation Loss', fontsize=14)
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_loss_curve.png"), dpi=150)
plt.show()
print("[INFO] fig1_loss_curve.png saved")

# ── 8b. Predicted vs Actual (full test) ─────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(y_true, label='Actual PM2.5',    color='steelblue', linewidth=0.8)
ax.plot(y_pred, label='Predicted PM2.5', color='orange',    linewidth=0.8, alpha=0.85)
ax.set_title('PM2.5 Forecast vs Actual — Full Test Set', fontsize=14)
ax.set_xlabel('Time Step (hours)'); ax.set_ylabel('PM2.5 (μg/m³)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_prediction_full.png"), dpi=150)
plt.show()
print("[INFO] fig2_prediction_full.png saved")

# ── 8c. Zoom: first 500 hours ────────────────────────────────
N_ZOOM = min(500, len(y_true))
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(y_true[:N_ZOOM], label='Actual PM2.5',    color='steelblue', linewidth=1)
ax.plot(y_pred[:N_ZOOM], label='Predicted PM2.5', color='orange',    linewidth=1, alpha=0.85)
ax.set_title(f'PM2.5 Forecast vs Actual — First {N_ZOOM} Hours', fontsize=14)
ax.set_xlabel('Time Step (hours)'); ax.set_ylabel('PM2.5 (μg/m³)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_prediction_zoom.png"), dpi=150)
plt.show()
print("[INFO] fig3_prediction_zoom.png saved")

# ── 8d. Scatter: actual vs predicted ─────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_true, y_pred, alpha=0.3, s=5, color='steelblue')
lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
ax.set_xlabel('Actual PM2.5 (μg/m³)'); ax.set_ylabel('Predicted PM2.5 (μg/m³)')
ax.set_title(f'Actual vs Predicted  (R²={r2:.4f})', fontsize=13)
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_scatter.png"), dpi=150)
plt.show()
print("[INFO] fig4_scatter.png saved")

# ── 8e. Residuals distribution ───────────────────────────────
residuals = y_true - y_pred
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(residuals, bins=80, color='steelblue', edgecolor='white', linewidth=0.3)
ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
ax.set_title('Residuals Distribution (Actual − Predicted)', fontsize=13)
ax.set_xlabel('Residual (μg/m³)'); ax.set_ylabel('Count')
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig5_residuals.png"), dpi=150)
plt.show()
print("[INFO] fig5_residuals.png saved")

# ── 8f. Feature correlation heatmap (train set) ──────────────
try:
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = train_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, linewidths=0.5)
    ax.set_title('Feature Correlation Matrix', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig6_correlation.png"), dpi=150)
    plt.show()
    print("[INFO] fig6_correlation.png saved")
except ImportError:
    print("[WARN] seaborn not installed — skipping correlation heatmap (pip install seaborn)")

# ============================================================
# 9. SAVE PREDICTIONS CSV
# ============================================================
pred_df = pd.DataFrame({'actual': y_true, 'predicted': y_pred,
                        'residual': residuals})
csv_path = os.path.join(OUT_DIR, "predictions.csv")
pred_df.to_csv(csv_path, index=False)
print(f"[INFO] Predictions saved → {csv_path}")

# ============================================================
# 10. SUMMARY
# ============================================================
print(f"""
╔══════════════════════════════════════════╗
║         DONE — all outputs in:           ║
║  {OUT_DIR}
╠══════════════════════════════════════════╣
║  fig1_loss_curve.png                     ║
║  fig2_prediction_full.png                ║
║  fig3_prediction_zoom.png                ║
║  fig4_scatter.png                        ║
║  fig5_residuals.png                      ║
║  fig6_correlation.png  (needs seaborn)   ║
║  metrics.txt                             ║
║  predictions.csv                         ║
╚══════════════════════════════════════════╝
Metrics → RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  MAPE={mape:.2f}%
""")