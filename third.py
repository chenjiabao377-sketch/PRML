# ============================================================
# LSTM 多变量空气质量 (PM2.5) 预测 - 修复版
# ============================================================

import os

# 屏蔽 TensorFlow 的 oneDNN 优化警告和系统日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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

# 忽略警告
warnings.filterwarnings('ignore')

# ── 随机种子设置（确保结果可复现） ──────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 0. 路径设置 (根据你的环境确认)
# ============================================================
TRAIN_PATH = r"D:\学习资料\大三下\prml\archive\LSTM-Multivariate_pollution.csv"
TEST_PATH = r"D:\学习资料\大三下\prml\archive\pollution_test_data1.csv"
OUT_DIR = r"D:\学习资料\大三下\prml\results"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. 数据加载与解析
# ============================================================
def load_and_parse(path):
    df = pd.read_csv(path)
    print(f"\n[INFO] 已加载数据 '{path}'  形状={df.shape}")

    # 建立日期索引
    if {'year', 'month', 'day', 'hour'}.issubset(df.columns):
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df.set_index('datetime', inplace=True)
        df.drop(columns=['No', 'year', 'month', 'day', 'hour'], errors='ignore', inplace=True)
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
        df.set_index('datetime', inplace=True)
        df.drop(columns=['date'], errors='ignore', inplace=True)

    # 重命名列名以统一
    df.rename(columns={
        'pm2.5': 'pm25', 'PM2.5': 'pm25', 'pollution': 'pm25',
        'DEWP': 'dewp', 'TEMP': 'temp', 'PRES': 'pres',
        'cbwd': 'cbwd', 'Iws': 'iws', 'Is': 'snow', 'Ir': 'rain'
    }, inplace=True)

    # 风向特征独热编码 (One-hot encoding)
    if 'cbwd' in df.columns:
        dummies = pd.get_dummies(df['cbwd'], prefix='wd')
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=['cbwd'], inplace=True)

    # 仅保留数值型列
    df = df.select_dtypes(include=[np.number])

    # 关键修复：处理缺失值
    # 新版 Pandas 不再支持 fillna(method='ffill')，直接使用 ffill() 和 bfill()
    df.dropna(subset=['pm25'], inplace=True)
    df.ffill(inplace=True)  # 前向填充
    df.bfill(inplace=True)  # 后向填充

    return df


# 加载数据
train_df = load_and_parse(TRAIN_PATH)
test_df = load_and_parse(TEST_PATH)

# 对齐训练集和测试集的特征列
common_cols = [c for c in train_df.columns if c in test_df.columns]
train_df = train_df[common_cols]
test_df = test_df[common_cols]

print(f"\n[INFO] 使用特征数量 ({len(common_cols)}): {common_cols}")
TARGET = 'pm25'

# ============================================================
# 2. 归一化
# ============================================================
feature_cols = common_cols
n_features = len(feature_cols)
target_idx = feature_cols.index(TARGET)

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df[feature_cols].values)
test_scaled = scaler.transform(test_df[feature_cols].values)

# ============================================================
# 3. 创建滑动窗口序列
# ============================================================
N_STEPS = 24  # 回溯过去 24 小时的数据


def make_sequences(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i, :])  # 包含所有特征
        y.append(data[i, target_idx])  # 目标值为 pm2.5
    return np.array(X), np.array(y)


X_train, y_train = make_sequences(train_scaled, N_STEPS)
X_test, y_test = make_sequences(test_scaled, N_STEPS)

print(f"\n[INFO] 训练集形状: X={X_train.shape}, y={y_train.shape}")
print(f"[INFO] 测试集形状: X={X_test.shape},  y={y_test.shape}")


# ============================================================
# 4. 构建 LSTM 模型
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
# 5. 训练
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
# 6. 预测与逆缩放
# ============================================================
def inverse_pm25(scaled_values, scaler, n_features, target_idx):
    """仅对 PM2.5 列进行逆归一化"""
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, target_idx] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]


y_pred_scaled = model.predict(X_test).flatten()
y_pred = inverse_pm25(y_pred_scaled, scaler, n_features, target_idx)
y_true = inverse_pm25(y_test, scaler, n_features, target_idx)

# 强制将负值裁剪为 0
y_pred = np.clip(y_pred, 0, None)

# ============================================================
# 7. 评估指标
# ============================================================
rmse = math.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

print("\n" + "=" * 50)
print("          评估结果 (测试集)")
print("=" * 50)
print(f"  RMSE : {rmse:.4f}  μg/m³")
print(f"  MAE  : {mae:.4f}  μg/m³")
print(f"  R²   : {r2:.4f}")
print(f"  MAPE : {mape:.2f} %")
print("=" * 50)

# 保存指标
metrics_path = os.path.join(OUT_DIR, "metrics.txt")
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(f"RMSE : {rmse:.4f}\nMAE  : {mae:.4f}\nR2   : {r2:.4f}\nMAPE : {mape:.2f}%\n")

# ============================================================
# 8. 绘图
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False

# 图 1: 训练损失
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='训练损失 (MSE)')
plt.plot(history.history['val_loss'], label='验证损失 (MSE)')
plt.title('模型损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUT_DIR, "fig1_loss_curve.png"))
plt.show()

# 图 2: 全量测试集预测结果对比
plt.figure(figsize=(15, 5))
plt.plot(y_true, label='真实值', color='steelblue', linewidth=0.8)
plt.plot(y_pred, label='预测值', color='orange', linewidth=0.8, alpha=0.8)
plt.title('PM2.5 预测 vs 真实值 (全量测试集)')
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "fig2_prediction_full.png"))
plt.show()

# ============================================================
# 9. 保存结果
# ============================================================
pred_df = pd.DataFrame({'actual': y_true, 'predicted': y_pred})
csv_path = os.path.join(OUT_DIR, "predictions.csv")
pred_df.to_csv(csv_path, index=False)
print(f"\n[DONE] 结果已保存在: {OUT_DIR}")
