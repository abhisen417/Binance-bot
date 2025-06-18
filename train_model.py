# Save as train_model.py
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier

X = np.random.rand(300, 32)  # 30 closes + rsi + macd
y = np.random.randint(0, 2, size=300)

model = GradientBoostingClassifier()
model.fit(X, y)

joblib.dump(model, "train_model.py")
