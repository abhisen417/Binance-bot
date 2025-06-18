import os
import time
import hmac
import hashlib
import requests
import numpy as np
import joblib
from urllib.parse import urlencode
from flask import Flask
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
BASE_URL = os.getenv("BINANCE_API_BASE_URL", "https://testnet.binance.vision")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

model = joblib.load("models/model.pkl")

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": message})

def sign(params):
    query = urlencode(params)
    signature = hmac.new(SECRET_KEY.encode(), query.encode(), hashlib.sha256).hexdigest()
    return f"{query}&signature={signature}"

def get_symbols():
    r = requests.get(f"{BASE_URL}/api/v3/exchangeInfo")
    symbols = [s["symbol"] for s in r.json()["symbols"] if s["quoteAsset"] == "USDT"]
    return list(set(symbols))

def get_price(symbol):
    try:
        r = requests.get(f"{BASE_URL}/api/v3/ticker/price", params={"symbol": symbol})
        return float(r.json()["price"])
    except:
        return None

def get_klines(symbol):
    try:
        r = requests.get(f"{BASE_URL}/api/v3/klines", params={"symbol": symbol, "interval": "5m", "limit": 100})
        return r.json()
    except:
        return []

def calc_rsi(prices, period=14):
    deltas = np.diff(prices)
    gain = deltas[deltas > 0].sum() / period
    loss = -deltas[deltas < 0].sum() / period
    rs = gain / loss if loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calc_macd(prices):
    ema12 = np.mean(prices[-12:])
    ema26 = np.mean(prices[-26:])
    macd = ema12 - ema26
    signal = np.mean(prices[-9:])
    return macd - signal

def strategy(symbol):
    klines = get_klines(symbol)
    if len(klines) < 30:
        return False

    closes = np.array([float(k[4]) for k in klines])
    rsi = calc_rsi(closes)
    macd_signal = calc_macd(closes)
    features = np.append(closes[-30:], [rsi, macd_signal]).reshape(1, -1)
    prediction = model.predict(features)[0]

    return prediction == 1 and 40 < rsi < 70

def place_trade(symbol, amount=500):
    price = get_price(symbol)
    if not price: return {"error": "No price"}

    qty = round(amount / price, 5)
    timestamp = int(time.time() * 1000)
    params = {
        "symbol": symbol,
        "side": "BUY",
        "type": "MARKET",
        "quantity": qty,
        "timestamp": timestamp
    }
    headers = {"X-MBX-APIKEY": API_KEY}
    url = f"{BASE_URL}/api/v3/order?{sign(params)}"
    result = requests.post(url, headers=headers).json()

    trail_tp = round(price * 1.07, 2)
    trail_sl = round(price * 0.975, 2)
    send_telegram(f"📈 TRADE on {symbol}\nPrice: {price}\nQty: {qty}\nTP: {trail_tp}, SL: {trail_sl}")
    return result

def scan_and_trade():
    symbols = get_symbols()
    traded = {}
    for symbol in symbols:
        if strategy(symbol):
            result = place_trade(symbol)
            traded[symbol] = result
    return traded or {"message": "No signals"}

@app.route("/")
def home():
    return "✅ Binance Testnet Bot Running"

@app.route("/run")
def run():
    return scan_and_trade()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
