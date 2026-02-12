# ==========================================
# Aviator Tracker & AI Predictor (Clean Version)
# ==========================================

import streamlit as st
import numpy as np
import plotly.express as px
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import pygame
from plyer import notification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import time
import random
import threading
import glob
from datetime import datetime

# -----------------------------
# File & Backup Setup
# -----------------------------
LOG_FILE = "aviator_history.csv"
BACKUP_FOLDER = "backups"
MAX_BACKUPS = 20
os.makedirs(BACKUP_FOLDER, exist_ok=True)

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        'timestamp','round','multiplier','bet_amount','winnings',
        'cumulative_bets','cumulative_wins','net_profit',
        '>1.5x','>3x','>5x','>10x','confidence',
        'alert_triggered','big_multiplier_spike',
        'ai_big_prob','ai_confidence','inter_round_seconds'
    ]).to_csv(LOG_FILE, index=False)

def backup_csv(df):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_file = os.path.join(BACKUP_FOLDER, f"backup_{timestamp}.csv")
    df.to_csv(backup_file, index=False)

    backups = sorted(os.listdir(BACKUP_FOLDER))
    if len(backups) > MAX_BACKUPS:
        for old in backups[:-MAX_BACKUPS]:
            os.remove(os.path.join(BACKUP_FOLDER, old))

# -----------------------------
# Streamlit Setup
# -----------------------------
st.set_page_config(page_title="Aviator Tracker", layout="wide")
st.title("Aviator Tracker & AI Predictor")

# Sidebar settings
st.sidebar.header("Settings")
refresh_rate = st.sidebar.slider("Refresh (seconds)", 5, 30, 10)
simulate_bet_min = st.sidebar.number_input("Min Bet", 1, 100, 1)
simulate_bet_max = st.sidebar.number_input("Max Bet", 1, 500, 10)

# -----------------------------
# Alert System
# -----------------------------
pygame.mixer.init()

def play_sound():
    if os.path.exists("alert.wav"):
        pygame.mixer.music.load("alert.wav")
        pygame.mixer.music.play()

def desktop_alert(msg):
    notification.notify(
        title="Aviator Alert",
        message=msg,
        timeout=5
    )

def alert_user(msg):
    threading.Thread(target=play_sound, daemon=True).start()
    threading.Thread(target=desktop_alert, args=(msg,), daemon=True).start()
    st.warning(msg)

# -----------------------------
# Web Scraping Setup
# -----------------------------
URL = "https://apkpkcas16.royalxcasino.club/"
ROUND_SELECTOR = ".round-multiplier"

options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)
driver.get(URL)

# -----------------------------
# Feature Functions
# -----------------------------
def rolling_features(df, window=5):
    df = df.copy()
    df['avg_last_5'] = df['multiplier'].rolling(window).mean()
    df['max_last_5'] = df['multiplier'].rolling(window).max()
    df['low_count_last_5'] = df['multiplier'].rolling(window).apply(lambda x: (x < 1.5).sum(), raw=True)
    df['high_count_last_5'] = df['multiplier'].rolling(window).apply(lambda x: (x > 5).sum(), raw=True)
    return df

# -----------------------------
# AI Training
# -----------------------------
def train_ai(df):
    df = rolling_features(df)
    df['big_multiplier'] = (df['multiplier'] > 5).astype(int)

    features = ['avg_last_5','max_last_5','low_count_last_5','high_count_last_5']
    df = df.dropna()

    if len(df) < 20:
        return None, None, features

    X = df[features]
    y = df['big_multiplier']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, features

# -----------------------------
# Live Loop
# -----------------------------
placeholder = st.empty()
alerted_rounds = set()

while True:
    try:
        elements = driver.find_elements(By.CSS_SELECTOR, ROUND_SELECTOR)
        multipliers = [
            float(e.text.replace('x',''))
            for e in elements if e.text.strip()
        ]

        df = pd.read_csv(LOG_FILE, parse_dates=['timestamp'])
        last_round = df['round'].max() if not df.empty else 0

        for idx, m in enumerate(multipliers[-5:], start=1):
            round_num = last_round + idx
            if round_num in df['round'].values:
                continue

            timestamp = datetime.now()
            bet = random.uniform(simulate_bet_min, simulate_bet_max)
            win = bet * m

            total_bet = df['bet_amount'].sum() + bet if not df.empty else bet
            total_win = df['winnings'].sum() + win if not df.empty else win
            net = total_win - total_bet

            new_row = pd.DataFrame([{
                'timestamp': timestamp,
                'round': round_num,
                'multiplier': m,
                'bet_amount': bet,
                'winnings': win,
                'cumulative_bets': total_bet,
                'cumulative_wins': total_win,
                'net_profit': net,
                'inter_round_seconds': 0
            }])

            df = pd.concat([df, new_row], ignore_index=True)

        if df.empty:
            time.sleep(refresh_rate)
            continue

        # AI Prediction
        model, scaler, features = train_ai(df)
        ai_prob = 0

        if model:
            latest = rolling_features(df).iloc[-1]
            X = scaler.transform([latest[features]])
            ai_prob = model.predict_proba(X)[0][1]

        df['ai_big_prob'] = ai_prob

        df.to_csv(LOG_FILE, index=False)
        backup_csv(df)

        # Dashboard
        with placeholder.container():
            st.subheader("Latest Rounds")
            st.dataframe(df.tail(20))

            fig = px.line(df, x='round', y='multiplier', title="Multiplier Trend")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("AI Prediction")
            st.metric("Probability of >5x next round", f"{ai_prob:.2f}")

        time.sleep(refresh_rate)

    except Exception as e:
        st.error(f"Error: {e}")
        time.sleep(refresh_rate)
