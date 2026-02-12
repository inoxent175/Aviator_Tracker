# ==========================================
# 🚀 Aviator Tracker & AI-Powered Predictor (Cloud Version)
# Fully Online, Streamlit Cloud Compatible
# ==========================================
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import os
import time
import random
import threading
import glob
from datetime import datetime

# -----------------------------
# CSV log & backup folders
# -----------------------------
LOG_FILE = "aviator_history.csv"
BACKUP_FOLDER = "backups"
MAX_BACKUPS = 20
os.makedirs(BACKUP_FOLDER, exist_ok=True)

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        'timestamp','round','multiplier','bet_amount','winnings','cumulative_bets','cumulative_wins','net_profit',
        '>1.5x','>3x','>5x','>10x','confidence','alert_triggered','big_multiplier_spike','ai_big_prob','ai_confidence','inter_round_seconds'
    ]).to_csv(LOG_FILE, index=False)

def backup_csv(df_feat):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_file = os.path.join(BACKUP_FOLDER, f"aviator_history_{timestamp}.csv")
    df_feat.to_csv(backup_file, index=False)
    backups = sorted(os.listdir(BACKUP_FOLDER))
    if len(backups) > MAX_BACKUPS:
        for old_backup in backups[:-MAX_BACKUPS]:
            os.remove(os.path.join(BACKUP_FOLDER, old_backup))

# -----------------------------
# Streamlit Setup
# -----------------------------
st.set_page_config(page_title="🚀 Aviator Tracker Ultimate", layout="wide")
st.title("🚀 Aviator Tracker & AI-Powered Predictor")

# -----------------------------
# Sidebar Settings
# -----------------------------
st.sidebar.header("Settings")
refresh_rate = st.sidebar.slider("Refresh interval (seconds)", 5, 30, 10)
alert_conf_threshold = st.sidebar.selectbox("Min Confidence for Alerts", ["Low","Medium","High","Best"], index=3)
COLUMNS = st.sidebar.slider("Columns in UI Layout", 7, 10, 8)
simulate_bet_min = st.sidebar.number_input("Min Simulated Bet", 1, 100, 1)
simulate_bet_max = st.sidebar.number_input("Max Simulated Bet", 1, 500, 10)
confidence_order = {"Low":1,"Medium":2,"High":3,"Best":4}
threshold_value = confidence_order[alert_conf_threshold]

# -----------------------------
# Alerts (browser-friendly)
# -----------------------------
def alert_user(message):
    st.warning(message)  # Display alerts in the web app

# -----------------------------
# Web Scraping Setup
# -----------------------------
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Automatically download driver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

URL = "https://apkpkcas16.royalxcasino.club/"  # Replace with real Aviator URL
ROUND_SELECTOR = ".round-multiplier"
driver.get(URL)

# -----------------------------
# Core Functions
# -----------------------------
def arrange_columns(multipliers, columns=COLUMNS):
    return [multipliers[i:i+columns] for i in range(0,len(multipliers),columns)]

def vertical_streaks(rows, low_thresh=1.5, high_thresh=5):
    max_cols = max(len(r) for r in rows)
    low_streaks = [0]*max_cols
    high_streaks = [0]*max_cols
    for col in range(max_cols):
        streak_low = streak_high = max_low = max_high = 0
        for row in rows:
            if col<len(row):
                val = row[col]
                streak_low = streak_low+1 if val<low_thresh else 0
                streak_high = streak_high+1 if val>high_thresh else 0
                max_low = max(max_low, streak_low)
                max_high = max(max_high, streak_high)
        low_streaks[col]=max_low
        high_streaks[col]=max_high
    return low_streaks, high_streaks

def rolling_features(df, window=5):
    df_feat = df.copy()
    df_feat[f'avg_last_{window}'] = df_feat['multiplier'].rolling(window).mean()
    df_feat[f'max_last_{window}'] = df_feat['multiplier'].rolling(window).max()
    df_feat[f'low_count_last_{window}'] = df_feat['multiplier'].rolling(window).apply(lambda x:(x<1.5).sum(), raw=True)
    df_feat[f'high_count_last_{window}'] = df_feat['multiplier'].rolling(window).apply(lambda x:(x>5).sum(), raw=True)
    def longest_low_streak(series):
        max_streak = streak = 0
        for val in series:
            if val<1.5:
                 streak+=1
                 max_streak=max(max_streak,streak)
            else: 
                streak=0
        return max_streak
    df_feat[f'longest_low_streak_last_{window}'] = df_feat['multiplier'].rolling(window).apply(longest_low_streak, raw=True)
    return df_feat

def predict_next_round(row):
    prob = {'>1.5x':0.35,'>3x':0.15,'>5x':0.08,'>10x':0.04}
    if row['longest_low_streak_last_5']>=3:
         prob['>1.5x']+=0.2
         prob['>3x']+=0.05
    if row['high_count_last_5']==0:
         prob['>5x']+=0.05
         prob['>10x']+=0.01
    max_prob = max(prob.values())
    if max_prob>0.5:
        conf="Best"
    elif max_prob>0.35:
        conf="High"
    elif max_prob>0.2:
        conf="Medium"
    else:
        conf="Low"
    return prob, conf

# -----------------------------
# AI Predictor
# -----------------------------
def load_all_sessions(log_file=LOG_FILE, backup_folder=BACKUP_FOLDER):
    files = glob.glob(os.path.join(backup_folder,"*.csv")) + [log_file]
    sessions = {}
    for f in files:
        try:
            df_sess = pd.read_csv(f, parse_dates=['timestamp'])
            sessions[os.path.basename(f).replace('.csv','')] = df_sess
        except Exception as e:
            print(f"Error Loading {f}: {e}")
    return sessions

def prepare_training_data(df_all):
    df_all = rolling_features(df_all, window=5)
    df_all['big_multiplier']=(df_all['multiplier']>5).astype(int)
    features = ['avg_last_5','max_last_5','low_count_last_5','high_count_last_5','longest_low_streak_last_5','vert_low_streak','vert_high_streak']
    df_all=df_all.dropna(subset=features+['big_multiplier'])
    X = df_all[features].values
    y = df_all['big_multiplier'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, features

def train_ai_predictor(df_all):
    X, y, scaler, features = prepare_training_data(df_all)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X,y)
    return model, scaler, features

def predict_ai_next_round(model, scaler, features, latest_row):
    X = np.array([latest_row[features].values])
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[0][1]

# -----------------------------
# Live Loop (Streamlit Cloud)
# -----------------------------
sessions_data = load_all_sessions()
placeholder = st.empty()
alerted_rounds = set()

while True:
    try:
        # Scrape multipliers
        elements = driver.find_elements(By.CSS_SELECTOR, ROUND_SELECTOR)
        multipliers = [float(e.text.replace('x','')) for e in elements if e.text.strip()]
        df = pd.read_csv(LOG_FILE)
        last_round_num = df['round'].max() if not df.empty else 0

        # Add new rounds
        for idx, m in enumerate(multipliers[-10:], start=1):
            round_num = last_round_num+idx
            if round_num in df['round'].values:
                continue
            timestamp = datetime.now()
            bet_amount = random.uniform(simulate_bet_min, simulate_bet_max)
            winnings = bet_amount*m
            cumulative_bets = df['bet_amount'].sum()+bet_amount if not df.empty else bet_amount
            cumulative_wins = df['winnings'].sum()+winnings if not df.empty else winnings
            net_profit = cumulative_wins-cumulative_bets
            inter_round_seconds = (timestamp - df['timestamp'].iloc[-1]).total_seconds() if not df.empty else 0
            new_row = pd.DataFrame([{
                'timestamp':timestamp,'round':round_num,'multiplier':m,'bet_amount':bet_amount,
                'winnings':winnings,'cumulative_bets':cumulative_bets,'cumulative_wins':cumulative_wins,
                'net_profit':net_profit,'inter_round_seconds':inter_round_seconds
            }])
            df=pd.concat([df,new_row],ignore_index=True)

        # Features & predictions
        df_feat=rolling_features(df)
        rows=arrange_columns(df_feat['multiplier'].values,COLUMNS)
        vert_low, vert_high=vertical_streaks(rows)
        df_feat['vert_low_streak']=list(np.resize(vert_low,len(df_feat)))
        df_feat['vert_high_streak']=list(np.resize(vert_high,len(df_feat)))
        df_feat[['>1.5x','>3x','>5x','>10x','confidence']] = df_feat.apply(lambda r: pd.Series([*predict_next_round(r)[0].values(), predict_next_round(r)[1]]), axis=1)
        window_size=3
        for col in ['>1.5x','>3x','>5x','>10x']:
            df_feat[f'{col}_smoothed']=df_feat[col].rolling(window_size,min_periods=1).mean()
        df_feat['big_multiplier_spike']=df_feat['>5x_smoothed']>0.6
        df_feat['alert_triggered']=df_feat.apply(lambda r: 1 if confidence_order.get(r['confidence'],0)>=threshold_value else 0, axis=1)

        # AI prediction
        all_sessions=load_all_sessions()
        df_full=pd.concat(all_sessions.values(),ignore_index=True)
        ai_model, ai_scaler, ai_features=train_ai_predictor(df_full)
        latest_row=df_feat.iloc[-1]
        ai_prob=predict_ai_next_round(ai_model, ai_scaler, ai_features, latest_row)
        df_feat.at[df_feat.index[-1],'ai_big_prob']=ai_prob
        df_feat.at[df_feat.index[-1],'ai_confidence']=("Best" if ai_prob>0.6 else "High" if ai_prob>0.45 else "Medium" if ai_prob>0.3 else "Low")

        # Save & backup
        df_feat.to_csv(LOG_FILE,index=False)
        backup_csv(df_feat)

        # Alerts
        latest_round=df_feat.iloc[-1]
        if latest_round['alert_triggered']==1 and latest_round['round'] not in alerted_rounds:
            alerted_rounds.add(latest_round['round'])
            alert_user(f"Round {int(latest_round['round'])} predicted {latest_round['confidence']} confidence multipliers >1.5x")
        if latest_round['big_multiplier_spike'] and latest_round['round'] not in alerted_rounds:
            alerted_rounds.add(latest_round['round'])
            alert_user(f"Round {int(latest_round['round'])} has a high probability spike (>5x smoothed)")
        if latest_round['ai_confidence'] in ['High','Best'] and latest_round['round'] not in alerted_rounds:
            alerted_rounds.add(latest_round['round'])
            alert_user(f"AI predicts BIG multiplier next round with {latest_round['ai_confidence']} confidence ({ai_prob:.2f})")

        # Dashboard Display
        with placeholder.container():
            st.subheader("Latest Rounds & Session Analytics")
            st.dataframe(df_feat.tail(20))

            st.subheader("Multiplier Heatmap")
            heatmap_data=arrange_columns(df_feat['multiplier'].values,COLUMNS)
            fig_heatmap=px.imshow(heatmap_data,text_auto=True,color_continuous_scale='Viridis')
            st.plotly_chart(fig_heatmap,use_container_width=True)

            st.subheader("Smoothed Big Multiplier Probabilities")
            prob_df=df_feat[['round','>1.5x_smoothed','>3x_smoothed','>5x_smoothed','>10x_smoothed']].copy().set_index('round')
            fig_prob=px.line(prob_df,title='Smoothed Big Multiplier Probabilities')
            spike_df=df_feat[df_feat['big_multiplier_spike']]
            fig_prob.add_scatter(x=spike_df['round'], y=spike_df['>5x_smoothed'], mode='markers', marker=dict(size=10,color='red',symbol='star'), name='Spike')
            fig_prob.update_layout(yaxis=dict(range=[0,1]))
            st.plotly_chart(fig_prob,use_container_width=True)

            st.subheader("AI-Predicted Big Multipliers")
            fig_ai=px.line(df_feat.tail(50),x='round',y='ai_big_prob',title="AI Predicted Probability (>5x) Last 50 Rounds")
            st.plotly_chart(fig_ai,use_container_width=True)

        time.sleep(refresh_rate)

    except Exception as e:
        st.error(f"Error in live loop: {e}")
        time.sleep(refresh_rate)
