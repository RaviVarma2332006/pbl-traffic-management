import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle

print("Generating Construction-Aware Dataset...")
np.random.seed(42)
data_size = 2500

hours = np.random.randint(0, 24, data_size)
months = np.random.randint(1, 13, data_size)
is_weekday = np.random.randint(0, 2, data_size)
is_steep_route = np.random.randint(0, 2, data_size)
is_blocked = np.random.choice([0, 1], p=[0.95, 0.05], size=data_size)

congestion_labels, accident_labels, speeds, aqis = [], [], [], []

for h, m, w, steep, blocked in zip(hours, months, is_weekday, is_steep_route, is_blocked):
    # Traffic Logic
    if w == 1 and ((8 <= h <= 10) or (17 <= h <= 20)): cong = 'High'
    elif 6 <= m <= 9 and (7 <= h <= 21): cong = np.random.choice(['Medium', 'High'])
    elif 4 <= m <= 5 or h < 7 or h > 21: cong = 'Low'
    elif w == 0: cong = np.random.choice(['Low', 'Medium'], p=[0.7, 0.3])
    else: cong = 'Medium'
        
    # Risk Logic
    if steep == 1 and (6 <= m <= 9) and cong == 'High': risk = 'Critical'
    elif (steep == 1 and cong != 'Low') or (steep == 0 and 6 <= m <= 9 and cong == 'High'): risk = 'Elevated'
    else: risk = 'Normal'
        
    # --- NEW REALISTIC SPEED CAPS (Construction / Bad Roads) ---
    if cong == 'High': spd = np.random.uniform(3, 10)     # Gridlock
    elif cong == 'Medium': spd = np.random.uniform(10, 20)  # Bumper to bumper
    else: spd = np.random.uniform(20, 30)                   # Best case scenario, capped at 30
    
    # Extra penalty for the steep Sunny's World patch
    if steep == 1: spd *= 0.75 

    # AQI Logic
    base_aqi = 60
    if m in [11, 12, 1, 2]: base_aqi += 45
    elif m in [6, 7, 8, 9]: base_aqi -= 25
    if cong == 'High': aqi = base_aqi + np.random.uniform(60, 100)
    elif cong == 'Medium': aqi = base_aqi + np.random.uniform(25, 50)
    else: aqi = base_aqi + np.random.uniform(0, 15)

    # Blockage Override
    if blocked == 1:
        cong = 'High'
        risk = 'Critical'
        spd = np.random.uniform(0, 2) # Complete standstill
        aqi += 50 

    congestion_labels.append(cong)
    accident_labels.append(risk)
    speeds.append(round(spd, 1))
    aqis.append(int(aqi))

df = pd.DataFrame({'Hour': hours, 'Month': months, 'Is_Weekday': is_weekday, 
                   'Is_Steep_Route': is_steep_route, 'Is_Blocked': is_blocked,
                   'Congestion': congestion_labels, 'Risk': accident_labels, 'Speed': speeds, 'AQI': aqis})

X = df[['Hour', 'Month', 'Is_Weekday', 'Is_Steep_Route', 'Is_Blocked']]

print("Training Upgraded Realistic Models...")
m_cong = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, df['Congestion'])
m_risk = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, df['Risk'])
m_spd = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, df['Speed'])
m_aqi = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, df['AQI'])

with open('congestion_model.pkl', 'wb') as f: pickle.dump(m_cong, f)
with open('risk_model.pkl', 'wb') as f: pickle.dump(m_risk, f)
with open('speed_model.pkl', 'wb') as f: pickle.dump(m_spd, f)
with open('aqi_model.pkl', 'wb') as f: pickle.dump(m_aqi, f)
print("Success! Overwrote the .pkl files with realistic construction-zone speeds.")