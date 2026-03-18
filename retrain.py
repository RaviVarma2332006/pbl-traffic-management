import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle

print("Generating Master Regional Dataset (Sus, Lavale, Nande, Pashan)...")
np.random.seed(42)
data_size = 4500

hours = np.random.randint(0, 24, data_size)
months = np.random.randint(1, 13, data_size)
is_weekday = np.random.randint(0, 2, data_size)
is_steep_route = np.random.randint(0, 2, data_size)
is_blocked = np.random.choice([0, 1], p=[0.95, 0.05], size=data_size)

# 0 = Sus/Baner, 1 = Lavale Village, 2 = Nande, 3 = Pashan/Sutarwadi
route_region = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.2, 0.3], size=data_size)

congestion_labels, accident_labels, speeds, aqis = [], [], [], []

for h, m, w, steep, blocked, region in zip(hours, months, is_weekday, is_steep_route, is_blocked, route_region):
    # --- TRAFFIC LOGIC ---
    is_rush = (8 <= h <= 11) or (17 <= h <= 21) # Pashan rush hours are longer
    is_monsoon = 6 <= m <= 9
    is_winter = m in [12, 1]
    is_summer_break = 4 <= m <= 5
    
    if region == 3: # Pashan / Sutarwadi Logic
        if w == 1 and is_rush: cong = 'High'
        elif w == 0 and (17 <= h <= 22): cong = 'High' # Weekend evening highway rush
        elif is_summer_break: cong = 'Low'
        elif 7 <= h <= 22: cong = 'Medium'
        else: cong = 'Low'
    else: # Sus, Lavale, Nande Logic
        if w == 1 and ((8 <= h <= 10) or (17 <= h <= 20)) and not is_summer_break:
            cong = 'High' if region != 2 else 'Medium'
        elif is_monsoon and (7 <= h <= 21):
            cong = np.random.choice(['Medium', 'High']) if region != 2 else 'Medium'
        elif is_summer_break or h < 7 or h > 21 or w == 0:
            cong = 'Low'
        else: 
            cong = 'Medium'
        
    # --- RISK LOGIC ---
    if region == 3 and cong == 'High' and (is_monsoon or is_winter): 
        risk = 'Critical' # Pashan suffers in Monsoon AND Winter Fog
    elif region == 1 and is_monsoon and cong == 'High': 
        risk = 'Critical' # Lavale monsoon spikes
    elif region == 2:
        risk = 'Normal' # Nande is historically safer
    elif steep == 1 and cong != 'Low': 
        risk = 'Elevated'
    else: 
        risk = 'Normal'
        
    # --- SPEED CAPS ---
    if cong == 'High': spd = np.random.uniform(3, 10)     
    elif cong == 'Medium': spd = np.random.uniform(10, 20) 
    else: spd = np.random.uniform(20, 35)                   
    if steep == 1: spd *= 0.75 

    # --- AQI LOGIC ---
    base_aqi = 60
    if is_winter: base_aqi += 45
    elif is_monsoon: base_aqi -= 25
    
    if region == 3 and h >= 17 and cong == 'High': aqi = base_aqi + np.random.uniform(100, 150) # Pashan evening AQI accumulation > 200
    elif region == 1 and cong == 'High': aqi = base_aqi + np.random.uniform(80, 100)
    elif region == 2 and cong == 'High': aqi = base_aqi + np.random.uniform(20, 40)
    elif cong == 'High': aqi = base_aqi + np.random.uniform(60, 100)
    elif cong == 'Medium': aqi = base_aqi + np.random.uniform(25, 50)
    else: aqi = base_aqi + np.random.uniform(0, 15)

    if blocked == 1:
        cong = 'High'
        risk = 'Critical'
        spd = np.random.uniform(0, 2) 
        aqi += 50 

    congestion_labels.append(cong)
    accident_labels.append(risk)
    speeds.append(round(spd, 1))
    aqis.append(int(aqi))

df = pd.DataFrame({'Hour': hours, 'Month': months, 'Is_Weekday': is_weekday, 
                   'Is_Steep_Route': is_steep_route, 'Is_Blocked': is_blocked, 'Route_Region': route_region,
                   'Congestion': congestion_labels, 'Risk': accident_labels, 'Speed': speeds, 'AQI': aqis})

X = df[['Hour', 'Month', 'Is_Weekday', 'Is_Steep_Route', 'Is_Blocked', 'Route_Region']]

print("Training Master 6-Variable Models...")
m_cong = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, df['Congestion'])
m_risk = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, df['Risk'])
m_spd = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, df['Speed'])
m_aqi = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, df['AQI'])

with open('congestion_model.pkl', 'wb') as f: pickle.dump(m_cong, f)
with open('risk_model.pkl', 'wb') as f: pickle.dump(m_risk, f)
with open('speed_model.pkl', 'wb') as f: pickle.dump(m_spd, f)
with open('aqi_model.pkl', 'wb') as f: pickle.dump(m_aqi, f)
print("Success! .pkl files updated for Sus, Lavale, Nande, AND Pashan.")