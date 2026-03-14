from flask import Flask, request, render_template
import pickle
from datetime import datetime
import requests # NEW: For fetching internet API data

app = Flask(__name__)

# --- 1. LOAD THE MODELS ---
with open('congestion_model.pkl', 'rb') as f: m_cong = pickle.load(f)
with open('risk_model.pkl', 'rb') as f: m_risk = pickle.load(f)
with open('speed_model.pkl', 'rb') as f: m_spd = pickle.load(f)
with open('aqi_model.pkl', 'rb') as f: m_aqi = pickle.load(f)

# --- 2. NEW: FETCH REAL-TIME WEATHER API ---
# --- 2. FETCH REAL-TIME WEATHER API (Hyper-Local Sus Gaon) ---
def get_realtime_weather():
    try:
        # Hyper-local GPS Coordinates for Sus Gaon / Hinjewadi Route
        lat = "18.548"
        lon = "73.744"
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code"
        # We changed this from 'pm2_5' to 'us_aqi' to get the 0-500 Index Score
        aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi"
        
        w_res = requests.get(weather_url).json()
        a_res = requests.get(aqi_url).json()
        
        temp = w_res['current']['temperature_2m']
        w_code = w_res['current']['weather_code']
        # Extract the calculated AQI Index
        real_aqi = a_res['current']['us_aqi'] 
        
        # Translate the WMO Weather Code to English
        if w_code in [0, 1]: condition = "Clear / Sunny"
        elif w_code in [2, 3]: condition = "Partly Cloudy"
        elif w_code in [45, 48]: condition = "Foggy"
        elif w_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: condition = "Raining"
        elif w_code in [95, 96, 99]: condition = "Thunderstorm"
        else: condition = "Unknown"
        
        return {"temp": f"{temp}°C", "condition": condition, "real_aqi": int(real_aqi)}
    except Exception as e:
        return {"temp": "N/A", "condition": "Offline", "real_aqi": "N/A"}


# --- 3. RECOMMENDER FUNCTION ---
def find_best_time(start_hour, month, is_weekday, is_steep, is_blocked, is_live=False, current_minute=0):
    best_offset = 0
    best_score = -1000
    
    for offset in range(2): 
        check_hour = (start_hour + offset) % 24
        sim_blocked = is_blocked if offset == 0 else 0
        test_scenario = [[check_hour, month, is_weekday, is_steep, sim_blocked]]
        
        test_cong = m_cong.predict(test_scenario)[0]
        test_risk = m_risk.predict(test_scenario)[0]
        test_speed = m_spd.predict(test_scenario)[0]
        
        score = test_speed 
        if test_cong == 'High': score -= 50
        elif test_cong == 'Medium': score -= 20
        if test_risk == 'Critical': score -= 100
        elif test_risk == 'Elevated': score -= 40
        
        if score > best_score:
            best_score = score
            best_offset = offset

    if is_live:
        if best_offset == 0: return "Leave Now (Optimal)"
        else:
            next_hour = (start_hour + 1) % 24
            am_pm = "AM" if next_hour < 12 else "PM"
            disp_h = next_hour if next_hour <= 12 else next_hour - 12
            disp_h = 12 if disp_h == 0 else disp_h
            min_str = f"{current_minute:02d}"
            return f"Wait 1 Hr (Leave ~{disp_h}:{min_str} {am_pm})"
    else:
        best_hour = (start_hour + best_offset) % 24
        am_pm = "AM" if best_hour < 12 else "PM"
        disp_h = best_hour if best_hour <= 12 else best_hour - 12
        disp_h = 12 if disp_h == 0 else disp_h
        if best_offset == 0: return f"Leave at {disp_h}:00 {am_pm} (No Wait)"
        else: return f"Wait 1 Hr (Leave at {disp_h}:00 {am_pm})"

# --- 4. LIVE LOGIC ---
def get_live_predictions():
    now = datetime.now()
    hour = now.hour
    minute = now.minute 
    month = now.month
    is_weekday = 1 if now.weekday() < 5 else 0
    scenario_live = [[hour, month, is_weekday, 0, 0]]
    
    # Grab the internet weather data!
    weather_data = get_realtime_weather()
    
    return {
        'time_str': now.strftime("%I:%M %p, %b %d"),
        'congestion': m_cong.predict(scenario_live)[0],
        'risk': m_risk.predict(scenario_live)[0],
        'speed': round(m_spd.predict(scenario_live)[0], 1),
        'best_time': find_best_time(hour, month, is_weekday, 0, is_blocked=0, is_live=True, current_minute=minute),
        'weather': weather_data # Send weather to HTML
    }

# --- 5. ROUTING ---
@app.route('/', methods=['GET', 'POST'])
def home():
    live_data = get_live_predictions()
    
    if request.method == 'POST':
        hour = int(request.form['hour'])
        month = int(request.form['month'])
        is_weekday = int(request.form['is_weekday'])
        is_steep = int(request.form['is_steep'])
        is_blocked = int(request.form['is_blocked'])

        scenario = [[hour, month, is_weekday, is_steep, is_blocked]]
        
        scenario_data = {
            'congestion': m_cong.predict(scenario)[0],
            'risk': m_risk.predict(scenario)[0],
            'speed': round(m_spd.predict(scenario)[0], 1),
            'aqi': int(m_aqi.predict(scenario)[0]),
            'best_time': find_best_time(hour, month, is_weekday, is_steep, is_blocked, is_live=False)
        }
        return render_template('index.html', live=live_data, scenario=scenario_data)
    
    return render_template('index.html', live=live_data, scenario=None)

if __name__ == '__main__':
    app.run(debug=True)