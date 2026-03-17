from flask import Flask, request, render_template
import pickle
from datetime import datetime
import requests 

app = Flask(__name__)

with open('congestion_model.pkl', 'rb') as f: m_cong = pickle.load(f)
with open('risk_model.pkl', 'rb') as f: m_risk = pickle.load(f)
with open('speed_model.pkl', 'rb') as f: m_spd = pickle.load(f)
with open('aqi_model.pkl', 'rb') as f: m_aqi = pickle.load(f)

def get_realtime_weather():
    try:
        lat, lon = "18.548", "73.744" # Central point for area
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code"
        aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi"
        
        w_res = requests.get(weather_url).json()
        a_res = requests.get(aqi_url).json()
        
        temp = w_res['current']['temperature_2m']
        w_code = w_res['current']['weather_code']
        real_aqi = a_res['current']['us_aqi'] 
        
        if w_code in [0, 1]: condition = "Clear / Sunny"
        elif w_code in [2, 3]: condition = "Partly Cloudy"
        elif w_code in [45, 48]: condition = "Foggy"
        elif w_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: condition = "Raining"
        elif w_code in [95, 96, 99]: condition = "Thunderstorm"
        else: condition = "Unknown"
        
        return {"temp": f"{temp}°C", "condition": condition, "real_aqi": int(real_aqi)}
    except Exception as e:
        return {"temp": "N/A", "condition": "Offline", "real_aqi": "N/A"}

def find_best_time(start_hour, month, is_weekday, is_steep, is_blocked, route_region, is_live=False, current_minute=0):
    best_offset, best_score = 0, -1000
    for offset in range(2): 
        check_hour = (start_hour + offset) % 24
        sim_blocked = is_blocked if offset == 0 else 0
        test_scenario = [[check_hour, month, is_weekday, is_steep, sim_blocked, route_region]]
        
        test_cong = m_cong.predict(test_scenario)[0]
        test_risk = m_risk.predict(test_scenario)[0]
        test_speed = m_spd.predict(test_scenario)[0]
        
        score = test_speed 
        if test_cong == 'High': score -= 50
        elif test_cong == 'Medium': score -= 20
        if test_risk == 'Critical': score -= 100
        elif test_risk == 'Elevated': score -= 40
        
        if score > best_score: best_score, best_offset = score, offset

    if is_live:
        if best_offset == 0: return "Leave Now (Optimal)"
        else:
            next_h = (start_hour + 1) % 24
            return f"Wait 1 Hr (Leave ~{next_h if next_h <= 12 and next_h != 0 else (next_h-12 if next_h != 0 else 12)}:{current_minute:02d} {'AM' if next_h < 12 else 'PM'})"
    else:
        best_h = (start_hour + best_offset) % 24
        disp_h = best_h if best_h <= 12 and best_h != 0 else (best_h-12 if best_h != 0 else 12)
        return f"Leave at {disp_h}:00 {'AM' if best_h < 12 else 'PM'} (No Wait)" if best_offset == 0 else f"Wait 1 Hr (Leave at {disp_h}:00 {'AM' if best_h < 12 else 'PM'})"

def get_live_predictions():
    now = datetime.now()
    scenario_live = [[now.hour, now.month, 1 if now.weekday() < 5 else 0, 0, 0, 0]]
    weather_data = get_realtime_weather()
    return {
        'time_str': now.strftime("%I:%M %p, %b %d"),
        'congestion': m_cong.predict(scenario_live)[0],
        'risk': m_risk.predict(scenario_live)[0],
        'speed': round(m_spd.predict(scenario_live)[0], 1),
        'best_time': find_best_time(now.hour, now.month, 1 if now.weekday() < 5 else 0, 0, 0, 0, is_live=True, current_minute=now.minute),
        'weather': weather_data
    }

@app.route('/', methods=['GET', 'POST'])
def home():
    live_data = get_live_predictions()
    if request.method == 'POST':
        h, m, w = int(request.form['hour']), int(request.form['month']), int(request.form['is_weekday'])
        steep, blk, region = int(request.form['is_steep']), int(request.form['is_blocked']), int(request.form['route_region'])

        scenario = [[h, m, w, steep, blk, region]]
        scenario_data = {
            'congestion': m_cong.predict(scenario)[0],
            'risk': m_risk.predict(scenario)[0],
            'speed': round(m_spd.predict(scenario)[0], 1),
            'aqi': int(m_aqi.predict(scenario)[0]),
            'best_time': find_best_time(h, m, w, steep, blk, region, is_live=False)
        }
        return render_template('index.html', live=live_data, scenario=scenario_data)
    
    return render_template('index.html', live=live_data, scenario=None)

if __name__ == '__main__':
    app.run(debug=True)