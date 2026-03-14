# Student Mobility & Traffic Dashboard (SIT Pune)

An AI-driven, real-time smart city dashboard designed for students commuting to Symbiosis Institute of Technology (SIT) at Lavale, Pune.

## Features
* **Live AI Predictions:** Uses Random Forest models to predict traffic congestion, accident risk, and AQI based on time and weather.
* **Real-Time Environmental Data:** Fetches live weather and AQI via the Open-Meteo API.
* **Smart Commute Recommender:** A look-ahead loop calculates the safest and fastest time to leave within a 1-hour window.
* **Interactive Mapping:** Uses Leaflet.js and OSRM to draw real-time physical driving routes from local Baner/Sus PGs to the campus.
* **MaaS Price Engine:** Simulates real-time surge pricing and generates deep links to book rides on Uber, Ola, and Rapido.

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run the server: `python app.py`
3. Open `http://127.0.0.1:5000` in your browser.