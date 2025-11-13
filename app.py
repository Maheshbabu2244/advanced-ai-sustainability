# FULL STREAMLIT DASHBOARD
# Uses:
# - Cerebras API (text)
# - OpenStreetMap (Folium)
# - OpenRouteService (routing + geocoding)
# - OpenChargeMap API
# - No Google Maps required

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import openrouteservice
from openrouteservice import convert
from prophet import Prophet
import shap
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from PIL import Image
import folium
from streamlit_folium import st_folium
import requests
import random
import re
import json
import base64
import io

# -------------------------------
# CONFIGURATION
# -------------------------------
st.set_page_config(page_title="AI Sustainability Dashboard", layout="wide", page_icon="🌍")

# ---------- Custom CSS for iOS-like liquid theme & animations ----------
st.markdown("""
<style>
:root{--bg:#0f1724;--card:#0b1220;--accent:#3ddc84;--muted:#9aa7bf}
body {background: radial-gradient(ellipse at 10% 0%, rgba(61,220,132,0.08), transparent 10%), radial-gradient(ellipse at 90% 100%, rgba(3,137,255,0.04), transparent 15%), var(--bg); color: #E6EEF8}
.css-1d391kg{background:transparent}
.stApp > header {background: linear-gradient(90deg, rgba(61,220,132,0.1), rgba(3,137,255,0.06));}
.section-card{background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:16px; border-radius:14px; box-shadow:0 6px 20px rgba(2,6,23,0.6);}
.header-row{display:flex;align-items:center;gap:12px}
.logo-circle{width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,#3ddc84,#0389ff);display:flex;align-items:center;justify-content:center;font-weight:700}
.blob{position:fixed;filter:blur(60px);opacity:0.35;border-radius:50%;}
.blob.one{width:420px;height:420px;background:#3ddc84;left:-120px;top:-80px}
.blob.two{width:520px;height:520px;background:#0389ff;right:-140px;bottom:-120px}
.small-muted{color:var(--muted);font-size:12px}
.card-title{font-size:18px;font-weight:700}
</style>
""", unsafe_allow_html=True)

# Floating blobs
st.markdown('<div class="blob one"></div><div class="blob two"></div>', unsafe_allow_html=True)

# --------------- API KEYS ------------------
# Expected keys in .streamlit/secrets.toml:
try:
    ORS_KEY = st.secrets["ORS_API_KEY"]
    OPENCHARGEMAP_KEY = st.secrets["OPENCHARGEMAP_API_KEY"]
    # --- CEREBRAS API ---
    CEREBRAS_KEY = st.secrets["CEREBRAS_API_KEY"]
    CEREBRAS_URL = st.secrets["CEREBRAS_ENDPOINT_URL"]
except Exception as e:
    st.error(f"Missing API keys. Add ORS_API_KEY, OPENCHARGEMAP_API_KEY, CEREBRAS_API_KEY, and CEREBRAS_ENDPOINT_URL to .streamlit/secrets.toml. Error: {e}")
    st.stop()

# ------------- Initialize ORS --------------
ors_client = openrouteservice.Client(key=ORS_KEY)


# ---------------------------------------------
# Helper Functions
# ---------------------------------------------

def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)

# --- !! CORRECTED CEREBRAS AI FUNCTION !! ---
@st.cache_data 
def get_ai_response(prompt, max_tokens=512, temperature=0.2):
    """Generates a text response using the Cerebras chat/completions API."""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CEREBRAS_KEY}"
    }

    # --- PAYLOAD UPDATED for chat/completions endpoint ---
    # We must specify a model. You can change this to any model ID you have access to.
    payload = {
        "model": "btlm-3b-8k-base", # Or "llama-3-8b-instruct", etc.
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        # CEREBRAS_URL should be "https://api.cerebras.ai/v1/chat/completions"
        response = requests.post(CEREBRAS_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # Raise an error for bad status codes
        
        data = response.json()
        
        # --- RESPONSE PATH UPDATED for chat/completions ---
        # The text is now at 'choices'[0]['message']['content']
        text = data.get('choices', [{}])[0].get('message', {}).get('content')
        
        if text:
            return clean_html(text)
        else:
            # If text is not found, show the raw response for debugging
            st.error(f"AI Error: Could not parse response. Response: {data}")
            return "AI Error: Response format not understood."

    except requests.exceptions.HTTPError as http_err:
        st.error(f"Cerebras API HTTP Error: {http_err} - Response: {http_err.response.text}")
        return "AI Error: HTTP error. Check your API key, endpoint URL, and model name."
    except Exception as e:
        st.error(f"Cerebras API Error: {e}")
        return "AI Error: Could not generate response."


# --- CACHE ADDED ---
@st.cache_data
def get_dummy_buildings(lat, lon, n=80):
    buildings = []
    for i in range(n):
        lat_off = np.random.randn() * 0.02
        lon_off = np.random.randn() * 0.02
        size = np.random.uniform(0.0005, 0.0016)
        bounds = [[lat + lat_off - size, lon + lon_off - size], [lat + lat_off + size, lon + lon_off + size]]
        buildings.append({
            "bounds": bounds,
            "solar_potential": random.randint(30, 100)
        })
    return pd.DataFrame(buildings)


# --- CACHE ADDED ---
@st.cache_data
def get_ev_stations(lat, lon):
    try:
        url = "https://api.openchargemap.io/v3/poi"
        params = {
            "output": "json",
            "latitude": lat,
            "longitude": lon,
            "distance": 20,
            "distanceunit": "km",
            "maxresults": 200,
            "key": OPENCHARGEMAP_KEY,
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status() 
        data = r.json()
        stations = []
        for cp in data:
            ai = cp.get("AddressInfo", {})
            if ai.get("Latitude") and ai.get("Longitude"): 
                stations.append({
                    "lat": ai.get("Latitude"),
                    "lon": ai.get("Longitude"),
                    "name": ai.get("Title", "Unknown"),
                })
        return pd.DataFrame(stations)
    except Exception as e:
        st.warning(f"OpenChargeMap error: {e}")
        return pd.DataFrame()


# --- CACHE ADDED & VERIFIED ---
@st.cache_data
def ors_geocode(location):
    """Geocodes a location using OpenRouteService."""
    try:
        url = "https://api.openrouteservice.org/geocode/search"
        params = {"api_key": ORS_KEY, "text": location}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get('features'):
            coords = data['features'][0]['geometry']['coordinates']
            return coords[1], coords[0] # Return (lat, lon)
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Geocoding network error for '{location}': {e}")
        return None
    except Exception as e:
        st.error(f"Geocoding error for '{location}': {e}")
        return None


# --- CACHE ADDED ---
@st.cache_data
def create_forecast():
    ds = pd.date_range(start='2023-01-01', periods=700, freq='D')
    y = 50 + 10*np.sin(np.arange(len(ds))/30) + np.random.normal(0, 3, len(ds))
    df = pd.DataFrame({"ds": ds, "y": y})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    return df, forecast


# --- XAI FIX ---
@st.cache_resource
def get_xai_model_and_explainer():
    print("--- Training XAI model (this should only run once) ---")
    np.random.seed(0)
    X = pd.DataFrame({'ev_adoption': np.random.randint(0, 100, 500), 'renewables_mix': np.random.randint(20, 100, 500), 'industrial_output': np.random.uniform(80, 120, 500)})
    y = 50 - (X['ev_adoption'] * 0.15) - ((X['renewables_mix'] - 20) * 0.25) + (X['industrial_output'] - 100) * 0.1 + np.random.normal(0,2,500)
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    return model, explainer


# --- CACHE ADDED ---
@st.cache_data
def ors_route(start_coords, end_coords, profile='foot-walking'):
    try:
        coords = ((start_coords[1], start_coords[0]), (end_coords[1], end_coords[0]))
        route = ors_client.directions(coords, profile=profile, format='geojson')
        return route
    except Exception as e:
        st.warning(f"ORS routing error: {e}")
        return None


# -------------------------
# UI STARTS HERE
# -------------------------

st.markdown('<div class="header-row"><div class="logo-circle">AI</div><div><div style="font-weight:800;font-size:20px">Advanced AI Sustainability Dashboard</div><div class="small-muted">Cerebras + OpenRouteService + OpenStreetMap</div></div></div>', unsafe_allow_html=True)

# --- "About / Help" button REMOVED ---
# cols = st.columns([3,1])
# with cols[1]:
#     st.write('')
#     if st.button('About / Help'):
#         st.info('This dashboard uses Cerebras for AI and OpenRouteService for routing. Add your keys to .streamlit/secrets.toml: ORS_API_KEY, OPENCHARGEMAP_API_KEY, CEREBRAS_API_KEY, CEREBRAS_ENDPOINT_URL')

# --- Main tabs UPDATED (Multimodal tab removed) ---
tabs = st.tabs([
    "🏙️ Global Urban Digital Twin",
    "💡 Energy & Carbon Forecast",
    "🏛️ XAI Policy Hub",
    "💰 Carbon & Policy Visuals",
    "🗺️ AI for Sustainable Transport"
])

# ---------------------------------------------------------
# TAB 1 — DIGITAL TWIN
# ---------------------------------------------------------
with tabs[0]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("🏙️ Global Urban Digital Twin — OSM + Folium")

    # --- EXPLANATION MOVED HERE ---
    with st.expander("ℹ️ What is this?"):
        st.markdown("""
        * **Definition:** A dynamic, digital replica of a physical city. It integrates real-time and static data to model the city's systems.
        * **Objective:** To visualize and analyze complex urban interactions in a single, map-based interface.
        * **Usecase:** An urban planner uses this tab to see where existing EV charging stations (`Real Data`) are located in relation to areas with high rooftop solar potential (`Simulated Data`). This helps them decide the best locations for new charging stations.
        * **Technology:**
            * **Folium (OpenStreetMap):** The core mapping library used to display the interactive map.
            * **OpenRouteService (ORS) Geocoding:** Converts a text location like "Delhi" into geographic coordinates (latitude, longitude) so the map knows where to center.
            * **OpenChargeMap API:** A real-world, crowd-sourced API that provides the locations of actual EV charging stations.
        """)

    city_col, action_col = st.columns([3,1])
    with city_col:
        city = st.text_input("Enter city or location", value=st.session_state.get('digital_twin_city','Delhi'))
    with action_col:
        st.write("") 
        st.write("")
        if st.button("Update View", key="update_twin"):
            ors_geocode.clear()
            get_ev_stations.clear()
            
            coords = ors_geocode(city)
            if coords:
                st.session_state['digital_twin_city'] = city
                st.session_state['digital_twin_coords'] = coords

    if 'digital_twin_coords' not in st.session_state:
        st.session_state['digital_twin_coords'] = ors_geocode('Delhi')
        st.session_state['digital_twin_city'] = 'Delhi'


    if 'digital_twin_coords' in st.session_state and st.session_state['digital_twin_coords']:
        lat, lon = st.session_state['digital_twin_coords']
        
        solar_df = get_dummy_buildings(lat, lon)
        ev_df = get_ev_stations(lat, lon)

        m = folium.Map(location=[lat, lon], zoom_start=12, tiles='OpenStreetMap')

        solar_layer = folium.FeatureGroup(name='Solar Potential (Simulated)')
        for _, building in solar_df.iterrows():
            color = "green" if building['solar_potential'] > 75 else "orange"
            folium.Rectangle(bounds=building['bounds'], color=color, fill=True, fill_color=color, fill_opacity=0.5, popup=f"Solar Potential: {building['solar_potential']} kWh/m²").add_to(solar_layer)
        solar_layer.add_to(m)

        ev_layer = folium.FeatureGroup(name='EV Stations (Real Data)')
        if not ev_df.empty:
            for _, station in ev_df.iterrows():
                folium.Marker(location=[station['lat'], station['lon']], popup=f"<b>{station['name']}</b>", icon=folium.Icon(color='blue', icon='bolt', prefix='fa')).add_to(ev_layer)
        ev_layer.add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, height=520, use_container_width=True)

        st.markdown("<div class='small-muted'>Tip: Use the layer control on the map to toggle simulated solar and EV station views.</div>", unsafe_allow_html=True)

        st.markdown("---")
        
        # --- NEW AI SUMMARY SECTION ---
        if st.button("🤖 Generate AI Map Summary"):
            ev_count = len(ev_df)
            avg_solar = solar_df['solar_potential'].mean() if not solar_df.empty else 0
            
            prompt = f"""You are an AI urban planning assistant. Summarize the sustainability map for {city}.
            We have identified:
            - {ev_count} EV stations (real data)
            - {len(solar_df)} buildings with an average simulated solar potential of {avg_solar:.0f}%

            Explain what this map shows and how a city planner would use this specific data.
            Please format your response in clear bullet points.
            """
            
            with st.spinner("Analyzing map data with Cerebras AI..."):
                summary = get_ai_response(prompt)
                st.info("AI Map Summary")
                st.markdown(summary)
        
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2 — ENERGY & CARBON FORECAST
# ---------------------------------------------------------
with tabs[1]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("💡 Energy Consumption & Carbon Forecast")

    # --- EXPLANATION MOVED HERE ---
    with st.expander("ℹ️ What is this?"):
        st.markdown("""
        * **Definition:** A time-series forecast that predicts future values (e.g., energy demand) based on past patterns.
        * **Objective:** To anticipate future trends and quantitatively test the potential impact of hypothetical policies.
        * **Usecase:** A policy analyst wants to see if a new "5% energy reduction" policy is enough to meet climate goals. They slide the "Policy Impact" slider to -5% and see how the forecast (the green line) changes compared to the original forecast (the dashed line).
        * **Technology (Prophet):**
            * **Prophet (by Meta):** A powerful forecasting library. It works by decomposing a time series into three main components:
            * **Formula (Simplified):** $y(t) = g(t) + s(t) + h(t) + \epsilon_t$
                * $g(t)$: **Trend** (the overall non-periodic, long-term direction, e.g., energy use is slowly increasing year-over-year).
                * $s(t)$: **Seasonality** (periodic changes, like higher energy use in summer (yearly) or on weekdays (weekly)).
                * $h(t)$: **Holidays** (the effects of specific, known events).
                * $\epsilon_t$: **Error** (random noise not captured by the model).
        """)
    
    hist_df, forecast_df = create_forecast()

    policy_impact = st.slider("Future Policy Impact (Target Reduction %)", -25, 25, 0, 5)
    adjusted_forecast = forecast_df.copy()
    last_hist = hist_df['ds'].max()
    future_mask = adjusted_forecast['ds'] > last_hist
    adjustment_factor = 1 - (policy_impact / 100)
    for col in ['yhat', 'yhat_upper', 'yhat_lower']:
        if col in adjusted_forecast.columns:
            adjusted_forecast.loc[future_mask, col] *= adjustment_factor

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df['ds'], y=hist_df['y'], mode='lines', name='Historical'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Original Forecast', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=adjusted_forecast['ds'], y=adjusted_forecast['yhat'], mode='lines', name='Policy-Adjusted', line=dict(width=3)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- AI SUMMARY MOVED TO FULL WIDTH ---
    if st.button("🤖 Generate AI Summary of Forecast"):
        peak_info = forecast_df['ds'].iloc[forecast_df['yhat'].idxmax()].strftime('%B') if 'yhat' in forecast_df else 'N/A'
        original_end = float(forecast_df['yhat'].iloc[-1]) if 'yhat' in forecast_df else 0
        adjusted_end = float(adjusted_forecast['yhat'].iloc[-1]) if 'yhat' in adjusted_forecast else 0
        prompt = f"""Summarize the energy forecast.
        - Peak season: {peak_info}
        - Original end-of-period forecast: {original_end:.2f}
        - Adjusted end-of-period forecast (after {policy_impact}% policy): {adjusted_end:.2f}
        
        Explain the implications and suggest 3 practical policy levers.
        Please format your response in clear bullet points.
        """
        with st.spinner("Analyzing with Cerebras AI..."):
            summary = get_ai_response(prompt)
            st.info("AI Forecast Summary")
            st.markdown(summary)
                
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3 — XAI POLICY HUB
# ---------------------------------------------------------
with tabs[2]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("🏛️ Explainable AI (XAI) Climate Policy Hub")

    # --- EXPLANATION MOVED HERE ---
    with st.expander("ℹ️ What is this?"):
        st.markdown("""
        * **Definition:** "Explainable AI" (XAI) is a set of tools and methods that help humans understand how an AI model arrives at its predictions. This tab moves beyond "the AI said 25.1" to "the AI said 25.1 *because* EV adoption was high."
        * **Objective:** To build trust and transparency in AI-driven policy, allowing users to understand the *why* behind a prediction, not just the *what*.
        * **Usecase:** A user slides "EV Adoption" to 80% and "Renewables" to 30%. The waterfall plot shows that the high EV adoption had a large *negative* impact on the score (e.g., higher grid strain), while renewables had a small *positive* impact. This shows the user that just focusing on EVs isn't enough; the grid must be cleaner too.
        * **Technology (SHAP):**
            * **SHAP (SHapley Additive exPlanations):** The leading XAI method. It's based on "Shapley values," a concept from game theory.
            * **Concept:** It treats each feature (e.g., "EV Adoption") as a "player" in a "game" where the "payout" is the model's prediction. The SHAP value for a feature is its average marginal contribution to the prediction across all possible combinations of other features. The waterfall plot visualizes these values, showing how each feature "pushes" the prediction from a "base value" (the average prediction) to the final prediction.
        """)
    
    model, explainer = get_xai_model_and_explainer()
    
    col1, col2 = st.columns(2)
    with col1:
        ev_adoption = st.slider("EV Adoption Rate (%)", 0, 100, 30, key="m_ev")
        renewables_mix = st.slider("Renewable Energy in Grid (%)", 20, 100, 40, key="m_ren")
        
        scenario = pd.DataFrame([{'ev_adoption': ev_adoption, 'renewables_mix': renewables_mix, 'industrial_output': 100}])
        shap_values = explainer.shap_values(scenario)
        prediction = model.predict(scenario)[0]
        
        st.metric("Projected Scenario Score", f"{prediction:.1f}")
    with col2:
        st.subheader("Why this prediction?")
        fig, ax = plt.subplots(figsize=(6,3))
        try:
            shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=scenario.iloc[0]), show=False)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.write("Unable to render SHAP waterfall — falling back to explanation text.")
            st.write(f"SHAP Error: {e}")
            text = f"Base: {explainer.expected_value:.2f}. EV Adoption contribution: {shap_values[0][0]:.2f}. Renewables contribution: {shap_values[0][1]:.2f}."
            st.write(text)

    xai_context = f"You are a friendly AI explaining policy sliders. Current: EV {ev_adoption}%, Renewables {renewables_mix}% -> projection {prediction:.1f}. Explain in simple terms using bullet points."
    with st.expander("Ask the AI about this scenario"):
        q = st.text_input("Question to AI", key="xai_q")
        if st.button("Ask AI", key="ask_xai"):
            prompt = f"{xai_context}\nUser question: {q or 'Explain the results simply.'}"
            answer = get_ai_response(prompt)
            st.markdown(answer)
            
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 4 — CARBON & POLICY VISUALS (This was Tab 5)
# ---------------------------------------------------------
with tabs[3]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("💰 Carbon & Policy Network Visuals")

    # --- EXPLANATION MOVED HERE ---
    with st.expander("ℹ️ What is this?"):
        st.markdown("""
        * **Definition:** Advanced data visualizations designed to show complex systems, flows, and relationships.
        * **Objective:** To simplify complex sustainability concepts (like carbon credit flows or policy interactions) into an intuitive visual format.
        * **Usecase:**
            * **Sankey Diagram:** A user wants to know where their $100M "Green Investment" goes. The Sankey chart shows them that $60M (60%) goes to "Renewable Energy" and $40M (40%) to "Reforestation." It then shows the *outcomes* of that investment, like how many "Credits Generated" each stream produced.
            * **Network Graph:** A policymaker is confused about indirect effects. This graph shows them that "EV Adoption Policy" (a node) has a direct relationship (an edge) to "CO2 Emissions" and "Fossil Fuel Imports," helping them understand the interconnectedness of their decisions.
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Carbon Credit Flow")
        inv = st.slider("Investment in Green Projects ($M)", 10, 200, 100)
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20, label=["Green Investments", "Reforestation", "Renewable Energy", "Credits Generated", "Credits Used (Internal)", "Credits Traded", "Credits Retired"]),
            link=dict(source=[0,0,1,2,3,3,5], target=[1,2,3,3,4,5,6], value=[inv*0.4, inv*0.6, inv*0.4*1.2, inv*0.6*1.5, inv*0.8, inv*0.2, inv*0.2*0.9])
        )])
        fig_sankey.update_layout(margin=dict(l=0, r=0, t=25, b=25))
        st.plotly_chart(fig_sankey, use_container_width=True)
    with col2:
        st.subheader("Policy Impact Network")
        node_x, node_y = [0.1,0.1,0.5,0.5,0.9,0.9], [0.8,0.2,0.9,0.1,0.8,0.2]
        labels = ["EV Adoption Policy","Fossil Fuel Imports","Renewable Energy Policy","Grid Demand","Air Quality","CO₂ Emissions"]
        fig_net = go.Figure()
        edges = [(0,1),(0,5),(2,3),(2,5),(1,5),(3,5)]
        edge_x, edge_y = [], []
        for edge in edges:
            edge_x.extend([node_x[edge[0]], node_x[edge[1]], None]); edge_y.extend([node_y[edge[0]], node_y[edge[1]], None])
        fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1.5, color='#888')))
        fig_net.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=labels, textposition='top center', marker=dict(size=25)))
        fig_net.update_layout(showlegend=False, xaxis_visible=False, yaxis_visible=False, margin=dict(l=0, r=0, t=25, b=25))
        st.plotly_chart(fig_net, use_container_width=True)

    carbon_viz_context = f"You are an AI explaining a Sankey for a ${inv}M investment and a small policy network. Explain flows and node relationships in bullet points."
    with st.expander("Ask AI to explain these visuals"):
        q = st.text_input("Question", key="carbon_q")
        if st.button("Explain visuals", key="carbon_ask"):
            prompt = f"{carbon_viz_context}\nUser question: {q or 'Explain the Sankey and network.'}"
            ans = get_ai_response(prompt)
            st.markdown(ans)
            
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 5 — AI FOR SUSTAINABLE TRANSPORT (This was Tab 6)
# ---------------------------------------------------------
with tabs[4]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("🗺️ AI for Sustainable Transport — OpenRouteService")

    # --- EXPLANATION MOVED HERE ---
    with st.expander("ℹ️ What is this?"):
        st.markdown("""
        * **Definition:** A hybrid approach combining AI-driven advice with a specific mapping engine to plan sustainable travel.
        * **Objective:** To nudge users towards more sustainable transport options (walking, cycling, public transit) by providing both a high-level "why" (from the AI) and a detailed "how" (from the map).
        * **Usecase:** A user plans a trip. The AI (Cerebras) analyzes the start and end points and suggests "The Delhi Metro is the most sustainable option for this route, as it avoids traffic and has low emissions." Simultaneously, the map (ORS) shows them the visual route for their *selected* mode (e.g., "foot-walking"), allowing them to compare the AI's advice with a concrete plan.
        * **Technology:**
            * **OpenRouteService (ORS) API:** This is a powerful routing engine built on **OpenStreetMap (OSM)** data. We send it a start coordinate, end coordinate, and a profile (like `cycling-regular`), and it calculates the most efficient route.
            * **Cerebras API:** The AI provides the high-level, context-aware sustainable travel *advice* that the ORS engine cannot.
        """)

    col1, col2 = st.columns(2)
    with col1:
        # --- SESSION STATE FIX ---
        # Clear the saved plan if the user starts typing a new location
        def clear_plan():
            if 'sustainable_plan' in st.session_state:
                del st.session_state['sustainable_plan']
        
        start_location = st.text_input("📍 Start Location", "India Gate, Delhi", on_change=clear_plan)
    with col2:
        end_location = st.text_input("🏁 Destination", "Qutub Minar, Delhi", on_change=clear_plan)

    mode = st.selectbox("Preferred mode (for visual route)", ['foot-walking','cycling-regular','driving-car'], index=0, on_change=clear_plan)

    if st.button("🌱 Generate Sustainable Plan"):
        if not start_location or not end_location:
            st.warning("Please provide both start and end.")
        else:
            with st.spinner("Planning route with ORS and AI..."):
                s_coords = ors_geocode(start_location)
                e_coords = ors_geocode(end_location)
                
                if not s_coords or not e_coords:
                    st.error("Could not geocode one of the locations. Check the error message above.")
                else:
                    ai_prompt = f"Create a step-by-step sustainable travel plan from {start_location} to {end_location} prioritizing public transport (like metro or bus) and walking. If the user's selected mode '{mode}' is not sustainable (like 'driving-car'), suggest it as a last resort and explain why the other options are better. Include distances and brief reasons why the sustainable options are better. Use bullet points."
                    ai_plan = get_ai_response(ai_prompt)
                    route = ors_route(s_coords, e_coords, profile=mode)
                    
                    # --- SESSION STATE FIX: Save results ---
                    st.session_state['sustainable_plan'] = ai_plan
                    st.session_state['sustainable_route'] = route
                    st.session_state['sustainable_s_coords'] = s_coords
                    st.session_state['sustainable_e_coords'] = e_coords
                    st.session_state['sustainable_mode'] = mode

    # --- SESSION STATE FIX: Display results if they exist ---
    if 'sustainable_plan' in st.session_state:
        st.subheader("📝 AI-Generated Sustainable Plan")
        st.markdown(st.session_state['sustainable_plan'])

        st.subheader(f"Visual Route for '{st.session_state['sustainable_mode']}'")
        
        route = st.session_state['sustainable_route']
        s_coords = st.session_state['sustainable_s_coords']
        e_coords = st.session_state['sustainable_e_coords']

        if route:
            geom = route['features'][0]['geometry']
            coords = geom['coordinates']
            route_latlon = [(c[1], c[0]) for c in coords]
            map_center = [(s_coords[0]+e_coords[0])/2, (s_coords[1]+e_coords[1])/2]
            
            m = folium.Map(location=map_center, zoom_start=13, tiles='OpenStreetMap')
            folium.PolyLine(route_latlon, color='purple', weight=6, opacity=0.8).add_to(m)
            folium.Marker(location=[s_coords[0], s_coords[1]], tooltip='Start', icon=folium.Icon(color='green')).add_to(m)
            folium.Marker(location=[e_coords[0], e_coords[1]], tooltip='Destination', icon=folium.Icon(color='red')).add_to(m)
            
            m.fit_bounds(folium.PolyLine(route_latlon).get_bounds())
            
            st_folium(m, height=480, use_container_width=True)
        else:
            st.warning(f"Could not retrieve a visual route for '{st.session_state['sustainable_mode']}'; AI plan is shown above.")
            
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown('--- \n <small class="small-muted">Built with Cerebras AI + OpenRouteService + OpenStreetMap · Demo UI</small>', unsafe_allow_html=True)

# End of file
