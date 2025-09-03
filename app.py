import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import googlemaps
import polyline
from datetime import datetime
import google.generativeai as genai
from PIL import Image
import random
import json
import re
import requests

# Advanced ML & XAI libraries
from prophet import Prophet
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Folium for realistic maps
import folium
from streamlit_folium import st_folium


# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced AI for Sustainability Dashboard",
    page_icon="🌍",
    layout="wide",
)

# --- API Client Initialization ---
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    gmaps_api_key = st.secrets["GOOGLE_MAPS_API_KEY"]
    openchargemap_api_key = st.secrets["OPENCHARGEMAP_API_KEY"]
    genai.configure(api_key=gemini_api_key)
    gemini_model_vision = genai.GenerativeModel('gemini-1.5-flash')
    gemini_model_text = genai.GenerativeModel('gemini-1.5-flash')
    gmaps = googlemaps.Client(key=gmaps_api_key)
except (KeyError, FileNotFoundError):
    st.error("API keys not found. Please ensure GEMINI_API_KEY, GOOGLE_MAPS_API_KEY, and OPENCHARGEMAP_API_KEY are in your `.streamlit/secrets.toml` file.")
    st.stop()

# --- Session State Initialization for Stability ---
if 'digital_twin_location' not in st.session_state:
    st.session_state.digital_twin_location = {"name": "Delhi", "lat": 28.6139, "lon": 77.2090}
if 'digital_twin_data' not in st.session_state:
    st.session_state.digital_twin_data = None
if 'navigator_data' not in st.session_state:
    st.session_state.navigator_data = None
if 'multimodal_result' not in st.session_state:
    st.session_state.multimodal_result = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'xai_messages' not in st.session_state:
    st.session_state.xai_messages = []
if 'carbon_viz_messages' not in st.session_state:
    st.session_state.carbon_viz_messages = []


# --- Helper & Caching Functions ---
def get_ai_response(prompt):
    """Generic function to call the Gemini API for text."""
    try:
        response = gemini_model_text.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate response. Error: {e}"

def clean_html(raw_html):
  """Removes HTML tags from a string."""
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def ai_agent_module(agent_name, header, context_prompt):
    """Creates a self-contained, expandable AI agent for any tab."""
    with st.expander(f"💬 {header}"):
        message_history_key = f"{agent_name}_messages"
        for msg in st.session_state[message_history_key]:
            st.chat_message(msg["role"]).write(msg["content"])
        if prompt := st.chat_input("Ask a question about this module...", key=f"chat_{agent_name}"):
            st.session_state[message_history_key].append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            full_prompt = f"{context_prompt}\n\nUser question: {prompt}"
            with st.spinner("🤖 Thinking..."):
                response = get_ai_response(full_prompt)
                st.session_state[message_history_key].append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)

@st.cache_data(ttl=3600)
def get_ev_stations(lat, lon, distance=10, max_results=200):
    """Fetches real EV charging stations from OpenChargeMap API."""
    try:
        url = "https://api.openchargemap.io/v3/poi"
        params = {"output": "json", "latitude": lat, "longitude": lon, "distance": distance, "distanceunit": "km", "maxresults": max_results, "key": openchargemap_api_key}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        points = []
        for charge_point in data:
            operator_info = charge_point.get("OperatorInfo")
            operator_name = operator_info.get("Title", "Unknown") if operator_info else "Unknown"
            points.append({
                "lon": charge_point["AddressInfo"]["Longitude"],
                "lat": charge_point["AddressInfo"]["Latitude"],
                "name": charge_point["AddressInfo"].get("Title", "N/A"),
                "operator": operator_name
            })
        return pd.DataFrame(points)
    except Exception as e:
        st.warning(f"Could not fetch EV station data. Error: {e}")
        return pd.DataFrame()

def get_dummy_building_data(center_lat, center_lon):
    """Generates dummy 2D building data around a specific location."""
    buildings = []
    for i in range(75):
        lat_offset, lon_offset = (np.random.randn() * 0.05, np.random.randn() * 0.05)
        lat_center, lon_center = center_lat + lat_offset, center_lon + lon_offset
        size = np.random.uniform(0.0005, 0.0015)
        bounds = [[lat_center - size, lon_center - size], [lat_center + size, lon_center + size]]
        buildings.append({'bounds': bounds, 'solar_potential': random.randint(40, 100)})
    return pd.DataFrame(buildings)

@st.cache_data
def create_prophet_forecast(periods=365):
    """Generates and forecasts a synthetic energy usage dataset."""
    ds = pd.date_range(start='2023-01-01', periods=365*2, freq='D')
    baseline = np.linspace(50, 80, len(ds))
    yearly_seasonality = 15 * np.sin(2 * np.pi * ds.dayofyear / 365.25)
    weekly_seasonality = 5 * np.sin(2 * np.pi * ds.dayofweek / 7)
    noise = np.random.normal(0, 5, len(ds))
    y = baseline + yearly_seasonality + weekly_seasonality + noise
    df = pd.DataFrame({'ds': ds, 'y': y})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return df, forecast

@st.cache_data
def get_shap_explanation(ev_adoption, renewables_mix):
    """Trains a model and returns SHAP values for a given policy scenario."""
    np.random.seed(0)
    X = pd.DataFrame({'ev_adoption': np.random.randint(0, 100, 500), 'renewables_mix': np.random.randint(20, 100, 500), 'industrial_output': np.random.uniform(80, 120, 500)})
    y = 50 - (X['ev_adoption'] * 0.15) - ((X['renewables_mix'] - 20) * 0.25) + (X['industrial_output'] - 100) * 0.1 + np.random.normal(0,2,500)
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    scenario = pd.DataFrame([{'ev_adoption': ev_adoption, 'renewables_mix': renewables_mix, 'industrial_output': 100}])
    shap_values = explainer.shap_values(scenario)
    return explainer, shap_values, scenario, model.predict(scenario)[0]

def get_ai_multimodal_response(prompt, image):
    """Calls Gemini with both text and image."""
    try:
        response = gemini_model_vision.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Could not process the image. Error: {e}"

# --- MAIN APP LAYOUT ---
st.title("🌍 Advanced AI for Sustainability Dashboard")

tabs = st.tabs([
    "🏙️ Global Urban Digital Twin",
    "💡 Energy & Carbon Forecast",
    "🏛️ XAI Policy Hub",
    "📸 Multimodal Analysis",
    "💰 Carbon & Policy Visuals",
    "🤖 AI for Sustainable Transport"
])

# --- TAB 1: GLOBAL URBAN DIGITAL TWIN ---
with tabs[0]:
    st.header("🏙️ Global Urban Digital Twin")
    st.markdown("Visualize sustainability data for any city using a realistic map interface.")

    col1, col2 = st.columns([2, 1])
    with col1:
        location_query = st.text_input("Enter a city or location:", value=st.session_state.digital_twin_location["name"])
    with col2:
        st.write("")
        if st.button("Update View", use_container_width=True):
            with st.spinner(f"Geocoding {location_query}..."):
                try:
                    geocode_result = gmaps.geocode(location_query)
                    if geocode_result:
                        loc = geocode_result[0]['geometry']['location']
                        st.session_state.digital_twin_location = {"name": location_query, "lat": loc['lat'], "lon": loc['lng']}
                        st.session_state.digital_twin_data = None
                    else:
                        st.error("Could not find location.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    if st.session_state.digital_twin_data is None:
        current_location = st.session_state.digital_twin_location
        lat, lon = current_location["lat"], current_location["lon"]
        with st.spinner(f"Fetching data for {current_location['name']}..."):
            st.session_state.digital_twin_data = {
                "solar_df": get_dummy_building_data(lat, lon),
                "ev_df": get_ev_stations(lat, lon, distance=20)
            }

    if st.session_state.digital_twin_data:
        current_location = st.session_state.digital_twin_location
        lat, lon = current_location["lat"], current_location["lon"]
        solar_df = st.session_state.digital_twin_data["solar_df"]
        ev_df = st.session_state.digital_twin_data["ev_df"]

        m = folium.Map(location=[lat, lon], zoom_start=12, tiles="CartoDB positron")
        solar_layer = folium.FeatureGroup(name='Solar Potential (Simulated)')
        for _, building in solar_df.iterrows():
            color = "green" if building['solar_potential'] > 75 else "orange"
            folium.Rectangle(bounds=building['bounds'], color=color, fill=True, fill_color=color, fill_opacity=0.5, popup=f"Solar Potential: {building['solar_potential']} kWh/m²").add_to(solar_layer)
        solar_layer.add_to(m)

        ev_layer = folium.FeatureGroup(name='EV Stations (Real Data)')
        if not ev_df.empty:
            for _, station in ev_df.iterrows():
                folium.Marker(location=[station['lat'], station['lon']], popup=f"<b>{station['name']}</b><br>Operator: {station['operator']}", icon=folium.Icon(color='blue', icon='charging-station', prefix='fa')).add_to(ev_layer)
        ev_layer.add_to(m)
        
        folium.LayerControl().add_to(m)
        st_folium(m, width=725, height=500, key="digital_twin_map")
        st.info(f"Displaying data for **{current_location['name']}**. Use the layer control icon in the top-right of the map to toggle views.")

# --- TAB 2: ENERGY & CARBON FORECAST ---
with tabs[1]:
    st.header("💡 Energy Consumption & Carbon Forecast")
    hist_df, forecast_df = create_prophet_forecast()
    st.subheader("Interactive AI Analysis & Adjustment")
    policy_impact = st.slider("Future Policy Impact (Target Reduction %)", -25, 25, 0, 5)
    adjusted_forecast_df = forecast_df.copy()
    last_hist_date = hist_df['ds'].max()
    future_mask = adjusted_forecast_df['ds'] > last_hist_date
    adjustment_factor = 1 - (policy_impact / 100)
    for col in ['yhat', 'yhat_upper', 'yhat_lower']:
        adjusted_forecast_df.loc[future_mask, col] *= adjustment_factor
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df['ds'], y=hist_df['y'], mode='lines', name='Historical Usage'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Original Forecast', line=dict(color='orange', dash='dash')))
    fig.add_trace(go.Scatter(x=adjusted_forecast_df['ds'], y=adjusted_forecast_df['yhat'], mode='lines', name='Policy-Adjusted Forecast', line=dict(color='green', width=3)))
    fig.add_trace(go.Scatter(x=adjusted_forecast_df['ds'], y=adjusted_forecast_df['yhat_upper'], fill=None, mode='lines', line=dict(color='rgba(0,128,0,0.2)')))
    fig.add_trace(go.Scatter(x=adjusted_forecast_df['ds'], y=adjusted_forecast_df['yhat_lower'], fill='tonexty', mode='lines', line=dict(color='rgba(0,128,0,0.2)')))
    fig.update_layout(title="Energy Consumption Forecast (GWh)", xaxis_title="Date", yaxis_title="Energy (GWh)", legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig, use_container_width=True)
    if st.button("🤖 Generate AI Summary of Forecast"):
        with st.spinner("AI is analyzing the forecast..."):
            original_end_value, adjusted_end_value = forecast_df['yhat'].iloc[-1], adjusted_forecast_df['yhat'].iloc[-1]
            peak_season = forecast_df.loc[forecast_df['yearly'].idxmax()]['ds'].strftime('%B')
            prompt = f"Summarize a time-series forecast. Context: Peaks are in {peak_season}. Trend is increasing. Original forecast ends at {original_end_value:.2f} GWh. User simulated a {policy_impact}% policy impact, resulting in an adjusted forecast of {adjusted_end_value:.2f} GWh. Explain the trend, policy's effect, and final difference."
            response = get_ai_response(prompt)
            st.info("💡 AI Forecast Analysis:", icon="🤖")
            st.markdown(response)

# --- TAB 3: XAI POLICY HUB ---
with tabs[2]:
    st.header("🏛️ Explainable AI (XAI) Climate Policy Hub")
    st.markdown("Simulate policy scenarios and understand *why* the AI predicts a certain outcome using **SHAP**.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Policy Levers")
        ev_adoption = st.slider("EV Adoption Rate (%)", 0, 100, 30, key="m7_ev")
        renewables_mix = st.slider("Renewable Energy in Grid (%)", 20, 100, 40, key="m7_renew")
        explainer, shap_values, scenario, prediction = get_shap_explanation(ev_adoption, renewables_mix)
        st.metric("Projected 2050 Emissions (Gt CO₂e)", f"{prediction:.1f}")
    with col2:
        st.subheader("Why this prediction?")
        st.markdown("This SHAP plot shows how each factor *pushed* the prediction away from the average.")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=scenario.iloc[0]), show=False)
        st.pyplot(fig)
    
    xai_context = f"""
    You are a friendly AI guide explaining climate policy simulations to a beginner.
    The user is looking at a SHAP (SHapley Additive exPlanations) plot.
    The current scenario is: EV Adoption at {ev_adoption}% and Renewable Energy at {renewables_mix}%, resulting in a projection of {prediction:.1f} Gt CO2e.
    Explain the sliders and the SHAP plot in simple terms. For the plot, explain that the 'base value' is the average prediction, and colored bars show how each policy pushes the final prediction up (red, increases emissions) or down (blue, decreases emissions).
    """
    ai_agent_module("xai", "Ask the AI Agent to Explain This Hub", xai_context)

# --- TAB 4: MULTIMODAL ANALYSIS ---
with tabs[3]:
    st.header("📸 Multimodal AI Analysis")
    analysis_type = st.radio("What are you analyzing?", ("♻️ Waste Item", "🧾 Store Receipt"), horizontal=True, key="multimodal_radio")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="multimodal_uploader")
    if uploaded_file is not None and st.session_state.uploaded_image != uploaded_file:
        st.session_state.uploaded_image = uploaded_file
        st.session_state.multimodal_result = None
    if st.session_state.uploaded_image is not None:
        image = Image.open(st.session_state.uploaded_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        if st.button("Analyze Image", key="analyze_image_btn"):
            with st.spinner("🤖 AI is thinking..."):
                prompt = "You are a waste classification expert. Analyze this image. Identify the material, its recyclability, and provide disposal instructions." if analysis_type == "♻️ Waste Item" else "You are a sustainability analyst. Analyze this receipt. Identify 2-3 products, comment on their likely environmental footprint, and suggest sustainable alternatives."
                st.session_state.multimodal_result = get_ai_multimodal_response(prompt, image)
    if st.session_state.multimodal_result:
        st.info("💡 AI Analysis:", icon="🤖")
        st.markdown(st.session_state.multimodal_result)

# --- TAB 5: CARBON & POLICY VISUALS ---
with tabs[4]:
    st.header("💰 Carbon & Policy Network Visuals")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Carbon Credit Flow (Sankey Diagram)")
        inv = st.slider("Investment in Green Projects ($M)", 10, 200, 100, key="investment_slider")
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20, label=["Green Investments", "Reforestation", "Renewable Energy", "Credits Generated", "Credits Used (Internal)", "Credits Traded", "Credits Retired"]),
            link=dict(source=[0, 0, 1, 2, 3, 3, 5], target=[1, 2, 3, 3, 4, 5, 6], value=[inv*0.4, inv*0.6, inv*0.4*1.2, inv*0.6*1.5, inv*0.8, inv*0.2, inv*0.2*0.9]))])
        st.plotly_chart(fig_sankey, use_container_width=True)
    with col2:
        st.subheader("Policy Impact Network Graph")
        node_x, node_y = [0.1, 0.1, 0.5, 0.5, 0.9, 0.9], [0.8, 0.2, 0.9, 0.1, 0.8, 0.2]
        labels = ["EV Adoption Policy", "Fossil Fuel Imports", "Renewable Energy Policy", "Grid Demand", "Air Quality", "CO₂ Emissions"]
        fig_net = go.Figure()
        edges = [(0,1), (0,5), (2,3), (2,5), (1,5), (3,5)]
        edge_x, edge_y = [], []
        for edge in edges:
            edge_x.extend([node_x[edge[0]], node_x[edge[1]], None]); edge_y.extend([node_y[edge[0]], node_y[edge[1]], None])
        fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1.5, color='#888')))
        fig_net.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=labels, textposition="top center", marker=dict(color='skyblue', size=25)))
        fig_net.update_layout(showlegend=False, xaxis_visible=False, yaxis_visible=False)
        st.plotly_chart(fig_net, use_container_width=True)
    carbon_viz_context = f"""
    You are an AI assistant explaining financial and policy visuals to a beginner.
    The user is viewing two charts: a Sankey diagram for a ${inv}M investment in carbon credits, and a policy network graph.
    Explain the Sankey as a flow chart where line width shows quantity.
    Explain the network graph as nodes (concepts) connected by lines (influences), e.g., 'EV Adoption Policy' impacts 'Fossil Fuel Imports'.
    """
    ai_agent_module("carbon_viz", "Ask the AI Agent to Explain These Visuals", carbon_viz_context)

# --- TAB 6: AI FOR SUSTAINABLE TRANSPORT (STABLE & REFINED) ---
with tabs[5]:
    st.header("🤖 AI for Sustainable Transport")
    st.markdown("Let the AI act as a sustainable travel planner. It will create a route prioritizing public transport and walking.")

    col1, col2 = st.columns(2)
    with col1:
        start_location = st.text_input("📍 Enter Start Location", "India Gate, Delhi", key="nav_start")
    with col2:
        end_location = st.text_input("🏁 Enter Destination", "Qutub Minar, Delhi", key="nav_end")

    if st.button("🌱 Generate Sustainable Plan", use_container_width=True, key="nav_get_plan"):
        if not start_location or not end_location:
            st.warning("Please provide both start and end locations.")
            st.session_state.navigator_data = None
        else:
            with st.spinner("AI is planning your sustainable route..."):
                try:
                    # AI part: Generate the textual plan
                    prompt = f"""
                    Act as an expert AI sustainable travel planner. Your goal is to create a detailed, step-by-step travel plan from "{start_location}" to "{end_location}".
                    Prioritize using public transportation (like metro and buses) and walking. Avoid private cars or taxis.
                    
                    Your output should be a clear, numbered list in Markdown. For each step, describe the action, mention the mode of transport, specific line/bus numbers if available, and approximate duration or distance.
                    Start with a brief summary of why this route is sustainable.
                    """
                    ai_plan = get_ai_response(prompt)
                    
                    # Google Maps part: Get a visual route for confirmation
                    directions_result = gmaps.directions(start_location, end_location, mode="transit")
                    
                    if directions_result:
                        st.session_state.navigator_data = {
                            "plan": ai_plan,
                            "directions": directions_result
                        }
                    else: # Fallback if transit fails
                        st.session_state.navigator_data = { "plan": ai_plan, "directions": None }

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.session_state.navigator_data = None

    if st.session_state.get('navigator_data'):
        data = st.session_state.navigator_data
        
        st.subheader("📝 AI-Generated Travel Plan")
        st.markdown(data['plan'])
        
        if data['directions']:
            st.subheader("🗺️ Visual Route Overview (Public Transit)")
            directions = data['directions']
            leg = directions[0]["legs"][0]
            
            overview_polyline_str = directions[0]['overview_polyline']['points']
            route_coords = polyline.decode(overview_polyline_str)
            
            start_coords = (leg['start_location']['lat'], leg['start_location']['lng'])
            end_coords = (leg['end_location']['lat'], leg['end_location']['lng'])

            map_center = [(start_coords[0] + end_coords[0]) / 2, (start_coords[1] + end_coords[1]) / 2]
            m = folium.Map(location=map_center, zoom_start=12, tiles="OpenStreetMap")

            folium.PolyLine(route_coords, color="purple", weight=6, opacity=0.8).add_to(m)
            folium.Marker(location=start_coords, tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
            folium.Marker(location=end_coords, tooltip="Destination", icon=folium.Icon(color="red")).add_to(m)
            
            st_folium(m, width=700, height=500, key="folium_map_sustainable")
        else:
            st.warning("Could not generate a visual map for the transit route, but the AI plan is provided above.")

