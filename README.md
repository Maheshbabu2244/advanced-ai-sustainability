Advanced AI for Sustainability Dashboard
An interactive web application demonstrating the power of predictive, generative, and explainable AI to tackle real-world sustainability challenges. This dashboard, built with Streamlit, integrates multiple AI models and live APIs to provide actionable insights into urban sustainability, energy consumption, climate policy, and sustainable transport.

🚀 Key Features
This dashboard is organized into several modules, each showcasing a unique application of AI for sustainability:

🏙️ Global Urban Digital Twin: A realistic, folium-based map that visualizes sustainability metrics for any city worldwide. It integrates live API data for EV charging stations and simulates building solar potential.

💡 Energy & Carbon Forecast: Uses the Prophet time-series model to predict future energy consumption. A generative AI agent (Gemini) provides easy-to-understand summaries of the complex forecast data.

🏛️ Explainable AI (XAI) Policy Hub: Allows users to simulate climate policy scenarios (e.g., EV adoption, renewable energy mix) and uses SHAP (SHapley Additive exPlanations) to provide a transparent, visual explanation of why the AI made its emission predictions.

📸 Multimodal Analysis: Leverages the Gemini Vision model to analyze user-uploaded images. It can classify waste items and provide recycling instructions or analyze a store receipt to suggest more sustainable product alternatives.

💰 Carbon & Policy Visuals: Simplifies complex financial and policy concepts using interactive Sankey and Network diagrams, with a dedicated AI agent to explain the charts to beginners.

🤖 AI for Sustainable Transport: Acts as an intelligent travel planner. The AI generates a detailed, step-by-step travel itinerary between two locations that prioritizes public transport and walking, providing a sustainable alternative to standard navigation.

🛠️ Tech Stack & APIs
Frontend: Streamlit

Data Science & Machine Learning: Pandas, NumPy, Prophet (by Meta), Scikit-learn, SHAP

Mapping & Visualization: Folium, Plotly, Matplotlib

AI Models: Google Gemini 1.5 Flash (Generative Text, Multimodal Vision)

External APIs:

Google Maps API (Geocoding, Directions)

OpenChargeMap API (EV Charging Stations)

⚙️ Setup and Installation
Follow these steps to run the dashboard locally.

1. Clone the Repository
git clone [https://github.com/your-username/ai-sustainability-dashboard.git](https://github.com/your-username/ai-sustainability-dashboard.git)
cd ai-sustainability-dashboard

2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

# For MacOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate

3. Install Dependencies
Install all required libraries using the requirements.txt file.

pip install -r requirements.txt

4. Configure API Keys
This project requires API keys from Google, and OpenChargeMap.

Create a folder named .streamlit in the root of the project directory.

Inside this folder, create a file named secrets.toml.

Add your API keys to the file as shown below:

# .streamlit/secrets.toml

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"
OPENCHARGEMAP_API_KEY = "YOUR_OPENCHARGEMAP_API_KEY"

▶️ Running the Application
Once the setup is complete, run the following command in your terminal from the project's root directory:

streamlit run app.py

The application will open in a new tab in your web browser.
