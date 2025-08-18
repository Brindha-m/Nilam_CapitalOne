import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from simple_model import SimpleCropRecommendationModel
import warnings
warnings.filterwarnings('ignore')

# Page configuration - commented out for main.py integration
# st.set_page_config(
#     page_title="Nilam Sense - Smart Crop Recommendation System",
#     page_icon="🌱",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.8rem 2.5rem;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Custom button styles for different actions */
    .predict-btn > button {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        border: 2px solid #4CAF50;
    }
    
    .predict-btn > button:hover {
        background: linear-gradient(135deg, #45a049 0%, #4CAF50 100%);
        border-color: #45a049;
    }
    
    .clear-btn > button {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        border: 2px solid #dc3545;
        padding: 0.6rem 1.5rem;
        font-size: 0.9rem;
    }
    
    .clear-btn > button:hover {
        background: linear-gradient(135deg, #c82333 0%, #dc3545 100%);
        border-color: #c82333;
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    /* Dark theme improvements */
    .stPlotlyChart {
        background-color: #1e1e1e !important;
    }
    
    /* Better text contrast */
    .stMarkdown {
        color: #333;
    }
    
    /* Enhanced metric cards */
    .metric-card h2 {
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-card h4 {
        color: #666;
        margin-bottom: 0.5rem;
    }
    
    /* Make selectbox labels black */
    .stSelectbox label,
    div[data-baseweb="select"] label,
    [data-testid="stSelectbox"] label {
        color: black !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Force black color for all possible label selectors */
    label,
    .stSelectbox label,
    div[data-baseweb="select"] label,
    [data-testid="stSelectbox"] label,
    .stSelectbox div label,
    div[data-baseweb="select"] div label,
    [data-testid="stSelectbox"] div label,
    .stSelectbox div div label,
    div[data-baseweb="select"] div div label,
    [data-testid="stSelectbox"] div div label,
    .stSelectbox div div div label,
    div[data-baseweb="select"] div div div label,
    [data-testid="stSelectbox"] div div div label {
        color: black !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        text-shadow: none !important;
        opacity: 1 !important;
    }
    
    /* Force black color for all label elements */
    * label {
        color: black !important;
        font-weight: 600 !important;
        text-shadow: none !important;
        opacity: 1 !important;
    }
    
    /* Ultra comprehensive black label styling */
    .stSelectbox > div > label,
    div[data-baseweb="select"] > div > label,
    [data-testid="stSelectbox"] > div > label,
    .stSelectbox > div > div > label,
    div[data-baseweb="select"] > div > div > label,
    [data-testid="stSelectbox"] > div > div > label,
    .stSelectbox > div > div > div > label,
    div[data-baseweb="select"] > div > div > div > label,
    [data-testid="stSelectbox"] > div > div > div > label {
        color: black !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        text-shadow: none !important;
        opacity: 1 !important;
    }
    
    /* Force black for all text elements that might be labels */
    .stSelectbox span,
    .stSelectbox div span,
    .stSelectbox div div span,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div span,
    div[data-baseweb="select"] div div span,
    [data-testid="stSelectbox"] span,
    [data-testid="stSelectbox"] div span,
    [data-testid="stSelectbox"] div div span {
        color: black !important;
        font-weight: 600 !important;
        text-shadow: none !important;
        opacity: 1 !important;
    }
    
    /* Nuclear option - force black on everything */
    .stSelectbox *,
    div[data-baseweb="select"] *,
    [data-testid="stSelectbox"] * {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

class CropRecommendationApp:
    def __init__(self):
        self.model = None
        self.df = None
        self.load_data_and_model()

    def load_data_and_model(self):
        """Load dataset and trained model"""
        try:
            # Load dataset
            self.df = pd.read_csv('data/nilamdata.csv')
            
            # Load trained model
            self.model = SimpleCropRecommendationModel()
            self.model.load_model()
            st.success("✅ Model and data loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model or data: {str(e)}")
            st.info("Please ensure the model is trained first by running simple_model.py")

    def get_feature_columns_from_data(self):
        """Get all feature columns from the dataset"""
        if self.df is not None:
            exclude_columns = ['Crop', 'Crop_Variety', 'Data_Collection_Date']
            feature_columns = [col for col in self.df.columns if col not in exclude_columns]
            return feature_columns
        return []

    def main_inputs(self):
        """Create main area for user inputs"""
        st.markdown("## 📝 Input Parameters", unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            
            # Location inputs
            st.markdown("### 🌍 Location Parameters")
            col1, col2 = st.columns(2)
            with col1:
                states = self.df['State'].unique() if self.df is not None else ['Andhra Pradesh']
                selected_state = st.selectbox("Select State", states, key="state")
            with col2:
                districts = self.df[self.df['State'] == selected_state]['District'].unique() if self.df is not None else ['Visakhapatnam']
                selected_district = st.selectbox("Select District", districts, key="district")
            
            # Get location data
            location_data = None
            if self.df is not None:
                location_filter = (self.df['State'] == selected_state) & (self.df['District'] == selected_district)
                if not self.df[location_filter].empty:
                    location_data = self.df[location_filter].iloc[0]

            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input("Latitude", 
                                        value=float(location_data['Latitude']) if location_data is not None else 17.69, 
                                        key="latitude")
            with col2:
                longitude = st.number_input("Longitude", 
                                          value=float(location_data['Longitude']) if location_data is not None else 83.3,
                                          key="longitude")

            # Soil parameters
            st.markdown("### 🌱 Soil Parameters")
            soil_types = ['Laterite', 'Alluvial', 'Black', 'Red', 'Sandy']
            soil_type = st.selectbox("Soil Type", soil_types, key="soil_type")

            col1, col2 = st.columns(2)
            with col1:
                ph = st.slider("pH Level", 4.0, 9.0, 
                             float(location_data['pH']) if location_data is not None else 7.72,
                             key="ph")
                nitrogen = st.number_input("Nitrogen (kg/ha)", 
                                        value=float(location_data['Nitrogen']) if location_data is not None else 370.5,
                                        key="nitrogen")
                phosphorus = st.number_input("Phosphorus (kg/ha)", 
                                           value=float(location_data['Phosphorus']) if location_data is not None else 14.3,
                                           key="phosphorus")
            with col2:
                potassium = st.number_input("Potassium (kg/ha)", 
                                          value=float(location_data['Potassium']) if location_data is not None else 277.9,
                                          key="potassium")
                organic_carbon = st.slider("Organic Carbon (%)", 0.1, 2.0, 
                                         float(location_data['Organic_Carbon']) if location_data is not None else 0.3,
                                         key="organic_carbon")
                moisture = st.slider("Moisture Content (%)", 10.0, 50.0, 
                                   float(location_data['Moisture_Content']) if location_data is not None else 30.6,
                                   key="moisture")

            # Weather parameters
            st.markdown("### 🌤️ Weather Parameters")
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider("Temperature (°C)", 15.0, 45.0, 
                                      float(location_data['Temperature']) if location_data is not None else 30.9,
                                      key="temperature")
                humidity = st.slider("Humidity (%)", 30.0, 100.0, 
                                   float(location_data['Humidity']) if location_data is not None else 57.3,
                                   key="humidity")
                rainfall = st.number_input("Rainfall (mm)", 
                                         value=float(location_data['Rainfall']) if location_data is not None else 14.59,
                                         key="rainfall")
            with col2:
                pressure = st.number_input("Pressure (hPa)", 
                                         value=float(location_data['Pressure']) if location_data is not None else 1003.7,
                                         key="pressure")
                wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, 
                                     float(location_data['Wind_Speed']) if location_data is not None else 9.5,
                                     key="wind_speed")
                uv_index = st.slider("UV Index", 0.0, 12.0, 
                                   float(location_data['UV_Index']) if location_data is not None else 2.97,
                                   key="uv_index")

            # Additional parameters
            st.markdown("### 📊 Additional Parameters")
            col1, col2 = st.columns(2)
            with col1:
                fertilizer_usage = st.number_input("Fertilizer Usage (kg/ha)", 
                                                 value=float(location_data['Fertilizer_Usage']) if location_data is not None and 'Fertilizer_Usage' in location_data else 250.0,
                                                 key="fertilizer")
                growing_period = st.number_input("Growing Period (days)", 
                                               value=int(location_data['Growing_Period_Days']) if location_data is not None and 'Growing_Period_Days' in location_data else 120,
                                               key="growing_period")
                market_price = st.number_input("Market Price (₹/quintal)", 
                                             value=float(location_data['Market_Price']) if location_data is not None and 'Market_Price' in location_data else 2500.0,
                                             key="market_price")
            with col2:
                ndvi = st.slider("NDVI", 0.0, 1.0, 
                               float(location_data['NDVI']) if location_data is not None and 'NDVI' in location_data else 0.7,
                               key="ndvi")
                pest_disease = st.number_input("Pest/Disease Incidence", 
                                             value=float(location_data['Pest_Disease_Incidence']) if location_data is not None and 'Pest_Disease_Incidence' in location_data else 5.0,
                                             min_value=0.0,
                                             max_value=20.0,
                                             step=0.1,
                                             key="pest_disease")

            st.markdown('</div>', unsafe_allow_html=True)

            return {
                'State': selected_state,
                'District': selected_district,
                'Latitude': latitude,
                'Longitude': longitude,
                'Soil_Type': soil_type,
                'pH': ph,
                'Nitrogen': nitrogen,
                'Phosphorus': phosphorus,
                'Potassium': potassium,
                'Organic_Carbon': organic_carbon,
                'Moisture_Content': moisture,
                'Temperature': temperature,
                'Humidity': humidity,
                'Rainfall': rainfall,
                'Pressure': pressure,
                'Wind_Speed': wind_speed,
                'UV_Index': uv_index,
                'Fertilizer_Usage': fertilizer_usage,
                'Growing_Period_Days': growing_period,
                'Market_Price': market_price,
                'NDVI': ndvi,
                'Pest_Disease_Incidence': pest_disease
            }

    def create_location_map(self, latitude, longitude, state, district):
        """Create location map"""
        m = folium.Map(location=[latitude, longitude], zoom_start=10)
        folium.Marker(
            [latitude, longitude],
            popup=f"{district}, {state}",
            tooltip=f"Selected Location: {district}",
            icon=folium.Icon(color='green', icon='leaf')
        ).add_to(m)
        return m

    def predict_crop(self, input_data):
        """Make crop prediction"""
        if self.model is None:
            return None, None, None
        
        try:
            # Create a DataFrame with all required columns
            input_df = pd.DataFrame([input_data])
            
            # Define all required features in the exact order the model expects
            required_features = [
                'State', 'District', 'Latitude', 'Longitude', 'Soil_Type', 'pH', 'Nitrogen', 
                'Phosphorus', 'Potassium', 'Organic_Carbon', 'Electrical_Conductivity', 
                'Moisture_Content', 'Water_Holding_Capacity', 'Zinc', 'Iron', 'Boron', 
                'Manganese', 'Copper', 'Sulfur', 'Soil_Fertility_Index', 'Soil_Health', 
                'pH_Classification', 'Salinity_Level', 'Temperature', 'Temperature_Min', 
                'Temperature_Max', 'Humidity', 'Pressure', 'Wind_Speed', 'Wind_Direction', 
                'Rainfall', 'Cloudiness', 'UV_Index', 'Visibility', 'Dew_Point', 
                'Solar_Radiation', 'Cropping_Season', 'Crop_Category', 'Growing_Period_Days', 
                'Predicted_Yield', 'Market_Price', 'Pest_Disease_Incidence', 'Fertilizer_Usage', 
                'Irrigation_Type', 'Water_Requirement', 'NDVI', 'Suitable_Months_Count'
            ]
            
            # Add missing features with appropriate default values
            default_values = {
                'Electrical_Conductivity': 1.79,
                'Water_Holding_Capacity': 34.3,
                'Zinc': 4.1,
                'Iron': 8.13,
                'Boron': 0.67,
                'Manganese': 1.63,
                'Copper': 2.72,
                'Sulfur': 15.7,
                'Soil_Fertility_Index': 88.1,
                'Soil_Health': 'Excellent',
                'pH_Classification': 'Neutral',
                'Salinity_Level': 'Moderately saline',
                'Temperature_Min': input_data['Temperature'] - 5,
                'Temperature_Max': input_data['Temperature'] + 5,
                'Wind_Direction': 118,
                'Cloudiness': 56,
                'Visibility': 9.6,
                'Dew_Point': 20.8,
                'Solar_Radiation': 22.29,
                'Cropping_Season': 'Kharif',
                'Crop_Category': 'Grains',
                'Irrigation_Type': 'Rainfed',
                'Suitable_Months_Count': 6,
                'Predicted_Yield': 5.0,
                'Water_Requirement': 500.0
            }
            
            # Add missing columns with default values
            for col, value in default_values.items():
                if col not in input_df.columns:
                    input_df[col] = value
            
            # Ensure all required features exist and reorder to match training order
            for feature in required_features:
                if feature not in input_df.columns:
                    # Add with default value if missing
                    if feature in default_values:
                        input_df[feature] = default_values[feature]
                    else:
                        input_df[feature] = 0.0  # Default numeric value
            
            # Reorder columns to match the exact order the model was trained on
            input_df = input_df[required_features]
            
            # Use the model's predict method directly
            crop_name, confidence, all_predictions = self.model.predict(input_df)
            
            return crop_name, confidence, all_predictions
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None, None

    def display_prediction_results(self, crop, confidence, all_predictions):
        """Display prediction results"""
        if crop is None:
            return
        
        # Main prediction card
        st.markdown(f"""
        <div class="prediction-card">
            <h2>🌾 Recommended Crop</h2>
            <h1>{crop}</h1>
            <h3>Confidence: {confidence:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display top 4 crop suggestions
        if all_predictions and len(all_predictions) > 1:
            st.markdown("### 🌱 Top 4 Crop Recommendations")
            
            # Create cards for top 4 crops
            top_4 = all_predictions[:4]
            cols = st.columns(2)
            
            # Define 4 different colors for variety
            colors = ["#28a745", "#007bff", "#ffc107", "#dc3545"]  # Green, Blue, Yellow, Red
            
            for i, (crop_name, prob) in enumerate(top_4):
                with cols[i % 2]:
                    confidence_color = colors[i]  # Use different color for each crop
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {confidence_color}20 0%, {confidence_color}10 100%);
                        border: 2px solid {confidence_color};
                        border-radius: 15px;
                        padding: 1.5rem;
                        margin: 0.5rem 0;
                        text-align: center;
                    ">
                        <h3 style="color: {confidence_color}; margin: 0;">{crop_name}</h3>
                        <h4 style="color: #ffffff; margin: 0.5rem 0; background-color: {confidence_color}; padding: 0.3rem 0.8rem; border-radius: 10px; display: inline-block;">{prob:.1%}</h4>
                        <p style="color: #ffffff; margin: 0; font-size: 0.9rem; background-color: {confidence_color}; padding: 0.2rem 0.6rem; border-radius: 8px; display: inline-block;">
                            {'🥇 Best Match' if i == 0 else '🥈 Good Alternative' if i == 1 else '🥉 Viable Option' if i == 2 else '📊 Alternative Choice'}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Charts section
            st.markdown("### 📊 Detailed Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart of top 8 predictions with dark theme
                top_8 = all_predictions[:8]
                crops = [pred[0] for pred in top_8]
                probs = [pred[1] for pred in top_8]
                
                fig = px.bar(
                    x=probs, 
                    y=crops, 
                    orientation='h',
                    title="Crop Recommendation Probabilities",
                    labels={'x': 'Probability', 'y': 'Crops'},
                    color=probs,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    height=400, 
                    plot_bgcolor='#1e1e1e', 
                    paper_bgcolor='#1e1e1e',
                    font=dict(color='white'),
                    title_font_color='white',
                    xaxis=dict(gridcolor='#333', color='white'),
                    yaxis=dict(gridcolor='#333', color='white')
                )
                fig.update_traces(marker_color='#4CAF50')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart with dark theme
                fig_pie = px.pie(
                    values=probs,
                    names=crops,
                    title="Crop Recommendation Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_layout(
                    height=400, 
                    plot_bgcolor='#1e1e1e', 
                    paper_bgcolor='#1e1e1e',
                    font=dict(color='white'),
                    title_font_color='white'
                )
                st.plotly_chart(fig_pie, use_container_width=True)

    def display_analytics(self, input_data):
        """Display agricultural analytics"""
        st.markdown("## 📈 Agricultural Analytics")
        
        # Enhanced metrics with insights
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temp_status = "Optimal" if 20 <= input_data['Temperature'] <= 35 else "High" if input_data['Temperature'] > 35 else "Low"
            temp_color = "#28a745" if temp_status == "Optimal" else "#ffc107" if temp_status == "High" else "#dc3545"
            st.markdown(f"""
            <div class="metric-card">
                <h4>🌡️ Temperature</h4>
                <h2 style="color: {temp_color};">{input_data['Temperature']:.1f}°C</h2>
                <p style="color: #666; font-size: 0.8rem;">Status: {temp_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            humidity_status = "Good" if 40 <= input_data['Humidity'] <= 80 else "High" if input_data['Humidity'] > 80 else "Low"
            humidity_color = "#28a745" if humidity_status == "Good" else "#ffc107"
            st.markdown(f"""
            <div class="metric-card">
                <h4>💧 Humidity</h4>
                <h2 style="color: {humidity_color};">{input_data['Humidity']:.1f}%</h2>
                <p style="color: #666; font-size: 0.8rem;">Status: {humidity_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            rainfall_status = "Adequate" if input_data['Rainfall'] >= 10 else "Low"
            rainfall_color = "#28a745" if rainfall_status == "Adequate" else "#dc3545"
            st.markdown(f"""
            <div class="metric-card">
                <h4>🌧️ Rainfall</h4>
                <h2 style="color: {rainfall_color};">{input_data['Rainfall']:.1f}mm</h2>
                <p style="color: #666; font-size: 0.8rem;">Status: {rainfall_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            ph_status = "Neutral" if 6.5 <= input_data['pH'] <= 7.5 else "Acidic" if input_data['pH'] < 6.5 else "Alkaline"
            ph_color = "#28a745" if ph_status == "Neutral" else "#ffc107"
            st.markdown(f"""
            <div class="metric-card">
                <h4>🧪 pH Level</h4>
                <h2 style="color: {ph_color};">{input_data['pH']:.2f}</h2>
                <p style="color: #666; font-size: 0.8rem;">Status: {ph_status}</p>
            </div>
            """, unsafe_allow_html=True)

        # Additional insights
        st.markdown("### 🔍 Environmental Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            # Soil Health Analysis
            st.markdown("#### 🌱 Soil Health Analysis")
            
            # NPK Analysis with dark theme
            npk_data = {
                'Nutrient': ['Nitrogen', 'Phosphorus', 'Potassium'],
                'Value': [input_data['Nitrogen'], input_data['Phosphorus'], input_data['Potassium']],
                'Status': [
                    "High" if input_data['Nitrogen'] > 300 else "Medium" if input_data['Nitrogen'] > 150 else "Low",
                    "High" if input_data['Phosphorus'] > 25 else "Medium" if input_data['Phosphorus'] > 10 else "Low",
                    "High" if input_data['Potassium'] > 250 else "Medium" if input_data['Potassium'] > 100 else "Low"
                ]
            }
            
            fig_npk = px.bar(
                x=npk_data['Nutrient'], 
                y=npk_data['Value'],
                title="NPK Levels Analysis",
                color=npk_data['Status'],
                color_discrete_map={'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'}
            )
            fig_npk.update_layout(
                showlegend=True, 
                plot_bgcolor='#1e1e1e', 
                paper_bgcolor='#1e1e1e',
                font=dict(color='white'),
                title_font_color='white',
                xaxis=dict(gridcolor='#333', color='white'),
                yaxis=dict(gridcolor='#333', color='white')
            )
            st.plotly_chart(fig_npk, use_container_width=True)
            
            # Soil insights
            st.markdown("**💡 Soil Insights:**")
            if input_data['Organic_Carbon'] < 0.5:
                st.warning("⚠️ Low organic carbon - consider adding organic matter")
            if input_data['Moisture_Content'] < 20:
                st.warning("⚠️ Low moisture content - irrigation may be needed")
            if input_data['Pest_Disease_Incidence'] > 10:
                st.warning("⚠️ High pest/disease incidence - monitor closely")
        
        with col2:
            # Weather Analysis
            st.markdown("#### 🌤️ Weather Analysis")
            
            # Weather radar chart
            weather_data = {
                'Parameter': ['Temperature', 'Humidity', 'Rainfall', 'Wind Speed', 'UV Index'],
                'Value': [
                    input_data['Temperature'] / 50,  # Normalize to 0-1
                    input_data['Humidity'] / 100,
                    min(input_data['Rainfall'] / 50, 1),  # Cap at 1
                    input_data['Wind_Speed'] / 20,
                    input_data['UV_Index'] / 12
                ]
            }
            
            fig_weather = px.line_polar(
                r=weather_data['Value'],
                theta=weather_data['Parameter'],
                line_close=True,
                title="Weather Conditions Radar"
            )
            fig_weather.update_layout(
                plot_bgcolor='#1e1e1e', 
                paper_bgcolor='#1e1e1e',
                font=dict(color='white'),
                title_font_color='white',
                polar=dict(
                    bgcolor='#1e1e1e',
                    radialaxis=dict(gridcolor='#333', color='white'),
                    angularaxis=dict(gridcolor='#333', color='white')
                )
            )
            fig_weather.update_traces(line_color='#4CAF50', fill='toself', fillcolor='rgba(76, 175, 80, 0.2)')
            st.plotly_chart(fig_weather, use_container_width=True)
            
            # Weather insights
            st.markdown("**💡 Weather Insights:**")
            if input_data['Temperature'] > 35:
                st.info("🌡️ High temperature - consider heat-tolerant crops")
            if input_data['Humidity'] > 80:
                st.info("💧 High humidity - good for moisture-loving crops")
            if input_data['Rainfall'] < 10:
                st.info("🌧️ Low rainfall - drought-resistant crops recommended")
        
        # Growing Conditions Summary
        st.markdown("### 📋 Growing Conditions Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #28a745;">
                <h4 style="color: #333;">🌾 Growing Period</h4>
                <h3 style="color: #333;">{input_data['Growing_Period_Days']} days</h3>
                <p style="color: #666;">Medium duration crop suitable</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #ffc107;">
                <h4 style="color: #333;">💰 Market Price</h4>
                <h3 style="color: #333;">₹{input_data['Market_Price']:.0f}/quintal</h3>
                <p style="color: #666;">Economic viability indicator</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #17a2b8;">
                <h4 style="color: #333;">🌱 NDVI</h4>
                <h3 style="color: #333;">{input_data['NDVI']:.2f}</h3>
                <p style="color: #666;">Vegetation health index</p>
            </div>
            """, unsafe_allow_html=True)

    def run(self):
        """Main application runner"""
        st.markdown('<h2 class="main-header">🌾 Smart Crop Recommendation System</h2>', unsafe_allow_html=True)
        
        # Main inputs
        input_data = self.main_inputs()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Prediction section
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
                if st.button("🔮 Get Crop Recommendation", key="predict_btn"):
                    with st.spinner("Analyzing soil and weather conditions..."):
                        crop, confidence, all_predictions = self.predict_crop(input_data)
                        
                        if crop:
                            # Store results in session state
                            st.session_state.last_prediction = {
                                'crop': crop,
                                'confidence': confidence,
                                'all_predictions': all_predictions,
                                'input_data': input_data
                            }
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_btn2:
                st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
                if st.button("🗑️ Clear Results", key="clear_btn"):
                    if 'last_prediction' in st.session_state:
                        del st.session_state.last_prediction
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display prediction and analytics if available (only once)
            if 'last_prediction' in st.session_state:
                pred = st.session_state.last_prediction
                self.display_prediction_results(pred['crop'], pred['confidence'], pred['all_predictions'])
                self.display_analytics(pred['input_data'])
        
        with col2:
            # Location map
            st.markdown("### 📍 Selected Location")
            location_map = self.create_location_map(
                input_data['Latitude'], 
                input_data['Longitude'],
                input_data['State'],
                input_data['District']
            )
            folium_static(location_map, width=400, height=300)
            
            # Additional info
            st.markdown("### ℹ️ Location Details")
            st.info(f"""
            **State:** {input_data['State']}
            **District:** {input_data['District']}
            **Coordinates:** {input_data['Latitude']:.2f}, {input_data['Longitude']:.2f}
            **Soil Type:** {input_data['Soil_Type']}
            """)

if __name__ == "__main__":
    app = CropRecommendationApp()
    app.run()