import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ---------- PAGE SETUP AND CONFIGURATION ----------
def setup_page():
    st.set_page_config(page_title="UHI Analysis System", page_icon="🌆", layout="wide")
    st.markdown("""
    <style>
        .main-header {font-size:2.5rem; color:#2c3e50; text-align:center; margin-bottom:1rem;}
        .sub-header {font-size:1.8rem; color:#34495e; margin-top:2rem; margin-bottom:1rem;}
        .card {background-color:#f8f9fa; border-radius:5px; padding:20px; margin-bottom:20px; box-shadow:0 4px 6px rgba(0,0,0,0.1);}
        .highlight {background-color:#e8f4f8; padding:10px; border-left:5px solid #3498db; margin-bottom:15px;}
        .footer {text-align:center; margin-top:3rem; color:#7f8c8d; font-size:0.8rem;}
        .metric-container {background-color:#f8f9fa; border-radius:5px; padding:15px; margin:10px 0; box-shadow:0 2px 4px rgba(0,0,0,0.1);}
        .metric-label {font-size:1rem; color:#7f8c8d;}
        .metric-value {font-size:1.8rem; font-weight:bold; color:#2c3e50;}
    </style>
    """, unsafe_allow_html=True)

# ---------- DATA GENERATION AND ANALYSIS FUNCTIONS ----------
def load_sample_data():
    """Generate synthetic UHI data for demonstration"""
    np.random.seed(42)
    lat_center, lon_center = 21.1458, 79.0882  # Nagpur, Maharashtra coordinates
    n_points = 500
    
    # Generate locations and urban features
    lats = lat_center + np.random.normal(0, 0.03, n_points)
    lons = lon_center + np.random.normal(0, 0.03, n_points)
    building_density = np.random.beta(2, 2, n_points)
    vegetation_index = 1 - np.random.beta(2, 1.5, n_points) * building_density
    albedo = np.random.beta(2, 5, n_points) * (1 - vegetation_index * 0.5)
    
    # Calculate distances and normalize
    dist_from_center = np.sqrt((lats - lat_center)**2 + (lons - lon_center)**2)
    dist_normalized = (dist_from_center - dist_from_center.min()) / (dist_from_center.max() - dist_from_center.min())
    
    # Calculate temperatures based on urban features
    base_temp = 32  # Higher base temperature for Nagpur
    uhi_effect = 5 * building_density - 3 * vegetation_index - 2 * albedo - 1 * dist_normalized
    temperature = base_temp + uhi_effect + np.random.normal(0, 0.5, n_points)
    surface_temp = temperature + 2 + 4 * building_density - 3 * vegetation_index + np.random.normal(0, 0.7, n_points)
    
    # Assign land use categories
    land_use = np.random.choice(
        ['Commercial', 'Residential', 'Industrial', 'Park', 'Water Body'],
        n_points, 
        p=[0.3, 0.4, 0.1, 0.15, 0.05]
    )
    
    # Create and return dataframe
    return pd.DataFrame({
        'latitude': lats, 'longitude': lons, 'building_density': building_density,
        'vegetation_index': vegetation_index, 'albedo': albedo, 'air_temperature': temperature,
        'surface_temperature': surface_temp, 'land_use': land_use, 'distance_from_center': dist_from_center
    })

def get_satellite_ndvi_data(lat, lon, date=None):
    """Mock function to simulate fetching NDVI data from satellite imagery"""
    np.random.seed(int(lat*100 + lon*100))
    lat_center, lon_center = 40.7128, -74.0060
    dist = np.sqrt((lat - lat_center)**2 + (lon - lon_center)**2)
    normalized_dist = min(dist / 0.1, 1)  # 0.1 is max_dist
    ndvi_value = -0.1 + normalized_dist * 0.7 + np.random.normal(0, 0.05)
    return max(-1, min(1, ndvi_value))  # Clamp to valid range

def get_temperature_prediction(features):
    """Predict temperature based on urban features"""
    return (25 + 5 * features['building_density'] - 4 * features['vegetation_index'] - 
            3 * features['albedo'] + np.random.normal(0, 0.2))

def create_cluster_map(data, n_clusters=5):
    """Create clusters of similar urban areas based on UHI characteristics"""
    # Prepare and cluster data
    features = data[['building_density', 'vegetation_index', 'albedo', 'air_temperature']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(scaled_features)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Create map
    colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F3']
    m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], 
                  zoom_start=12, tiles='CartoDB positron')
    
    # Add markers and legend
    for idx, row in data.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']], radius=5,
            color=colors[row['cluster']], fill=True, fill_color=colors[row['cluster']], fill_opacity=0.7,
            popup=f"Cluster: {row['cluster']}<br>Temp: {row['air_temperature']:.1f}°C<br>" +
                  f"Building Density: {row['building_density']:.2f}<br>" +
                  f"Vegetation: {row['vegetation_index']:.2f}<br>Land Use: {row['land_use']}"
        ).add_to(m)
    
    # Create legend
    legend_html = '''
    <div style="position:fixed; bottom:50px; right:50px; width:150px; height:160px; 
    border:2px solid grey; z-index:9999; font-size:14px; background-color:white;
    padding:10px; border-radius:5px;"><span style="font-weight:bold;">Clusters</span><br>
    '''
    
    for i, center in enumerate(centers):
        if center[0] > 0.6: cluster_desc = "Urban Core"
        elif center[1] > 0.6: cluster_desc = "Green Zone"
        elif center[3] > 27: cluster_desc = "Hot Spot"
        elif center[2] > 0.4: cluster_desc = "Reflective Area"
        else: cluster_desc = f"Cluster {i}"
        legend_html += f'<div style="background-color:{colors[i]}; width:20px; height:20px; display:inline-block;"></div> {cluster_desc}<br>'
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m, data

def suggest_interventions(data, location):
    """Suggest UHI mitigation strategies based on analysis"""
    # Get neighborhood data
    lat, lon = location
    distances = np.sqrt((data['latitude'] - lat)**2 + (data['longitude'] - lon)**2)
    neighborhood_data = data[distances < 0.01]  # ~1km radius
    
    if len(neighborhood_data) == 0:
        return {"message": "No data available for this location.", "suggestions": []}
    
    # Calculate neighborhood averages
    avg_temp = neighborhood_data['air_temperature'].mean()
    avg_building_density = neighborhood_data['building_density'].mean()
    avg_vegetation = neighborhood_data['vegetation_index'].mean()
    avg_albedo = neighborhood_data['albedo'].mean()
    suggestions = []
    
    # Generate suggestions based on conditions
    if avg_vegetation < 0.3:
        suggestions.append({
            "type": "Green Infrastructure", "priority": "High", "score": 5,
            "description": "Increase urban vegetation through tree planting, green roofs, or pocket parks.",
            "impact": f"Could reduce local temperature by {1.5 + np.random.uniform(0, 0.5):.1f}°C",
            "cost_estimate": "$$", "implementation_time": "Medium-term"
        })
    elif avg_vegetation < 0.5:
        suggestions.append({
            "type": "Green Infrastructure", "priority": "Medium", "score": 3,
            "description": "Enhance existing green spaces and add vegetation to streets.",
            "impact": f"Could reduce local temperature by {0.8 + np.random.uniform(0, 0.3):.1f}°C",
            "cost_estimate": "$", "implementation_time": "Short-term"
        })
    
    if avg_albedo < 0.2:
        suggestions.append({
            "type": "High-Albedo Surfaces", "priority": "High", "score": 5,
            "description": "Implement cool roofs and pavements to reflect more solar radiation.",
            "impact": f"Could reduce local temperature by {1.2 + np.random.uniform(0, 0.4):.1f}°C",
            "cost_estimate": "$$", "implementation_time": "Medium-term"
        })
    elif avg_albedo < 0.4:
        suggestions.append({
            "type": "High-Albedo Surfaces", "priority": "Medium", "score": 3,
            "description": "Gradually replace dark surfaces with lighter materials.",
            "impact": f"Could reduce local temperature by {0.7 + np.random.uniform(0, 0.3):.1f}°C",
            "cost_estimate": "$", "implementation_time": "Ongoing"
        })
    
    if avg_building_density > 0.7:
        suggestions.append({
            "type": "Urban Design", "priority": "High", "score": 4,
            "description": "Modify building arrangements to improve air flow and reduce heat trapping.",
            "impact": f"Could reduce local temperature by {1.0 + np.random.uniform(0, 0.5):.1f}°C",
            "cost_estimate": "$$$", "implementation_time": "Long-term"
        })
    elif avg_building_density > 0.5:
        suggestions.append({
            "type": "Urban Design", "priority": "Medium", "score": 2,
            "description": "Consider height variations in future developments to enhance ventilation.",
            "impact": f"Could reduce local temperature by {0.5 + np.random.uniform(0, 0.3):.1f}°C",
            "cost_estimate": "$$", "implementation_time": "Long-term"
        })
    
    if avg_temp > 28:
        suggestions.append({
            "type": "Water Features", "priority": "Medium", "score": 3,
            "description": "Incorporate water elements like fountains or retention ponds for cooling.",
            "impact": f"Could reduce local temperature by {0.8 + np.random.uniform(0, 0.3):.1f}°C",
            "cost_estimate": "$$", "implementation_time": "Medium-term"
        })
    
    # Sort suggestions by priority
    suggestions.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "message": "Analysis complete. Interventions suggested based on local conditions.",
        "local_temperature": f"{avg_temp:.1f}°C",
        "building_density": f"{avg_building_density:.2f}",
        "vegetation_index": f"{avg_vegetation:.2f}",
        "albedo": f"{avg_albedo:.2f}",
        "suggestions": suggestions
    }

def simulate_intervention_impact(current_data, intervention_type, intensity=0.5):
    """Simulate the impact of implementing a specific intervention"""
    new_data = current_data.copy()
    
    # Apply intervention effects
    if intervention_type == "Green Infrastructure":
        new_data['vegetation_index'] = new_data['vegetation_index'] + (1 - new_data['vegetation_index']) * intensity * 0.5
        new_data['vegetation_index'] = new_data['vegetation_index'].clip(upper=1.0)
    elif intervention_type == "High-Albedo Surfaces":
        new_data['albedo'] = new_data['albedo'] + (1 - new_data['albedo']) * intensity * 0.6
        new_data['albedo'] = new_data['albedo'].clip(upper=1.0)
    elif intervention_type == "Urban Design":
        new_data['building_density'] = new_data['building_density'] * (1 - intensity * 0.3)
    elif intervention_type == "Water Features":
        lat_center, lon_center = new_data['latitude'].mean(), new_data['longitude'].mean()
        distances = np.sqrt((new_data['latitude'] - lat_center)**2 + (new_data['longitude'] - lon_center)**2)
        max_dist = distances.max()
        cooling_effect = intensity * 0.5 * (1 - distances/max_dist)
        new_data['vegetation_index'] = new_data['vegetation_index'] + cooling_effect * 0.3
        new_data['vegetation_index'] = new_data['vegetation_index'].clip(upper=1.0)
    
    # Recalculate temperatures
    base_temp = 25
    uhi_effect = (5 * new_data['building_density'] - 3 * new_data['vegetation_index'] - 
                 2 * new_data['albedo'] - 1 * new_data['distance_from_center'].clip(0, 1))
    np.random.seed(42)
    new_data['air_temperature'] = base_temp + uhi_effect + np.random.normal(0, 0.2, len(new_data))
    
    return new_data

def optimize_interventions(data, budget_level='medium', priority='temperature'):
    """Optimize intervention strategy based on budget constraints and priorities"""
    # Define budget levels in Indian Rupees (₹)
    budget_map = {'low': 2000000, 'medium': 5000000, 'high': 10000000}  # ₹20 lakh, ₹50 lakh, ₹1 crore
    budget = budget_map.get(budget_level, 5000000)
    
    # Define intervention options with costs in INR
    interventions = [
        {'name': 'Tree Planting', 'type': 'Green Infrastructure', 'cost_per_unit': 200000, 
         'temp_reduction_per_unit': 0.05, 'max_units': 30},
        {'name': 'Cool Roofs', 'type': 'High-Albedo Surfaces', 'cost_per_unit': 300000, 
         'temp_reduction_per_unit': 0.08, 'max_units': 20},
        {'name': 'Cool Pavements', 'type': 'High-Albedo Surfaces', 'cost_per_unit': 400000, 
         'temp_reduction_per_unit': 0.06, 'max_units': 15},
        {'name': 'Green Roofs', 'type': 'Green Infrastructure', 'cost_per_unit': 500000, 
         'temp_reduction_per_unit': 0.1, 'max_units': 12},
        {'name': 'Water Features', 'type': 'Water Features', 'cost_per_unit': 600000, 
         'temp_reduction_per_unit': 0.12, 'max_units': 8}
    ]
    
    # Sort interventions by priority
    if priority == 'temperature':
        interventions.sort(key=lambda x: x['temp_reduction_per_unit'] / x['cost_per_unit'], reverse=True)
    elif priority == 'cost':
        interventions.sort(key=lambda x: x['cost_per_unit'])
    elif priority == 'implementation':
        interventions.sort(key=lambda x: x['cost_per_unit'])
    
    # Allocate budget using greedy algorithm
    allocation, remaining_budget = [], budget
    for intervention in interventions:
        affordable_units = min(intervention['max_units'], int(remaining_budget / intervention['cost_per_unit']))
        if affordable_units > 0:
            cost = affordable_units * intervention['cost_per_unit']
            temp_reduction = affordable_units * intervention['temp_reduction_per_unit']
            allocation.append({
                'name': intervention['name'], 'type': intervention['type'], 'units': affordable_units,
                'cost': cost, 'temperature_reduction': temp_reduction
            })
            remaining_budget -= cost
    
    # Calculate totals
    total_cost = sum(item['cost'] for item in allocation)
    total_reduction = sum(item['temperature_reduction'] for item in allocation)
    
    return {
        'budget': budget, 'used_budget': total_cost, 'remaining_budget': remaining_budget,
        'estimated_temperature_reduction': total_reduction, 'allocation': allocation
    }

# ---------- UI MODULES ----------
def show_dashboard(data):
    """Display the main dashboard with overview metrics and visualizations"""
    st.markdown('<h2 class="sub-header">UHI Dashboard</h2>', unsafe_allow_html=True)
    
    # Summary metrics
    cols = st.columns(4)
    with cols[0]: st.markdown(f'<div class="metric-container"><div class="metric-label">Average Temperature</div><div class="metric-value">{data["air_temperature"].mean():.1f}°C</div></div>', unsafe_allow_html=True)
    with cols[1]: st.markdown(f'<div class="metric-container"><div class="metric-label">Max Temperature</div><div class="metric-value">{data["air_temperature"].max():.1f}°C</div></div>', unsafe_allow_html=True)
    with cols[2]: st.markdown(f'<div class="metric-container"><div class="metric-label">Average Vegetation Index</div><div class="metric-value">{data["vegetation_index"].mean():.2f}</div></div>', unsafe_allow_html=True)
    with cols[3]: st.markdown(f'<div class="metric-container"><div class="metric-label">Temperature Anomaly</div><div class="metric-value">+{data["air_temperature"].max() - data["air_temperature"].min():.1f}°C</div></div>', unsafe_allow_html=True)
    
    st.selectbox("Select City", ["Nagpur, Maharashtra (Demo)"], index=0)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Heat Map", "Analysis", "Trends"])
    
    with tab1:
        st.markdown("### Urban Heat Map")
        st.write("This heat map shows the temperature distribution across Nagpur.")
        
        # Create heat map
        m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=12, tiles='CartoDB positron')
        heat_data = [[row['latitude'], row['longitude'], row['air_temperature']] for _, row in data.iterrows()]
        HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'}).add_to(m)
        folium_static(m)
    
    with tab2:
        st.markdown("### UHI Analysis")
        
        # Create charts
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(data, x='building_density', y='air_temperature', color='vegetation_index',
                            color_continuous_scale='Viridis', title='Temperature vs Building Density',
                            labels={'building_density': 'Building Density', 'air_temperature': 'Temperature (°C)',
                                   'vegetation_index': 'Vegetation Index'}, size_max=10, hover_data=['land_use'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            land_use_temp = data.groupby('land_use')['air_temperature'].mean().reset_index()
            fig = px.bar(land_use_temp, x='land_use', y='air_temperature', color='air_temperature',
                        color_continuous_scale='Thermal', title='Average Temperature by Land Use',
                        labels={'land_use': 'Land Use Type', 'air_temperature': 'Avg Temperature (°C)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.markdown("### Feature Correlations")
        corr_features = ['building_density', 'vegetation_index', 'albedo', 'air_temperature', 'surface_temperature']
        corr_matrix = data[corr_features].corr()
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                       title='Correlation Between Urban Features',
                       labels=dict(x='Features', y='Features', color='Correlation'))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Temperature Trends")
        
        # Generate temporal data (for demo)
        dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
        avg_temps = [data['air_temperature'].mean() + np.sin(i/5) + np.random.normal(0, 0.3) for i in range(30)]
        max_temps = [t + 2 + np.random.normal(0, 0.2) for t in avg_temps]
        time_df = pd.DataFrame({'date': dates, 'avg_temperature': avg_temps, 'max_temperature': max_temps})
        
        # Plot time series
        fig = px.line(time_df, x='date', y=['avg_temperature', 'max_temperature'],
                     title='Temperature Trends Over Time',
                     labels={'date': 'Date', 'value': 'Temperature (°C)', 'variable': 'Metric'})
        fig.update_layout(legend_title_text='')
        st.plotly_chart(fig, use_container_width=True)
        
        # Temperature forecast
        st.markdown("### Temperature Forecast")
        forecast_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=7, freq='D')
        forecast_temps = [avg_temps[-1] + 0.1*i + np.random.normal(0, 0.2) for i in range(7)]
        full_dates = dates.append(forecast_dates)
        full_temps = avg_temps + forecast_temps
        forecast_df = pd.DataFrame({
            'date': full_dates,
            'temperature': full_temps,
            'type': ['Historical']*30 + ['Forecast']*7
        })
        
        fig = px.line(forecast_df, x='date', y='temperature', color='type',
                     title='7-Day Temperature Forecast',
                     labels={'date': 'Date', 'temperature': 'Avg. Temperature (°C)'})
        st.plotly_chart(fig, use_container_width=True)

def show_uhi_detection(data):
    """Display the UHI detection and analysis module"""
    st.markdown('<h2 class="sub-header">UHI Detection & Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module uses satellite imagery and street-level data to detect urban heat island hotspots. Upload your own data or use our demo data to visualize UHI patterns.</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Satellite Analysis", "Cluster Analysis", "Temporal Analysis"])
    
    with tab1:
        st.markdown("### Satellite-Based UHI Detection")
        st.write("Select an area to analyze or use the demo data:")
        
        # Input controls
        col1, col2 = st.columns(2)
        with col1:
            location = st.selectbox("Select location:", ["New York City, NY", "Custom Location"])
            lat, lon = (40.7128, -74.0060) if location == "New York City, NY" else (
                st.number_input("Latitude:", value=40.7128, format="%.4f"),
                st.number_input("Longitude:", value=-74.0060, format="%.4f")
            )
        
        with col2:
            analysis_date = st.date_input("Select date for analysis:", datetime.date(2025, 6, 1))
            data_source = st.selectbox("Data source:", ["Landsat 9", "Sentinel-2", "MODIS"])
        
        # Run analysis
        if st.button("Run Satellite Analysis"):
            st.markdown("#### Analysis Results")
            
            # Metrics
            cols = st.columns(3)
            with cols[0]:
                ndvi = get_satellite_ndvi_data(lat, lon, analysis_date)
                st.metric("NDVI Index", f"{ndvi:.2f}", "-0.05")
            with cols[1]:
                surface_temp = data[data['latitude'].between(lat-0.01, lat+0.01) & 
                                    data['longitude'].between(lon-0.01, lon+0.01)]['surface_temperature'].mean()
                st.metric("Surface Temperature", f"{surface_temp:.1f}°C", "+2.3°C")
            with cols[2]:
                building_density = data[data['latitude'].between(lat-0.01, lat+0.01) & 
                                        data['longitude'].between(lon-0.01, lon+0.01)]['building_density'].mean()
                st.metric("Building Density", f"{building_density:.2f}", "+0.04")
            
            # Heat map
            st.markdown("#### Surface Temperature Map")
            area_data = data[data['latitude'].between(lat-0.03, lat+0.03) & data['longitude'].between(lon-0.03, lon+0.03)]
            m = folium.Map(location=[lat, lon], zoom_start=14, tiles='CartoDB positron')
            heat_data = [[row['latitude'], row['longitude'], row['surface_temperature']] for _, row in area_data.iterrows()]
            HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'},
                   min_opacity=0.5, blur=10).add_to(m)
            folium.Marker([lat, lon], popup=f"Selected Location<br>NDVI: {ndvi:.2f}<br>Temp: {surface_temp:.1f}°C",
                         icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
            folium_static(m)
            
            # NDVI vs Temperature
            st.markdown("#### NDVI vs Surface Temperature")
            fig = px.scatter(area_data, x='vegetation_index', y='surface_temperature', color='surface_temperature',
                            color_continuous_scale='Thermal', title='Vegetation Index vs Surface Temperature',
                            labels={'vegetation_index': 'Vegetation Index (NDVI)', 
                                   'surface_temperature': 'Surface Temperature (°C)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis summary
            st.markdown("#### Analysis Summary")
            avg_temp = area_data['surface_temperature'].mean()
            uhi_severity = "Severe" if avg_temp > 30 else "Moderate" if avg_temp > 27 else "Low"
            impact = "High" if avg_temp > 30 else "Medium" if avg_temp > 27 else "Low"
            
            st.markdown(f"""
            **UHI Severity:** {uhi_severity}  
            **Potential Impact:** {impact}  
            **Key Factors:**
            - Building density contributes approximately {(building_density * 100):.1f}% to the UHI effect
            - Vegetation cover is {(area_data['vegetation_index'].mean() * 100):.1f}% of the analyzed area
            - Average surface temperature is {avg_temp:.1f}°C, which is {avg_temp - 25:.1f}°C above the baseline temperature
            """)
    
    with tab2:
        st.markdown("### UHI Cluster Analysis")
        st.write("This analysis identifies similar urban areas based on their heat characteristics.")
        
        # Clustering
        n_clusters = st.slider("Number of clusters:", min_value=3, max_value=7, value=5)
        cluster_map, clustered_data = create_cluster_map(data, n_clusters)
        
        st.markdown("#### Urban Heat Clusters")
        folium_static(cluster_map)
        
        # Cluster characteristics
        st.markdown("#### Cluster Characteristics")
        cluster_means = clustered_data.groupby('cluster')[
            ['building_density', 'vegetation_index', 'albedo', 'air_temperature']
        ].mean().reset_index()
        
        # Parallel coordinates plot
        fig = px.parallel_coordinates(cluster_means,
                                     dimensions=['building_density', 'vegetation_index', 'albedo', 'air_temperature'],
                                     color='cluster',
                                     labels={'building_density': 'Building Density', 'vegetation_index': 'Vegetation',
                                            'albedo': 'Albedo', 'air_temperature': 'Temperature (°C)'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster summary
        st.markdown("#### Cluster Summary")
        cluster_descriptions = []
        # Cluster summary
        st.markdown("#### Cluster Summary")
        cluster_descriptions = []
        for _, row in cluster_means.iterrows():
            if row['air_temperature'] > 28 and row['building_density'] > 0.6:
                description = "Urban Core - Hot Spot"
            elif row['vegetation_index'] > 0.6:
                description = "Green Zone - Cool Area"
            elif row['albedo'] > 0.4:
                description = "High Reflectivity Zone"
            elif row['building_density'] > 0.5 and row['vegetation_index'] < 0.3:
                description = "Dense Urban - Moderate Heat"
            else:
                description = "Mixed Urban Zone"
            cluster_descriptions.append(description)
        
        # Display cluster summary
        cluster_means['description'] = cluster_descriptions
        display_df = cluster_means.copy()
        display_df.columns = ['Cluster', 'Building Density', 'Vegetation', 'Albedo', 'Temperature (°C)', 'Description']
        display_df = display_df[['Cluster', 'Description', 'Temperature (°C)', 'Building Density', 'Vegetation', 'Albedo']]
        st.dataframe(display_df.round(2))
    
    with tab3:
        st.markdown("### Temporal UHI Analysis")
        st.write("Analyze how UHI patterns change over time.")
        
        # Time period selection
        period = st.selectbox("Select analysis period:", ["Daily Cycle", "Seasonal Variation", "Annual Trend"])
        
        if period == "Daily Cycle":
            # Generate data for daily cycle
            hours = list(range(24))
            urban_temps = [25 + 5 * np.sin((h - 2) * np.pi / 24) for h in hours]
            rural_temps = [22 + 4 * np.sin((h - 2) * np.pi / 24) for h in hours]
            daily_df = pd.DataFrame({
                'hour': hours,
                'urban_temperature': urban_temps,
                'rural_temperature': rural_temps,
                'uhi_intensity': [u - r for u, r in zip(urban_temps, rural_temps)]
            })
            
            # Temperature comparison
            fig = px.line(daily_df, x='hour', y=['urban_temperature', 'rural_temperature'],
                         title='Daily Temperature Cycle: Urban vs Rural',
                         labels={'hour': 'Hour of Day', 'value': 'Temperature (°C)', 'variable': 'Location'})
            st.plotly_chart(fig, use_container_width=True)
            
            # UHI intensity
            fig = px.bar(daily_df, x='hour', y='uhi_intensity', title='Urban Heat Island Intensity Throughout the Day',
                        labels={'hour': 'Hour of Day', 'uhi_intensity': 'UHI Intensity (°C)'},
                        color='uhi_intensity', color_continuous_scale='Thermal')
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis
            max_uhi_hour = daily_df.loc[daily_df['uhi_intensity'].idxmax(), 'hour']
            max_uhi = daily_df['uhi_intensity'].max()
            
            st.markdown(f"""
            #### Key Findings:
            - Maximum UHI intensity of {max_uhi:.1f}°C occurs at {int(max_uhi_hour):02d}:00 hours
            - UHI effect is strongest during night and early morning hours
            - Minimum temperature difference observed during mid-day
            
            This pattern is typical of urban areas where built surfaces release stored heat during the night,
            while rural areas cool more rapidly after sunset.
            """)
        
        elif period == "Seasonal Variation":
            # Generate data for seasonal variation
            months = list(range(1, 13))
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            summer_peak = 7  # July
            urban_temps = [25 + 10 * np.sin((m - 1) * np.pi / 6) for m in months]
            rural_temps = [22 + 12 * np.sin((m - 1) * np.pi / 6) for m in months]
            uhi_intensity = [3 + 1.5 * np.sin((m - summer_peak) * np.pi / 6) for m in months]
            
            seasonal_df = pd.DataFrame({
                'month': month_names, 'month_num': months,
                'urban_temperature': urban_temps, 'rural_temperature': rural_temps,
                'uhi_intensity': uhi_intensity
            })
            
            # Temperature comparison
            fig = px.line(seasonal_df, x='month', y=['urban_temperature', 'rural_temperature'],
                         title='Seasonal Temperature Variation: Urban vs Rural',
                         labels={'month': 'Month', 'value': 'Temperature (°C)', 'variable': 'Location'})
            fig.update_xaxes(categoryorder='array', categoryarray=month_names)
            st.plotly_chart(fig, use_container_width=True)
            
            # UHI intensity
            fig = px.bar(seasonal_df, x='month', y='uhi_intensity', title='Urban Heat Island Intensity by Season',
                        labels={'month': 'Month', 'uhi_intensity': 'UHI Intensity (°C)'},
                        color='uhi_intensity', color_continuous_scale='Thermal')
            fig.update_xaxes(categoryorder='array', categoryarray=month_names)
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis
            max_uhi_month = seasonal_df.loc[seasonal_df['uhi_intensity'].idxmax(), 'month']
            max_uhi = seasonal_df['uhi_intensity'].max()
            
            st.markdown(f"""
            #### Key Findings:
            - Maximum UHI intensity of {max_uhi:.1f}°C occurs in {max_uhi_month}
            - Summer months generally show higher UHI intensity
            - Urban and rural temperature gap varies by season
            """)
        
        elif period == "Annual Trend":
            # Generate data for annual trend
            years = list(range(2020, 2026))
            base_uhi = [2.8, 3.0, 3.3, 3.5, 3.7, 4.0]
            uhi_trend = [b + np.random.normal(0, 0.1) for b in base_uhi]
            
            trend_df = pd.DataFrame({'year': years, 'uhi_intensity': uhi_trend})
            
            # Plot trend
            fig = px.line(trend_df, x='year', y='uhi_intensity',
                         title='Urban Heat Island Intensity Annual Trend',
                         labels={'year': 'Year', 'uhi_intensity': 'Average UHI Intensity (°C)'},
                         markers=True)
            
            # Add trendline
            fig.add_trace(px.scatter(trend_df, x='year', y='uhi_intensity', trendline='ols').data[1])
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis
            initial_uhi = trend_df.iloc[0]['uhi_intensity']
            final_uhi = trend_df.iloc[-1]['uhi_intensity']
            percent_change = ((final_uhi - initial_uhi) / initial_uhi) * 100
            
            st.markdown(f"""
            #### Key Findings:
            - UHI intensity has increased by {percent_change:.1f}% over the past 6 years
            - Average annual increase of {(final_uhi - initial_uhi) / 5:.2f}°C per year
            - If this trend continues, UHI intensity could reach {final_uhi + (final_uhi - initial_uhi) / 5 * 5:.1f}°C by 2030
            """)

def show_temperature_prediction(data):
    """Display the temperature prediction module"""
    st.markdown('<h2 class="sub-header">Temperature Prediction Model</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module uses machine learning to predict hyperlocal temperature variations based on urban features. Adjust the parameters to see how different urban configurations affect local temperatures.</div>', unsafe_allow_html=True)
    
    # Create layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Parameter inputs
        st.markdown("### Urban Feature Parameters")
        building_density = st.slider("Building Density:", 0.0, 1.0, 0.5, 0.01,
                                   help="Higher values indicate more dense urban development")
        vegetation_index = st.slider("Vegetation Index:", 0.0, 1.0, 0.3, 0.01,
                                   help="Higher values indicate more vegetation (trees, parks, etc.)")
        albedo = st.slider("Surface Albedo:", 0.0, 1.0, 0.2, 0.01,
                          help="Higher values indicate more reflective surfaces")
        
        st.markdown("### Additional Factors")
        land_use = st.selectbox("Land Use Type:", ["Residential", "Commercial", "Industrial", "Park", "Mixed Use"])
        time_of_day = st.select_slider("Time of Day:", 
                                      options=["Early Morning", "Morning", "Noon", "Afternoon", "Evening", "Night"])
        season = st.selectbox("Season:", ["Winter", "Spring", "Summer", "Fall"])
        predict_button = st.button("Predict Temperature")
    
    with col2:
        if predict_button:
            # Prepare prediction
            features = {'building_density': building_density, 'vegetation_index': vegetation_index, 'albedo': albedo}
            predicted_temp = get_temperature_prediction(features)
            
            # Adjustments
            time_factors = {"Early Morning": -2.0, "Morning": -0.5, "Noon": 1.5, 
                           "Afternoon": 2.0, "Evening": 0.0, "Night": -1.5}
            season_factors = {"Winter": -5.0, "Spring": 0.0, "Summer": 5.0, "Fall": 0.0}
            land_use_factors = {"Residential": 0.0, "Commercial": 1.0, "Industrial": 1.5, 
                               "Park": -2.0, "Mixed Use": 0.5}
            
            predicted_temp += time_factors[time_of_day] + season_factors[season] + land_use_factors[land_use]
            
            # Display prediction
            st.markdown(f"""
            <div style="text-align:center; margin:20px; padding:20px; background-color:#f8f9fa; 
            border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
                <h3>Predicted Temperature</h3>
                <div style="font-size:48px; font-weight:bold; color:#e74c3c;">{predicted_temp:.1f}°C</div>
                <p>Based on the urban features and conditions you specified</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparison to baseline
            baseline_temp = 25.0
            temp_difference = predicted_temp - baseline_temp
            st.markdown(f"""
            <div style="margin:20px 0;">
                <strong>Comparison to Baseline:</strong> {temp_difference:.1f}°C 
                {'warmer' if temp_difference > 0 else 'cooler'} than the baseline temperature
            </div>
            """, unsafe_allow_html=True)
            
            # Feature impact analysis
            st.markdown("### Feature Impact Analysis")
            building_impact = 5 * building_density
            vegetation_impact = -4 * vegetation_index
            albedo_impact = -3 * albedo
            
            impact_data = pd.DataFrame({
                'Feature': ['Building Density', 'Vegetation', 'Albedo', 'Time of Day', 'Season', 'Land Use'],
                'Impact': [building_impact, vegetation_impact, albedo_impact, 
                          time_factors[time_of_day], season_factors[season], land_use_factors[land_use]]
            })
            
            impact_data['Abs_Impact'] = impact_data['Impact'].abs()
            impact_data = impact_data.sort_values('Abs_Impact', ascending=False)
            
            # Impact visualization
            fig = px.bar(impact_data, y='Feature', x='Impact', orientation='h',
                        title='Feature Impact on Temperature', color='Impact',
                        color_continuous_scale='RdBu_r', labels={'Impact': 'Temperature Impact (°C)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### Recommendations")
            if predicted_temp > 28:
                st.markdown("""
                #### High Temperature Zone
                This configuration results in elevated temperatures that could contribute to urban heat island effects.
                Consider the following mitigation strategies:
                - **Increase vegetation coverage**: Adding trees, green roofs, or pocket parks could reduce temperature by 1-2°C
                - **Implement cool surfaces**: Replacing dark surfaces with high-albedo materials
                - **Redesign urban geometry**: Modify building arrangements to improve air flow
                """)
            elif predicted_temp > 25:
                st.markdown("""
                #### Moderate Temperature Zone
                This configuration shows a moderate urban heat island effect. Some improvements could help:
                - **Enhance existing green spaces**: Increase vegetation in available areas
                - **Gradual surface replacements**: Consider light-colored materials for upcoming renovations
                - **Water features**: Small water elements could provide localized cooling
                """)
            else:
                st.markdown("""
                #### Low Temperature Zone
                This configuration effectively minimizes urban heat island effects. To maintain:
                - **Preserve existing vegetation**: Protect and maintain current green spaces
                - **Continue high-albedo practices**: Maintain reflective surfaces when replacing materials
                - **Use as model**: Consider applying similar configurations to other urban areas
                """)
        else:
            # Placeholder content
            st.markdown("""
            ### Temperature Prediction
            Adjust the parameters on the left and click "Predict Temperature" to see results.
            
            The prediction model considers:
            - Building density
            - Vegetation coverage
            - Surface reflectivity (albedo)
            - Land use type
            - Time of day and seasonal factors
            """)
            
            # Example visualization
            building_range = np.linspace(0, 1, 50)
            vegetation_range = np.linspace(0, 1, 50)
            X, Y = np.meshgrid(building_range, vegetation_range)
            Z = 25 + 5 * X - 4 * Y  # Simplified temperature model
            
            fig = go.Figure(data=[go.Contour(
                z=Z, x=building_range, y=vegetation_range, colorscale='Thermal',
                contours=dict(start=22, end=30, size=0.5, showlabels=True),
                colorbar=dict(title='Temperature (°C)', titleside='right')
            )])
            
            fig.update_layout(title='Temperature Prediction by Building Density and Vegetation',
                             xaxis_title='Building Density', yaxis_title='Vegetation Index', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            The contour plot above shows how temperature varies with building density and vegetation coverage.
            - **Higher building density** tends to increase temperature (moving right on the x-axis)
            - **Higher vegetation** tends to decrease temperature (moving up on the y-axis)
            - The contour lines represent equal temperature values
            """)

def show_intervention_planning(data):
    """Display the intervention planning module"""
    st.markdown('<h2 class="sub-header">UHI Intervention Planning</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module helps planners identify the most effective UHI mitigation strategies for specific urban areas. Select a location and the system will analyze local conditions and recommend tailored interventions.</div>', unsafe_allow_html=True)
    
    # Create layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Select Location")
        location_method = st.radio("Selection method:", ["Map Selection", "Address Search"])
        
        # Location input
        if location_method == "Map Selection":
            st.write("Map will appear here in a real application.")
            lat = st.number_input("Latitude:", value=40.7128, format="%.4f")
            lon = st.number_input("Longitude:", value=-74.0060, format="%.4f")
        else:
            address = st.text_input("Enter address:", "Times Square, New York, NY")
            if address:
                st.write("Address geocoded to coordinates:")
                lat, lon = 40.7580, -73.9855  # Times Square coordinates
                st.write(f"Latitude: {lat}, Longitude: {lon}")
            else:
                lat, lon = 40.7128, -74.0060  # Default NYC coordinates
        
        # Parameters
        radius = st.slider("Analysis radius (km):", 0.5, 5.0, 1.0, 0.5)
        priority = st.selectbox("Optimization priority:", ["Temperature Reduction", "Cost Efficiency", 
                                                          "Implementation Speed", "Balanced Approach"])
        
        # Constraints
        st.markdown("### Constraints")
        budget_constraint = st.select_slider("Budget level:", options=["Low", "Medium", "High"])
        time_constraint = st.select_slider("Implementation timeframe:", 
                                          options=["Short-term", "Medium-term", "Long-term"])
        
        analyze_button = st.button("Analyze & Suggest Interventions")
    
    with col2:
        if analyze_button:
            st.markdown("### Analysis Results")
            
            # Get recommendations
            location = (lat, lon)
            intervention_results = suggest_interventions(data, location)
            
            # Display local conditions
            st.markdown("#### Local Conditions")
            metric_cols = st.columns(4)
            with metric_cols[0]: st.metric("Temperature", intervention_results["local_temperature"], "+3.2°C")
            with metric_cols[1]: st.metric("Building Density", intervention_results["building_density"], "+0.15")
            with metric_cols[2]: st.metric("Vegetation Index", intervention_results["vegetation_index"], "-0.08")
            with metric_cols[3]: st.metric("Albedo", intervention_results["albedo"], "-0.12")
            
            # Area map
            st.markdown("#### Area Map")
            area_data = data[(data['latitude'] - lat)**2 + (data['longitude'] - lon)**2 <= (radius/111)**2]
            
            m = folium.Map(location=[lat, lon], zoom_start=14, tiles='CartoDB positron')
            heat_data = [[row['latitude'], row['longitude'], row['air_temperature']] for _, row in area_data.iterrows()]
            HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'},
                   min_opacity=0.5, blur=10).add_to(m)
            
            folium.Marker([lat, lon], popup=f"Selected Location<br>Temp: {intervention_results['local_temperature']}",
                         icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
            
            folium.Circle(location=[lat, lon], radius=radius * 1000, color='blue',
                         fill=True, fill_opacity=0.1).add_to(m)
            
            folium_static(m)
            
            # Intervention recommendations
            st.markdown("#### Recommended Interventions")
            
            if len(intervention_results["suggestions"]) > 0:
                # Create tabs for different intervention types
                intervention_types = list(set(s["type"] for s in intervention_results["suggestions"]))
                tabs = st.tabs(intervention_types + ["All Interventions"])
                
                for i, tab in enumerate(tabs):
                    with tab:
                        if i < len(intervention_types):
                            # Filter for this type
                            current_type = intervention_types[i]
                            type_suggestions = [s for s in intervention_results["suggestions"] 
                                               if s["type"] == current_type]
                            
                            for j, suggestion in enumerate(type_suggestions):
                                st.markdown(f"""
                                <div style="margin-bottom:15px; padding:15px; 
                                background-color:{'#e8f4f8' if j % 2 == 0 else '#f0f7fa'}; border-radius:5px;">
                                    <h4>{suggestion["description"]}</h4>
                                    <p><strong>Impact:</strong> {suggestion["impact"]}</p>
                                    <p><strong>Priority:</strong> {suggestion["priority"]}</p>
                                    <p><strong>Cost:</strong> {suggestion["cost_estimate"]}</p>
                                    <p><strong>Implementation:</strong> {suggestion["implementation_time"]}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            # Show all interventions
                            for j, suggestion in enumerate(intervention_results["suggestions"]):
                                st.markdown(f"""
                                <div style="margin-bottom:15px; padding:15px; 
                                background-color:{'#e8f4f8' if j % 2 == 0 else '#f0f7fa'}; border-radius:5px;">
                                    <h4>{suggestion["type"]}: {suggestion["description"]}</h4>
                                    <p><strong>Impact:</strong> {suggestion["impact"]}</p>
                                    <p><strong>Priority:</strong> {suggestion["priority"]}</p>
                                    <p><strong>Cost:</strong> {suggestion["cost_estimate"]}</p>
                                    <p><strong>Implementation:</strong> {suggestion["implementation_time"]}</p>
                                </div>
                                """, unsafe_allow_html=True)
            else:
                st.write("No specific interventions found for this location.")
            
            # Impact visualization
            if len(intervention_results["suggestions"]) > 0:
                st.markdown("#### Intervention Impact Visualization")
                
                # Select intervention to simulate
                intervention_options = [f"{s['type']}: {s['description']}" 
                                       for s in intervention_results["suggestions"]]
                selected_intervention = st.selectbox("Select intervention to visualize:", intervention_options)
                selected_type = selected_intervention.split(":")[0]
                
                # Intensity slider
                intensity = st.slider("Implementation intensity:", 0.1, 1.0, 0.5, 0.1,
                                     help="Higher values represent more extensive implementation")
                
                # Simulate impact
                new_data = simulate_intervention_impact(area_data, selected_type, intensity)
                
                # Impact statistics
                original_avg_temp = area_data['air_temperature'].mean()
                new_avg_temp = new_data['air_temperature'].mean()
                temp_reduction = original_avg_temp - new_avg_temp
                
                st.markdown(f"""
                #### Projected Impact
                **Temperature reduction:** {temp_reduction:.2f}°C  
                **Original average temperature:** {original_avg_temp:.2f}°C  
                **New average temperature:** {new_avg_temp:.2f}°C  
                """)
                
                # Before/after visualization
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=area_data['air_temperature'], name='Before Intervention',
                                          opacity=0.75, marker=dict(color='rgba(231, 76, 60, 0.7)')))
                fig.add_trace(go.Histogram(x=new_data['air_temperature'], name='After Intervention',
                                          opacity=0.75, marker=dict(color='rgba(46, 204, 113, 0.7)')))
                
                fig.update_layout(title=f'Temperature Distribution Before & After {selected_type} Implementation',
                                 xaxis_title='Temperature (°C)', yaxis_title='Frequency',
                                 barmode='overlay', bargap=0.1, height=400)
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Placeholder content
            st.markdown("""
            ### Intervention Planning
            
            Select a location and analysis parameters, then click "Analyze & Suggest Interventions" 
            to receive customized UHI mitigation recommendations.
            
            The system will:
            1. Analyze local urban characteristics
            2. Identify key contributors to UHI
            3. Recommend targeted interventions
            4. Visualize potential impact
            """)
            
            # Example strategies
            st.markdown("### Sample Intervention Strategies")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                #### Green Infrastructure
                - Tree planting programs
                - Green roofs and walls
                - Urban parks and green spaces
                - Vegetation corridors
                
                #### Cool Materials
                - High-albedo roofing
                - Reflective pavements
                - Cool building materials
                - Permeable surfaces
                """)
            
            with col2:
                st.markdown("""
                #### Urban Design
                - Building orientation
                - Street canyon modifications
                - Air flow optimization
                - Shade structures
                
                #### Water Features
                - Fountains and spray parks
                - Retention ponds
                - Urban streams restoration
                - Blue roofs
                """)

def show_optimization(data):
    """Display the optimization and simulation module""", '₹')}</p>
                                    <p><strong>Implementation:</strong> {suggestion["implementation_time"]}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            # Show all interventions
                            for j, suggestion in enumerate(intervention_results["suggestions"]):
                                st.markdown(f"""
                                <div style="margin-bottom:15px; padding:15px; 
                                background-color:{'#e8f4f8' if j % 2 == 0 else '#f0f7fa'}; border-radius:5px;">
                                    <h4>{suggestion["type"]}: {suggestion["description"]}</h4>
                                    <p><strong>Impact:</strong> {suggestion["impact"]}</p>
                                    <p><strong>Priority:</strong> {suggestion["priority"]}</p>
                                    <p><strong>Cost:</strong> {suggestion["cost_estimate"].replace('def show_uhi_detection(data):
    """Display the UHI detection and analysis module"""
    st.markdown('<h2 class="sub-header">UHI Detection & Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module uses ground temperature measurements and urban feature analysis to detect urban heat island hotspots in Nagpur.</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Temperature Analysis", "Cluster Analysis"])
    
    with tab1:
        st.markdown("### Ground Temperature Analysis")
        st.write("Select an area to analyze or use the demo data:")
        
        # Input controls
        col1, col2 = st.columns(2)
        with col1:
            location = st.selectbox("Select location:", ["Nagpur City Center", "Dharampeth", "Sadar", "Custom Location"])
            if location == "Nagpur City Center":
                lat, lon = 21.1458, 79.0882
            elif location == "Dharampeth":
                lat, lon = 21.1350, 79.0650
            elif location == "Sadar":
                lat, lon = 21.1530, 79.0800
            else:  # Custom Location
                lat = st.number_input("Latitude:", value=21.1458, format="%.4f")
                lon = st.number_input("Longitude:", value=79.0882, format="%.4f")
        
        with col2:
            analysis_date = st.date_input("Select date for analysis:", datetime.date(2025, 6, 1))
            measurement_type = st.selectbox("Measurement type:", ["Ground Temperature", "Surface Temperature"])
        
        # Run analysis
        if st.button("Run Temperature Analysis"):
            st.markdown("#### Analysis Results")
            
            # Metrics
            cols = st.columns(3)
            with cols[0]:
                ground_temp = get_ground_temperature_data(lat, lon, analysis_date)
                st.metric("Ground Temperature", f"{ground_temp:.1f}°C", "+3.2°C")
            with cols[1]:
                surface_temp = data[data['latitude'].between(lat-0.01, lat+0.01) & 
                                   data['longitude'].between(lon-0.01, lon+0.01)]['surface_temperature'].mean()
                st.metric("Surface Temperature", f"{surface_temp:.1f}°C", "+4.5°C")
            with cols[2]:
                building_density = data[data['latitude'].between(lat-0.01, lat+0.01) & 
                                       data['longitude'].between(lon-0.01, lon+0.01)]['building_density'].mean()
                st.metric("Building Density", f"{building_density:.2f}", "+0.04")
            
            # Heat map
            st.markdown("#### Temperature Map")
            area_data = data[data['latitude'].between(lat-0.03, lat+0.03) & data['longitude'].between(lon-0.03, lon+0.03)]
            m = folium.Map(location=[lat, lon], zoom_start=14, tiles='CartoDB positron')
            heat_data = [[row['latitude'], row['longitude'], row['surface_temperature']] for _, row in area_data.iterrows()]def show_optimization(data):
    """Display the optimization and simulation module"""
    st.markdown('<h2 class="sub-header">Optimization & Simulation</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module uses optimization algorithms to find the most effective allocation of resources for UHI mitigation. It allows planners to simulate different scenarios and compare outcomes.</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Resource Optimization", "Scenario Simulation"])
    
    with tab1:
        st.markdown("### Resource Allocation Optimization")
        
        # Inputs
        budget_level = st.select_slider("Budget level:", options=["low", "medium", "high"])
        priority = st.selectbox("Optimization priority:", ["temperature", "cost", "implementation"])
        
        if st.button("Run Optimization"):
            # Get optimization results
            results = optimize_interventions(data, budget_level, priority)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Budget", f"${results['budget']}")
            with col2: st.metric("Used Budget", f"${results['used_budget']}", 
                               f"{results['used_budget']/results['budget']*100:.1f}%")
            with col3: st.metric("Temperature Reduction", f"{results['estimated_temperature_reduction']:.2f}°C")
            
            # Resource allocation chart
            st.markdown("#### Resource Allocation")
            allocation_df = pd.DataFrame(results['allocation'])
            
            fig = px.bar(allocation_df, x='cost', y='name', orientation='h', color='temperature_reduction',
                        color_continuous_scale='Blues', title='Intervention Resource Allocation',
                        labels={'cost': 'Budget Allocation ($)', 'name': 'Intervention',
                               'temperature_reduction': 'Temp. Reduction (°C)'}, text='units')
            fig.update_traces(texttemplate='%{text} units', textposition='inside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Type distribution
            type_summary = allocation_df.groupby('type').agg({
                'cost': 'sum', 'temperature_reduction': 'sum', 'units': 'sum'
            }).reset_index()
            
            fig = px.pie(type_summary, values='cost', names='type',
                        title='Budget Distribution by Intervention Type', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cost-effectiveness analysis
            st.markdown("#### Cost-Effectiveness Analysis")
            allocation_df['cost_per_degree'] = allocation_df['cost'] / allocation_df['temperature_reduction']
            cost_effectiveness = allocation_df.sort_values('cost_per_degree')
            
            fig = px.bar(cost_effectiveness, x='name', y='cost_per_degree', color='type',
                        title='Cost per Degree of Cooling ($ / °C)',
                        labels={'cost_per_degree': 'Cost per °C Reduction ($)', 'name': 'Intervention'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Implementation timeline
            st.markdown("#### Implementation Timeline")
            
            # Create simple timeline data
            timeline_data = []
            start_date = datetime.date(2025, 7, 1)
            
            for i, row in allocation_df.iterrows():
                # Assign durations based on type
                if row['type'] == 'Green Infrastructure': duration = 90
                elif row['type'] == 'High-Albedo Surfaces': duration = 60
                elif row['type'] == 'Water Features': duration = 120
                else: duration = 30
                
                end_date = start_date + datetime.timedelta(days=duration)
                timeline_data.append({
                    'Task': row['name'], 'Start': start_date, 'Finish': end_date, 'Type': row['type']
                })
                start_date = start_date + datetime.timedelta(days=30)
            
            timeline_df = pd.DataFrame(timeline_data)
            
            fig = px.timeline(timeline_df, x_start='Start', x_end='Finish', y='Task',
                             color='Type', title='Implementation Timeline')
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Future Scenario Simulation")
        
        # Inputs
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Scenario Parameters")
            year = st.slider("Target Year:", 2025, 2050, 2030, 5)
            urban_growth = st.select_slider("Urban Growth Rate:", options=["Low", "Medium", "High"])
            climate_scenario = st.select_slider("Climate Change Scenario:", 
                                               options=["Optimistic", "Moderate", "Pessimistic"])
            mitigation_level = st.select_slider("UHI Mitigation Implementation:", 
                                              options=["Minimal", "Moderate", "Aggressive"])
            simulate_button = st.button("Run Simulation")
        
        with col2:
            if simulate_button:
                st.markdown("#### Simulation Results")
                
                # Calculate climate scenario impacts
                if climate_scenario == "Optimistic":
                    climate_increase = 0.5 * (year - 2025) / 5  # 0.5°C per 5 years
                elif climate_scenario == "Moderate":
                    climate_increase = 1.0 * (year - 2025) / 5  # 1.0°C per 5 years
                else:  # Pessimistic
                    climate_increase = 1.5 * (year - 2025) / 5  # 1.5°C per 5 years
                
                # Calculate urban growth impacts
                if urban_growth == "Low":
                    growth_factor = 0.2 * (year - 2025) / 5  # 0.2°C per 5 years
                elif urban_growth == "Medium":
                    growth_factor = 0.5 * (year - 2025) / 5  # 0.5°C per 5 years
                else:  # High
                    growth_factor = 0.8 * (year - 2025) / 5  # 0.8°C per 5 years
                
                # Calculate mitigation effects
                if mitigation_level == "Minimal":
                    mitigation_effect = 0.2 * (year - 2025) / 5  # 0.2°C reduction per 5 years
                elif mitigation_level == "Moderate":
                    mitigation_effect = 0.7 * (year - 2025) / 5  # 0.7°C reduction per 5 years
                else:  # Aggressive
                    mitigation_effect = 1.2 * (year - 2025) / 5  # 1.2°C reduction per 5 years
                
                # Calculate UHI change
                current_uhi = data['air_temperature'].mean() - 25  # Assuming 25°C is the baseline
                future_uhi = current_uhi + climate_increase + growth_factor - mitigation_effect
                
                # Display results
                st.markdown(f"""
                #### Projected UHI Intensity for {year}
                
                **Current UHI Intensity (2025):** {current_uhi:.2f}°C  
                **Projected UHI Intensity ({year}):** {future_uhi:.2f}°C  
                
                **Contributing Factors:**
                - Climate change impact: +{climate_increase:.2f}°C
                - Urban growth impact: +{growth_factor:.2f}°C
                - Mitigation effect: -{mitigation_effect:.2f}°C
                
                **Net Change:** {future_uhi - current_uhi:.2f}°C
                """)
                
                # Waterfall chart
                waterfall_data = pd.DataFrame({
                    'Factor': ['Current UHI', 'Climate Change', 'Urban Growth', 'Mitigation', f'UHI in {year}'],
                    'Value': [current_uhi, climate_increase, growth_factor, -mitigation_effect, future_uhi],
                    'Type': ['Total', 'Increase', 'Increase', 'Decrease', 'Total']
                })
                
                fig = go.Figure(go.Waterfall(
                    name="UHI Components", orientation="v", measure=waterfall_data['Type'],
                    x=waterfall_data['Factor'], y=waterfall_data['Value'],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": "#e74c3c"}},
                    decreasing={"marker": {"color": "#2ecc71"}},
                    totals={"marker": {"color": "#3498db"}}
                ))
                
                fig.update_layout(title=f"UHI Intensity Change from 2025 to {year}",
                                 yaxis_title="Temperature Change (°C)", height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Temperature distribution comparison
                st.markdown("#### Temperature Distribution Comparison")
                
                # Create synthetic distributions
                current_temps = data['air_temperature'].values
                future_temps = current_temps + (future_uhi - current_uhi)
                future_temps += np.random.normal(0, 0.5, size=len(future_temps))  # Add variability
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=current_temps, name='Current (2025)', opacity=0.75,
                                          marker=dict(color='rgba(52, 152, 219, 0.7)')))
                fig.add_trace(go.Histogram(x=future_temps, name=f'Projected ({year})', opacity=0.75,
                                          marker=dict(color='rgba(231, 76, 60, 0.7)')))
                
                fig.update_layout(title=f'Temperature Distribution: Current vs {year}',
                                 xaxis_title='Temperature (°C)', yaxis_title='Frequency',
                                 barmode='overlay', bargap=0.1, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Heat wave risk assessment
                st.markdown("#### Heat Wave Risk Assessment")
                
                # Determine risk level
                if future_uhi > 5:
                    risk_level, risk_color = "High", "#e74c3c"
                elif future_uhi > 3:
                    risk_level, risk_color = "Medium", "#f39c12"
                else:
                    risk_level, risk_color = "Low", "#2ecc71"
                
                # Calculate additional metrics
                extreme_heat_days_current = sum(current_temps > 30) / len(current_temps) * 365
                extreme_heat_days_future = sum(future_temps > 30) / len(future_temps) * 365
                
                st.markdown(f"""
                <div style="padding:20px; background-color:{risk_color}25; border-left:5px solid {risk_color}; margin-bottom:20px;">
                    <h4>Heat Wave Risk Level: <span style="color:{risk_color}">{risk_level}</span></h4>
                    <p>
                        <strong>Days over 30°C per year:</strong><br>
                        Current (2025): {extreme_heat_days_current:.1f} days<br>
                        Projected ({year}): {extreme_heat_days_future:.1f} days<br>
                        <strong>Increase: {extreme_heat_days_future - extreme_heat_days_current:.1f} days</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("#### Recommendations")
                
                if risk_level == "High":
                    st.markdown("""
                    To address the high heat wave risk projected in this scenario:
                    
                    1. **Implement comprehensive UHI mitigation plan** with emphasis on cooling interventions
                    2. **Develop heat emergency response protocols** for vulnerable populations
                    3. **Increase green infrastructure budget** to maximize cooling effect
                    4. **Revise building codes** to mandate cool roofs and energy-efficient designs
                    5. **Create cooling centers network** accessible within 10-minute walks citywide
                    """)
                elif risk_level == "Medium":
                    st.markdown("""
                    To address the medium heat wave risk projected in this scenario:
                    
                    1. **Gradually increase green cover** in hotspot areas
                    2. **Implement cool pavement program** during regular maintenance cycles
                    3. **Develop targeted interventions** for vulnerable neighborhoods
                    4. **Create incentives for green roofs** and cool building materials
                    5. **Monitor temperature trends** and adjust strategies accordingly
                    """)
                else:
                    st.markdown("""
                    To maintain the low heat wave risk projected in this scenario:
                    
                    1. **Continue current mitigation efforts** to maintain progress
                    2. **Preserve existing green spaces** and expand when possible
                    3. **Incorporate UHI considerations** in all future development
                    4. **Monitor temperature data** to detect any unexpected changes
                    5. **Document successful strategies** to share with other cities
                    """)
            else:
                # Placeholder content
                st.markdown("""
                ### Scenario Simulation
                
                Configure the parameters on the left and click "Run Simulation" to see projections of future UHI patterns
                based on different climate change, urban growth, and mitigation scenarios.
                
                The simulation will show:
                - Projected UHI intensity changes
                - Temperature distribution shifts
                - Heat wave risk assessment
                - Tailored recommendations based on outcomes
                """)
                
                # Sample projection chart
                years = list(range(2025, 2051, 5))
                no_action = [3.0 + 0.4 * i for i in range(len(years))]
                moderate_action = [3.0 + 0.3 * i - 0.1 * i**2 for i in range(len(years))]
                aggressive_action = [3.0 + 0.2 * i - 0.15 * i**2 for i in range(len(years))]
                
                scenario_df = pd.DataFrame({
                    'Year': years * 3,
                    'UHI Intensity (°C)': no_action + moderate_action + aggressive_action,
                    'Scenario': ['No Action'] * len(years) + ['Moderate Action'] * len(years) + 
                                ['Aggressive Action'] * len(years)
                })
                
                fig = px.line(scenario_df, x='Year', y='UHI Intensity (°C)', color='Scenario',
                             title='UHI Intensity Projections by Mitigation Scenario')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

def show_about():
    """Display the about page with project information"""
    st.markdown('<h2 class="sub-header">About This Project</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>Urban Heat Island Analysis & Mitigation System</h3>
        <p>This project aims to develop an integrated AI-based system that helps city planners and environmental 
        scientists detect, analyze, and mitigate urban heat island effects through data-driven decision making.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key features
    st.markdown("### Key Features")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Detection & Analysis
        - Satellite imagery analysis for UHI detection
        - Street-level temperature mapping
        - Temporal and spatial pattern identification
        - Cluster analysis of similar urban areas
        
        #### Prediction
        - Machine learning models for temperature prediction
        - Impact assessment of new developments
        - Future scenario simulation
        - Climate change integration
        """)
    
    with col2:
        st.markdown("""
        #### Intervention Planning
        - Customized intervention recommendations
        - Cost-benefit analysis of strategies
        - Implementation priority ranking
        - Visualization of potential impacts
        
        #### Optimization
        - Resource allocation optimization
        - Multi-objective decision support
        - Budget-constrained planning
        - Scenario comparison
        """)
    
    # Technical details
    st.markdown("### Technical Details")
    st.markdown("""
    This system integrates multiple technologies and data sources:
    
    - **Satellite Data**: Utilizes freely available Landsat, Sentinel-2, and MODIS data
    - **Machine Learning**: Employs random forest and gradient boosting models
    - **Optimization Algorithms**: Uses multi-objective optimization for planning
    - **GIS Integration**: Provides spatial analysis and mapping capabilities
    - **Simulation Models**: Enables scenario testing and future projections
    
    The application is built using Python and Streamlit, making it accessible through any web browser.
    No proprietary software or paid services are required to run the system.
    """)
    
    # Data sources
    st.markdown("### Data Sources")
    st.markdown("""
    The system can utilize data from various free sources:
    
    - NASA Earth Data (https://earthdata.nasa.gov/)
    - USGS Earth Explorer (https://earthexplorer.usgs.gov/)
    - Copernicus Open Access Hub (https://scihub.copernicus.eu/)
    - OpenStreetMap (https://www.openstreetmap.org/)
    - National Weather Service (https://www.weather.gov/)
    - Local municipal GIS data portals
    
    For demonstration purposes, this app uses synthetic data that simulates realistic urban temperature patterns.
    """)

# ---------- MAIN APPLICATION ----------
def main():
    """Main application entry point"""
    # Setup page
    setup_page()
    
    # Page header
    st.markdown('<h1 class="main-header">Urban Heat Island Analysis & Mitigation System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
        This AI-based system helps city planners and environmental scientists analyze urban heat island (UHI) effects 
        and develop data-driven strategies to mitigate their impact. Using satellite imagery, environmental data, 
        and machine learning, it provides insights for sustainable urban planning.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.image("https://www.epa.gov/sites/default/files/styles/medium/public/2020-07/urban-heat-island.jpg", 
                    use_container_width=True)
    st.sidebar.title("Navigation")
    
    # Page selection
    page = st.sidebar.radio("Select a Module", 
                           ["Dashboard", "UHI Detection", "Temperature Prediction", 
                            "Intervention Planning", "Optimization & Simulation", "About"])
    
    # Load sample data
    data = load_sample_data()
    
    # Display selected page
    if page == "Dashboard": show_dashboard(data)
    elif page == "UHI Detection": show_uhi_detection(data)
    elif page == "Temperature Prediction": show_temperature_prediction(data)
    elif page == "Intervention Planning": show_intervention_planning(data)
    elif page == "Optimization & Simulation": show_optimization(data)
    elif page == "About": show_about()
    
    # Footer
    st.markdown('<div class="footer">Urban Heat Island Analysis & Mitigation System © 2025</div>', 
               unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()"""
Urban Heat Island Analysis & Mitigation System - Streamlined Version
Author: Claude AI
Date: June 2025
"""

import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ---------- PAGE SETUP AND CONFIGURATION ----------
def setup_page():
    st.set_page_config(page_title="UHI Analysis System", page_icon="🌆", layout="wide")
    st.markdown("""
    <style>
        .main-header {font-size:2.5rem; color:#2c3e50; text-align:center; margin-bottom:1rem;}
        .sub-header {font-size:1.8rem; color:#34495e; margin-top:2rem; margin-bottom:1rem;}
        .card {background-color:#f8f9fa; border-radius:5px; padding:20px; margin-bottom:20px; box-shadow:0 4px 6px rgba(0,0,0,0.1);}
        .highlight {background-color:#e8f4f8; padding:10px; border-left:5px solid #3498db; margin-bottom:15px;}
        .footer {text-align:center; margin-top:3rem; color:#7f8c8d; font-size:0.8rem;}
        .metric-container {background-color:#f8f9fa; border-radius:5px; padding:15px; margin:10px 0; box-shadow:0 2px 4px rgba(0,0,0,0.1);}
        .metric-label {font-size:1rem; color:#7f8c8d;}
        .metric-value {font-size:1.8rem; font-weight:bold; color:#2c3e50;}
    </style>
    """, unsafe_allow_html=True)

# ---------- DATA GENERATION AND ANALYSIS FUNCTIONS ----------
def load_sample_data():
    """Generate synthetic UHI data for demonstration"""
    np.random.seed(42)
    lat_center, lon_center = 21.1458, 79.0882  # Nagpur, Maharashtra coordinates
    n_points = 500
    
    # Generate locations and urban features
    lats = lat_center + np.random.normal(0, 0.03, n_points)
    lons = lon_center + np.random.normal(0, 0.03, n_points)
    building_density = np.random.beta(2, 2, n_points)
    vegetation_index = 1 - np.random.beta(2, 1.5, n_points) * building_density
    albedo = np.random.beta(2, 5, n_points) * (1 - vegetation_index * 0.5)
    
    # Calculate distances and normalize
    dist_from_center = np.sqrt((lats - lat_center)**2 + (lons - lon_center)**2)
    dist_normalized = (dist_from_center - dist_from_center.min()) / (dist_from_center.max() - dist_from_center.min())
    
    # Calculate temperatures based on urban features
    base_temp = 32  # Higher base temperature for Nagpur
    uhi_effect = 5 * building_density - 3 * vegetation_index - 2 * albedo - 1 * dist_normalized
    temperature = base_temp + uhi_effect + np.random.normal(0, 0.5, n_points)
    surface_temp = temperature + 2 + 4 * building_density - 3 * vegetation_index + np.random.normal(0, 0.7, n_points)
    
    # Assign land use categories
    land_use = np.random.choice(
        ['Commercial', 'Residential', 'Industrial', 'Park', 'Water Body'],
        n_points, 
        p=[0.3, 0.4, 0.1, 0.15, 0.05]
    )
    
    # Create and return dataframe
    return pd.DataFrame({
        'latitude': lats, 'longitude': lons, 'building_density': building_density,
        'vegetation_index': vegetation_index, 'albedo': albedo, 'air_temperature': temperature,
        'surface_temperature': surface_temp, 'land_use': land_use, 'distance_from_center': dist_from_center
    })

def get_satellite_ndvi_data(lat, lon, date=None):
    """Mock function to simulate fetching NDVI data from satellite imagery"""
    np.random.seed(int(lat*100 + lon*100))
    lat_center, lon_center = 40.7128, -74.0060
    dist = np.sqrt((lat - lat_center)**2 + (lon - lon_center)**2)
    normalized_dist = min(dist / 0.1, 1)  # 0.1 is max_dist
    ndvi_value = -0.1 + normalized_dist * 0.7 + np.random.normal(0, 0.05)
    return max(-1, min(1, ndvi_value))  # Clamp to valid range

def get_temperature_prediction(features):
    """Predict temperature based on urban features"""
    return (25 + 5 * features['building_density'] - 4 * features['vegetation_index'] - 
            3 * features['albedo'] + np.random.normal(0, 0.2))

def create_cluster_map(data, n_clusters=5):
    """Create clusters of similar urban areas based on UHI characteristics"""
    # Prepare and cluster data
    features = data[['building_density', 'vegetation_index', 'albedo', 'air_temperature']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(scaled_features)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Create map
    colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F3']
    m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], 
                  zoom_start=12, tiles='CartoDB positron')
    
    # Add markers and legend
    for idx, row in data.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']], radius=5,
            color=colors[row['cluster']], fill=True, fill_color=colors[row['cluster']], fill_opacity=0.7,
            popup=f"Cluster: {row['cluster']}<br>Temp: {row['air_temperature']:.1f}°C<br>" +
                  f"Building Density: {row['building_density']:.2f}<br>" +
                  f"Vegetation: {row['vegetation_index']:.2f}<br>Land Use: {row['land_use']}"
        ).add_to(m)
    
    # Create legend
    legend_html = '''
    <div style="position:fixed; bottom:50px; right:50px; width:150px; height:160px; 
    border:2px solid grey; z-index:9999; font-size:14px; background-color:white;
    padding:10px; border-radius:5px;"><span style="font-weight:bold;">Clusters</span><br>
    '''
    
    for i, center in enumerate(centers):
        if center[0] > 0.6: cluster_desc = "Urban Core"
        elif center[1] > 0.6: cluster_desc = "Green Zone"
        elif center[3] > 27: cluster_desc = "Hot Spot"
        elif center[2] > 0.4: cluster_desc = "Reflective Area"
        else: cluster_desc = f"Cluster {i}"
        legend_html += f'<div style="background-color:{colors[i]}; width:20px; height:20px; display:inline-block;"></div> {cluster_desc}<br>'
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m, data

def suggest_interventions(data, location):
    """Suggest UHI mitigation strategies based on analysis"""
    # Get neighborhood data
    lat, lon = location
    distances = np.sqrt((data['latitude'] - lat)**2 + (data['longitude'] - lon)**2)
    neighborhood_data = data[distances < 0.01]  # ~1km radius
    
    if len(neighborhood_data) == 0:
        return {"message": "No data available for this location.", "suggestions": []}
    
    # Calculate neighborhood averages
    avg_temp = neighborhood_data['air_temperature'].mean()
    avg_building_density = neighborhood_data['building_density'].mean()
    avg_vegetation = neighborhood_data['vegetation_index'].mean()
    avg_albedo = neighborhood_data['albedo'].mean()
    suggestions = []
    
    # Generate suggestions based on conditions
    if avg_vegetation < 0.3:
        suggestions.append({
            "type": "Green Infrastructure", "priority": "High", "score": 5,
            "description": "Increase urban vegetation through tree planting, green roofs, or pocket parks.",
            "impact": f"Could reduce local temperature by {1.5 + np.random.uniform(0, 0.5):.1f}°C",
            "cost_estimate": "$$", "implementation_time": "Medium-term"
        })
    elif avg_vegetation < 0.5:
        suggestions.append({
            "type": "Green Infrastructure", "priority": "Medium", "score": 3,
            "description": "Enhance existing green spaces and add vegetation to streets.",
            "impact": f"Could reduce local temperature by {0.8 + np.random.uniform(0, 0.3):.1f}°C",
            "cost_estimate": "$", "implementation_time": "Short-term"
        })
    
    if avg_albedo < 0.2:
        suggestions.append({
            "type": "High-Albedo Surfaces", "priority": "High", "score": 5,
            "description": "Implement cool roofs and pavements to reflect more solar radiation.",
            "impact": f"Could reduce local temperature by {1.2 + np.random.uniform(0, 0.4):.1f}°C",
            "cost_estimate": "$$", "implementation_time": "Medium-term"
        })
    elif avg_albedo < 0.4:
        suggestions.append({
            "type": "High-Albedo Surfaces", "priority": "Medium", "score": 3,
            "description": "Gradually replace dark surfaces with lighter materials.",
            "impact": f"Could reduce local temperature by {0.7 + np.random.uniform(0, 0.3):.1f}°C",
            "cost_estimate": "$", "implementation_time": "Ongoing"
        })
    
    if avg_building_density > 0.7:
        suggestions.append({
            "type": "Urban Design", "priority": "High", "score": 4,
            "description": "Modify building arrangements to improve air flow and reduce heat trapping.",
            "impact": f"Could reduce local temperature by {1.0 + np.random.uniform(0, 0.5):.1f}°C",
            "cost_estimate": "$$$", "implementation_time": "Long-term"
        })
    elif avg_building_density > 0.5:
        suggestions.append({
            "type": "Urban Design", "priority": "Medium", "score": 2,
            "description": "Consider height variations in future developments to enhance ventilation.",
            "impact": f"Could reduce local temperature by {0.5 + np.random.uniform(0, 0.3):.1f}°C",
            "cost_estimate": "$$", "implementation_time": "Long-term"
        })
    
    if avg_temp > 28:
        suggestions.append({
            "type": "Water Features", "priority": "Medium", "score": 3,
            "description": "Incorporate water elements like fountains or retention ponds for cooling.",
            "impact": f"Could reduce local temperature by {0.8 + np.random.uniform(0, 0.3):.1f}°C",
            "cost_estimate": "$$", "implementation_time": "Medium-term"
        })
    
    # Sort suggestions by priority
    suggestions.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "message": "Analysis complete. Interventions suggested based on local conditions.",
        "local_temperature": f"{avg_temp:.1f}°C",
        "building_density": f"{avg_building_density:.2f}",
        "vegetation_index": f"{avg_vegetation:.2f}",
        "albedo": f"{avg_albedo:.2f}",
        "suggestions": suggestions
    }

def simulate_intervention_impact(current_data, intervention_type, intensity=0.5):
    """Simulate the impact of implementing a specific intervention"""
    new_data = current_data.copy()
    
    # Apply intervention effects
    if intervention_type == "Green Infrastructure":
        new_data['vegetation_index'] = new_data['vegetation_index'] + (1 - new_data['vegetation_index']) * intensity * 0.5
        new_data['vegetation_index'] = new_data['vegetation_index'].clip(upper=1.0)
    elif intervention_type == "High-Albedo Surfaces":
        new_data['albedo'] = new_data['albedo'] + (1 - new_data['albedo']) * intensity * 0.6
        new_data['albedo'] = new_data['albedo'].clip(upper=1.0)
    elif intervention_type == "Urban Design":
        new_data['building_density'] = new_data['building_density'] * (1 - intensity * 0.3)
    elif intervention_type == "Water Features":
        lat_center, lon_center = new_data['latitude'].mean(), new_data['longitude'].mean()
        distances = np.sqrt((new_data['latitude'] - lat_center)**2 + (new_data['longitude'] - lon_center)**2)
        max_dist = distances.max()
        cooling_effect = intensity * 0.5 * (1 - distances/max_dist)
        new_data['vegetation_index'] = new_data['vegetation_index'] + cooling_effect * 0.3
        new_data['vegetation_index'] = new_data['vegetation_index'].clip(upper=1.0)
    
    # Recalculate temperatures
    base_temp = 25
    uhi_effect = (5 * new_data['building_density'] - 3 * new_data['vegetation_index'] - 
                 2 * new_data['albedo'] - 1 * new_data['distance_from_center'].clip(0, 1))
    np.random.seed(42)
    new_data['air_temperature'] = base_temp + uhi_effect + np.random.normal(0, 0.2, len(new_data))
    
    return new_data

def optimize_interventions(data, budget_level='medium', priority='temperature'):
    """Optimize intervention strategy based on budget constraints and priorities"""
    # Define budget levels in Indian Rupees (₹)
    budget_map = {'low': 2000000, 'medium': 5000000, 'high': 10000000}  # ₹20 lakh, ₹50 lakh, ₹1 crore
    budget = budget_map.get(budget_level, 5000000)
    
    # Define intervention options with costs in INR
    interventions = [
        {'name': 'Tree Planting', 'type': 'Green Infrastructure', 'cost_per_unit': 200000, 
         'temp_reduction_per_unit': 0.05, 'max_units': 30},
        {'name': 'Cool Roofs', 'type': 'High-Albedo Surfaces', 'cost_per_unit': 300000, 
         'temp_reduction_per_unit': 0.08, 'max_units': 20},
        {'name': 'Cool Pavements', 'type': 'High-Albedo Surfaces', 'cost_per_unit': 400000, 
         'temp_reduction_per_unit': 0.06, 'max_units': 15},
        {'name': 'Green Roofs', 'type': 'Green Infrastructure', 'cost_per_unit': 500000, 
         'temp_reduction_per_unit': 0.1, 'max_units': 12},
        {'name': 'Water Features', 'type': 'Water Features', 'cost_per_unit': 600000, 
         'temp_reduction_per_unit': 0.12, 'max_units': 8}
    ]
    
    # Sort interventions by priority
    if priority == 'temperature':
        interventions.sort(key=lambda x: x['temp_reduction_per_unit'] / x['cost_per_unit'], reverse=True)
    elif priority == 'cost':
        interventions.sort(key=lambda x: x['cost_per_unit'])
    elif priority == 'implementation':
        interventions.sort(key=lambda x: x['cost_per_unit'])
    
    # Allocate budget using greedy algorithm
    allocation, remaining_budget = [], budget
    for intervention in interventions:
        affordable_units = min(intervention['max_units'], int(remaining_budget / intervention['cost_per_unit']))
        if affordable_units > 0:
            cost = affordable_units * intervention['cost_per_unit']
            temp_reduction = affordable_units * intervention['temp_reduction_per_unit']
            allocation.append({
                'name': intervention['name'], 'type': intervention['type'], 'units': affordable_units,
                'cost': cost, 'temperature_reduction': temp_reduction
            })
            remaining_budget -= cost
    
    # Calculate totals
    total_cost = sum(item['cost'] for item in allocation)
    total_reduction = sum(item['temperature_reduction'] for item in allocation)
    
    return {
        'budget': budget, 'used_budget': total_cost, 'remaining_budget': remaining_budget,
        'estimated_temperature_reduction': total_reduction, 'allocation': allocation
    }

# ---------- UI MODULES ----------
def show_dashboard(data):
    """Display the main dashboard with overview metrics and visualizations"""
    st.markdown('<h2 class="sub-header">UHI Dashboard</h2>', unsafe_allow_html=True)
    
    # Summary metrics
    cols = st.columns(4)
    with cols[0]: st.markdown(f'<div class="metric-container"><div class="metric-label">Average Temperature</div><div class="metric-value">{data["air_temperature"].mean():.1f}°C</div></div>', unsafe_allow_html=True)
    with cols[1]: st.markdown(f'<div class="metric-container"><div class="metric-label">Max Temperature</div><div class="metric-value">{data["air_temperature"].max():.1f}°C</div></div>', unsafe_allow_html=True)
    with cols[2]: st.markdown(f'<div class="metric-container"><div class="metric-label">Average Vegetation Index</div><div class="metric-value">{data["vegetation_index"].mean():.2f}</div></div>', unsafe_allow_html=True)
    with cols[3]: st.markdown(f'<div class="metric-container"><div class="metric-label">Temperature Anomaly</div><div class="metric-value">+{data["air_temperature"].max() - data["air_temperature"].min():.1f}°C</div></div>', unsafe_allow_html=True)
    
    st.selectbox("Select City", ["Nagpur, Maharashtra (Demo)"], index=0)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Heat Map", "Analysis", "Trends"])
    
    with tab1:
        st.markdown("### Urban Heat Map")
        st.write("This heat map shows the temperature distribution across Nagpur.")
        
        # Create heat map
        m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=12, tiles='CartoDB positron')
        heat_data = [[row['latitude'], row['longitude'], row['air_temperature']] for _, row in data.iterrows()]
        HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'}).add_to(m)
        folium_static(m)
    
    with tab2:
        st.markdown("### UHI Analysis")
        
        # Create charts
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(data, x='building_density', y='air_temperature', color='vegetation_index',
                            color_continuous_scale='Viridis', title='Temperature vs Building Density',
                            labels={'building_density': 'Building Density', 'air_temperature': 'Temperature (°C)',
                                   'vegetation_index': 'Vegetation Index'}, size_max=10, hover_data=['land_use'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            land_use_temp = data.groupby('land_use')['air_temperature'].mean().reset_index()
            fig = px.bar(land_use_temp, x='land_use', y='air_temperature', color='air_temperature',
                        color_continuous_scale='Thermal', title='Average Temperature by Land Use',
                        labels={'land_use': 'Land Use Type', 'air_temperature': 'Avg Temperature (°C)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.markdown("### Feature Correlations")
        corr_features = ['building_density', 'vegetation_index', 'albedo', 'air_temperature', 'surface_temperature']
        corr_matrix = data[corr_features].corr()
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                       title='Correlation Between Urban Features',
                       labels=dict(x='Features', y='Features', color='Correlation'))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Temperature Trends")
        
        # Generate temporal data (for demo)
        dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
        avg_temps = [data['air_temperature'].mean() + np.sin(i/5) + np.random.normal(0, 0.3) for i in range(30)]
        max_temps = [t + 2 + np.random.normal(0, 0.2) for t in avg_temps]
        time_df = pd.DataFrame({'date': dates, 'avg_temperature': avg_temps, 'max_temperature': max_temps})
        
        # Plot time series
        fig = px.line(time_df, x='date', y=['avg_temperature', 'max_temperature'],
                     title='Temperature Trends Over Time',
                     labels={'date': 'Date', 'value': 'Temperature (°C)', 'variable': 'Metric'})
        fig.update_layout(legend_title_text='')
        st.plotly_chart(fig, use_container_width=True)
        
        # Temperature forecast
        st.markdown("### Temperature Forecast")
        forecast_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=7, freq='D')
        forecast_temps = [avg_temps[-1] + 0.1*i + np.random.normal(0, 0.2) for i in range(7)]
        full_dates = dates.append(forecast_dates)
        full_temps = avg_temps + forecast_temps
        forecast_df = pd.DataFrame({
            'date': full_dates,
            'temperature': full_temps,
            'type': ['Historical']*30 + ['Forecast']*7
        })
        
        fig = px.line(forecast_df, x='date', y='temperature', color='type',
                     title='7-Day Temperature Forecast',
                     labels={'date': 'Date', 'temperature': 'Avg. Temperature (°C)'})
        st.plotly_chart(fig, use_container_width=True)

def show_uhi_detection(data):
    """Display the UHI detection and analysis module"""
    st.markdown('<h2 class="sub-header">UHI Detection & Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module uses satellite imagery and street-level data to detect urban heat island hotspots. Upload your own data or use our demo data to visualize UHI patterns.</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Satellite Analysis", "Cluster Analysis", "Temporal Analysis"])
    
    with tab1:
        st.markdown("### Satellite-Based UHI Detection")
        st.write("Select an area to analyze or use the demo data:")
        
        # Input controls
        col1, col2 = st.columns(2)
        with col1:
            location = st.selectbox("Select location:", ["New York City, NY", "Custom Location"])
            lat, lon = (40.7128, -74.0060) if location == "New York City, NY" else (
                st.number_input("Latitude:", value=40.7128, format="%.4f"),
                st.number_input("Longitude:", value=-74.0060, format="%.4f")
            )
        
        with col2:
            analysis_date = st.date_input("Select date for analysis:", datetime.date(2025, 6, 1))
            data_source = st.selectbox("Data source:", ["Landsat 9", "Sentinel-2", "MODIS"])
        
        # Run analysis
        if st.button("Run Satellite Analysis"):
            st.markdown("#### Analysis Results")
            
            # Metrics
            cols = st.columns(3)
            with cols[0]:
                ndvi = get_satellite_ndvi_data(lat, lon, analysis_date)
                st.metric("NDVI Index", f"{ndvi:.2f}", "-0.05")
            with cols[1]:
                surface_temp = data[data['latitude'].between(lat-0.01, lat+0.01) & 
                                    data['longitude'].between(lon-0.01, lon+0.01)]['surface_temperature'].mean()
                st.metric("Surface Temperature", f"{surface_temp:.1f}°C", "+2.3°C")
            with cols[2]:
                building_density = data[data['latitude'].between(lat-0.01, lat+0.01) & 
                                        data['longitude'].between(lon-0.01, lon+0.01)]['building_density'].mean()
                st.metric("Building Density", f"{building_density:.2f}", "+0.04")
            
            # Heat map
            st.markdown("#### Surface Temperature Map")
            area_data = data[data['latitude'].between(lat-0.03, lat+0.03) & data['longitude'].between(lon-0.03, lon+0.03)]
            m = folium.Map(location=[lat, lon], zoom_start=14, tiles='CartoDB positron')
            heat_data = [[row['latitude'], row['longitude'], row['surface_temperature']] for _, row in area_data.iterrows()]
            HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'},
                   min_opacity=0.5, blur=10).add_to(m)
            folium.Marker([lat, lon], popup=f"Selected Location<br>NDVI: {ndvi:.2f}<br>Temp: {surface_temp:.1f}°C",
                         icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
            folium_static(m)
            
            # NDVI vs Temperature
            st.markdown("#### NDVI vs Surface Temperature")
            fig = px.scatter(area_data, x='vegetation_index', y='surface_temperature', color='surface_temperature',
                            color_continuous_scale='Thermal', title='Vegetation Index vs Surface Temperature',
                            labels={'vegetation_index': 'Vegetation Index (NDVI)', 
                                   'surface_temperature': 'Surface Temperature (°C)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis summary
            st.markdown("#### Analysis Summary")
            avg_temp = area_data['surface_temperature'].mean()
            uhi_severity = "Severe" if avg_temp > 30 else "Moderate" if avg_temp > 27 else "Low"
            impact = "High" if avg_temp > 30 else "Medium" if avg_temp > 27 else "Low"
            
            st.markdown(f"""
            **UHI Severity:** {uhi_severity}  
            **Potential Impact:** {impact}  
            **Key Factors:**
            - Building density contributes approximately {(building_density * 100):.1f}% to the UHI effect
            - Vegetation cover is {(area_data['vegetation_index'].mean() * 100):.1f}% of the analyzed area
            - Average surface temperature is {avg_temp:.1f}°C, which is {avg_temp - 25:.1f}°C above the baseline temperature
            """)
    
    with tab2:
        st.markdown("### UHI Cluster Analysis")
        st.write("This analysis identifies similar urban areas based on their heat characteristics.")
        
        # Clustering
        n_clusters = st.slider("Number of clusters:", min_value=3, max_value=7, value=5)
        cluster_map, clustered_data = create_cluster_map(data, n_clusters)
        
        st.markdown("#### Urban Heat Clusters")
        folium_static(cluster_map)
        
        # Cluster characteristics
        st.markdown("#### Cluster Characteristics")
        cluster_means = clustered_data.groupby('cluster')[
            ['building_density', 'vegetation_index', 'albedo', 'air_temperature']
        ].mean().reset_index()
        
        # Parallel coordinates plot
        fig = px.parallel_coordinates(cluster_means,
                                     dimensions=['building_density', 'vegetation_index', 'albedo', 'air_temperature'],
                                     color='cluster',
                                     labels={'building_density': 'Building Density', 'vegetation_index': 'Vegetation',
                                            'albedo': 'Albedo', 'air_temperature': 'Temperature (°C)'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster summary
        st.markdown("#### Cluster Summary")
        cluster_descriptions = []
        # Cluster summary
        st.markdown("#### Cluster Summary")
        cluster_descriptions = []
        for _, row in cluster_means.iterrows():
            if row['air_temperature'] > 28 and row['building_density'] > 0.6:
                description = "Urban Core - Hot Spot"
            elif row['vegetation_index'] > 0.6:
                description = "Green Zone - Cool Area"
            elif row['albedo'] > 0.4:
                description = "High Reflectivity Zone"
            elif row['building_density'] > 0.5 and row['vegetation_index'] < 0.3:
                description = "Dense Urban - Moderate Heat"
            else:
                description = "Mixed Urban Zone"
            cluster_descriptions.append(description)
        
        # Display cluster summary
        cluster_means['description'] = cluster_descriptions
        display_df = cluster_means.copy()
        display_df.columns = ['Cluster', 'Building Density', 'Vegetation', 'Albedo', 'Temperature (°C)', 'Description']
        display_df = display_df[['Cluster', 'Description', 'Temperature (°C)', 'Building Density', 'Vegetation', 'Albedo']]
        st.dataframe(display_df.round(2))
    
    with tab3:
        st.markdown("### Temporal UHI Analysis")
        st.write("Analyze how UHI patterns change over time.")
        
        # Time period selection
        period = st.selectbox("Select analysis period:", ["Daily Cycle", "Seasonal Variation", "Annual Trend"])
        
        if period == "Daily Cycle":
            # Generate data for daily cycle
            hours = list(range(24))
            urban_temps = [25 + 5 * np.sin((h - 2) * np.pi / 24) for h in hours]
            rural_temps = [22 + 4 * np.sin((h - 2) * np.pi / 24) for h in hours]
            daily_df = pd.DataFrame({
                'hour': hours,
                'urban_temperature': urban_temps,
                'rural_temperature': rural_temps,
                'uhi_intensity': [u - r for u, r in zip(urban_temps, rural_temps)]
            })
            
            # Temperature comparison
            fig = px.line(daily_df, x='hour', y=['urban_temperature', 'rural_temperature'],
                         title='Daily Temperature Cycle: Urban vs Rural',
                         labels={'hour': 'Hour of Day', 'value': 'Temperature (°C)', 'variable': 'Location'})
            st.plotly_chart(fig, use_container_width=True)
            
            # UHI intensity
            fig = px.bar(daily_df, x='hour', y='uhi_intensity', title='Urban Heat Island Intensity Throughout the Day',
                        labels={'hour': 'Hour of Day', 'uhi_intensity': 'UHI Intensity (°C)'},
                        color='uhi_intensity', color_continuous_scale='Thermal')
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis
            max_uhi_hour = daily_df.loc[daily_df['uhi_intensity'].idxmax(), 'hour']
            max_uhi = daily_df['uhi_intensity'].max()
            
            st.markdown(f"""
            #### Key Findings:
            - Maximum UHI intensity of {max_uhi:.1f}°C occurs at {int(max_uhi_hour):02d}:00 hours
            - UHI effect is strongest during night and early morning hours
            - Minimum temperature difference observed during mid-day
            
            This pattern is typical of urban areas where built surfaces release stored heat during the night,
            while rural areas cool more rapidly after sunset.
            """)
        
        elif period == "Seasonal Variation":
            # Generate data for seasonal variation
            months = list(range(1, 13))
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            summer_peak = 7  # July
            urban_temps = [25 + 10 * np.sin((m - 1) * np.pi / 6) for m in months]
            rural_temps = [22 + 12 * np.sin((m - 1) * np.pi / 6) for m in months]
            uhi_intensity = [3 + 1.5 * np.sin((m - summer_peak) * np.pi / 6) for m in months]
            
            seasonal_df = pd.DataFrame({
                'month': month_names, 'month_num': months,
                'urban_temperature': urban_temps, 'rural_temperature': rural_temps,
                'uhi_intensity': uhi_intensity
            })
            
            # Temperature comparison
            fig = px.line(seasonal_df, x='month', y=['urban_temperature', 'rural_temperature'],
                         title='Seasonal Temperature Variation: Urban vs Rural',
                         labels={'month': 'Month', 'value': 'Temperature (°C)', 'variable': 'Location'})
            fig.update_xaxes(categoryorder='array', categoryarray=month_names)
            st.plotly_chart(fig, use_container_width=True)
            
            # UHI intensity
            fig = px.bar(seasonal_df, x='month', y='uhi_intensity', title='Urban Heat Island Intensity by Season',
                        labels={'month': 'Month', 'uhi_intensity': 'UHI Intensity (°C)'},
                        color='uhi_intensity', color_continuous_scale='Thermal')
            fig.update_xaxes(categoryorder='array', categoryarray=month_names)
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis
            max_uhi_month = seasonal_df.loc[seasonal_df['uhi_intensity'].idxmax(), 'month']
            max_uhi = seasonal_df['uhi_intensity'].max()
            
            st.markdown(f"""
            #### Key Findings:
            - Maximum UHI intensity of {max_uhi:.1f}°C occurs in {max_uhi_month}
            - Summer months generally show higher UHI intensity
            - Urban and rural temperature gap varies by season
            """)
        
        elif period == "Annual Trend":
            # Generate data for annual trend
            years = list(range(2020, 2026))
            base_uhi = [2.8, 3.0, 3.3, 3.5, 3.7, 4.0]
            uhi_trend = [b + np.random.normal(0, 0.1) for b in base_uhi]
            
            trend_df = pd.DataFrame({'year': years, 'uhi_intensity': uhi_trend})
            
            # Plot trend
            fig = px.line(trend_df, x='year', y='uhi_intensity',
                         title='Urban Heat Island Intensity Annual Trend',
                         labels={'year': 'Year', 'uhi_intensity': 'Average UHI Intensity (°C)'},
                         markers=True)
            
            # Add trendline
            fig.add_trace(px.scatter(trend_df, x='year', y='uhi_intensity', trendline='ols').data[1])
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis
            initial_uhi = trend_df.iloc[0]['uhi_intensity']
            final_uhi = trend_df.iloc[-1]['uhi_intensity']
            percent_change = ((final_uhi - initial_uhi) / initial_uhi) * 100
            
            st.markdown(f"""
            #### Key Findings:
            - UHI intensity has increased by {percent_change:.1f}% over the past 6 years
            - Average annual increase of {(final_uhi - initial_uhi) / 5:.2f}°C per year
            - If this trend continues, UHI intensity could reach {final_uhi + (final_uhi - initial_uhi) / 5 * 5:.1f}°C by 2030
            """)

def show_temperature_prediction(data):
    """Display the temperature prediction module"""
    st.markdown('<h2 class="sub-header">Temperature Prediction Model</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module uses machine learning to predict hyperlocal temperature variations based on urban features. Adjust the parameters to see how different urban configurations affect local temperatures.</div>', unsafe_allow_html=True)
    
    # Create layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Parameter inputs
        st.markdown("### Urban Feature Parameters")
        building_density = st.slider("Building Density:", 0.0, 1.0, 0.5, 0.01,
                                   help="Higher values indicate more dense urban development")
        vegetation_index = st.slider("Vegetation Index:", 0.0, 1.0, 0.3, 0.01,
                                   help="Higher values indicate more vegetation (trees, parks, etc.)")
        albedo = st.slider("Surface Albedo:", 0.0, 1.0, 0.2, 0.01,
                          help="Higher values indicate more reflective surfaces")
        
        st.markdown("### Additional Factors")
        land_use = st.selectbox("Land Use Type:", ["Residential", "Commercial", "Industrial", "Park", "Mixed Use"])
        time_of_day = st.select_slider("Time of Day:", 
                                      options=["Early Morning", "Morning", "Noon", "Afternoon", "Evening", "Night"])
        season = st.selectbox("Season:", ["Winter", "Spring", "Summer", "Fall"])
        predict_button = st.button("Predict Temperature")
    
    with col2:
        if predict_button:
            # Prepare prediction
            features = {'building_density': building_density, 'vegetation_index': vegetation_index, 'albedo': albedo}
            predicted_temp = get_temperature_prediction(features)
            
            # Adjustments
            time_factors = {"Early Morning": -2.0, "Morning": -0.5, "Noon": 1.5, 
                           "Afternoon": 2.0, "Evening": 0.0, "Night": -1.5}
            season_factors = {"Winter": -5.0, "Spring": 0.0, "Summer": 5.0, "Fall": 0.0}
            land_use_factors = {"Residential": 0.0, "Commercial": 1.0, "Industrial": 1.5, 
                               "Park": -2.0, "Mixed Use": 0.5}
            
            predicted_temp += time_factors[time_of_day] + season_factors[season] + land_use_factors[land_use]
            
            # Display prediction
            st.markdown(f"""
            <div style="text-align:center; margin:20px; padding:20px; background-color:#f8f9fa; 
            border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
                <h3>Predicted Temperature</h3>
                <div style="font-size:48px; font-weight:bold; color:#e74c3c;">{predicted_temp:.1f}°C</div>
                <p>Based on the urban features and conditions you specified</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparison to baseline
            baseline_temp = 25.0
            temp_difference = predicted_temp - baseline_temp
            st.markdown(f"""
            <div style="margin:20px 0;">
                <strong>Comparison to Baseline:</strong> {temp_difference:.1f}°C 
                {'warmer' if temp_difference > 0 else 'cooler'} than the baseline temperature
            </div>
            """, unsafe_allow_html=True)
            
            # Feature impact analysis
            st.markdown("### Feature Impact Analysis")
            building_impact = 5 * building_density
            vegetation_impact = -4 * vegetation_index
            albedo_impact = -3 * albedo
            
            impact_data = pd.DataFrame({
                'Feature': ['Building Density', 'Vegetation', 'Albedo', 'Time of Day', 'Season', 'Land Use'],
                'Impact': [building_impact, vegetation_impact, albedo_impact, 
                          time_factors[time_of_day], season_factors[season], land_use_factors[land_use]]
            })
            
            impact_data['Abs_Impact'] = impact_data['Impact'].abs()
            impact_data = impact_data.sort_values('Abs_Impact', ascending=False)
            
            # Impact visualization
            fig = px.bar(impact_data, y='Feature', x='Impact', orientation='h',
                        title='Feature Impact on Temperature', color='Impact',
                        color_continuous_scale='RdBu_r', labels={'Impact': 'Temperature Impact (°C)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### Recommendations")
            if predicted_temp > 28:
                st.markdown("""
                #### High Temperature Zone
                This configuration results in elevated temperatures that could contribute to urban heat island effects.
                Consider the following mitigation strategies:
                - **Increase vegetation coverage**: Adding trees, green roofs, or pocket parks could reduce temperature by 1-2°C
                - **Implement cool surfaces**: Replacing dark surfaces with high-albedo materials
                - **Redesign urban geometry**: Modify building arrangements to improve air flow
                """)
            elif predicted_temp > 25:
                st.markdown("""
                #### Moderate Temperature Zone
                This configuration shows a moderate urban heat island effect. Some improvements could help:
                - **Enhance existing green spaces**: Increase vegetation in available areas
                - **Gradual surface replacements**: Consider light-colored materials for upcoming renovations
                - **Water features**: Small water elements could provide localized cooling
                """)
            else:
                st.markdown("""
                #### Low Temperature Zone
                This configuration effectively minimizes urban heat island effects. To maintain:
                - **Preserve existing vegetation**: Protect and maintain current green spaces
                - **Continue high-albedo practices**: Maintain reflective surfaces when replacing materials
                - **Use as model**: Consider applying similar configurations to other urban areas
                """)
        else:
            # Placeholder content
            st.markdown("""
            ### Temperature Prediction
            Adjust the parameters on the left and click "Predict Temperature" to see results.
            
            The prediction model considers:
            - Building density
            - Vegetation coverage
            - Surface reflectivity (albedo)
            - Land use type
            - Time of day and seasonal factors
            """)
            
            # Example visualization
            building_range = np.linspace(0, 1, 50)
            vegetation_range = np.linspace(0, 1, 50)
            X, Y = np.meshgrid(building_range, vegetation_range)
            Z = 25 + 5 * X - 4 * Y  # Simplified temperature model
            
            fig = go.Figure(data=[go.Contour(
                z=Z, x=building_range, y=vegetation_range, colorscale='Thermal',
                contours=dict(start=22, end=30, size=0.5, showlabels=True),
                colorbar=dict(title='Temperature (°C)', titleside='right')
            )])
            
            fig.update_layout(title='Temperature Prediction by Building Density and Vegetation',
                             xaxis_title='Building Density', yaxis_title='Vegetation Index', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            The contour plot above shows how temperature varies with building density and vegetation coverage.
            - **Higher building density** tends to increase temperature (moving right on the x-axis)
            - **Higher vegetation** tends to decrease temperature (moving up on the y-axis)
            - The contour lines represent equal temperature values
            """)

def show_intervention_planning(data):
    """Display the intervention planning module"""
    st.markdown('<h2 class="sub-header">UHI Intervention Planning</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module helps planners identify the most effective UHI mitigation strategies for specific urban areas. Select a location and the system will analyze local conditions and recommend tailored interventions.</div>', unsafe_allow_html=True)
    
    # Create layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Select Location")
        location_method = st.radio("Selection method:", ["Map Selection", "Address Search"])
        
        # Location input
        if location_method == "Map Selection":
            st.write("Map will appear here in a real application.")
            lat = st.number_input("Latitude:", value=40.7128, format="%.4f")
            lon = st.number_input("Longitude:", value=-74.0060, format="%.4f")
        else:
            address = st.text_input("Enter address:", "Times Square, New York, NY")
            if address:
                st.write("Address geocoded to coordinates:")
                lat, lon = 40.7580, -73.9855  # Times Square coordinates
                st.write(f"Latitude: {lat}, Longitude: {lon}")
            else:
                lat, lon = 40.7128, -74.0060  # Default NYC coordinates
        
        # Parameters
        radius = st.slider("Analysis radius (km):", 0.5, 5.0, 1.0, 0.5)
        priority = st.selectbox("Optimization priority:", ["Temperature Reduction", "Cost Efficiency", 
                                                          "Implementation Speed", "Balanced Approach"])
        
        # Constraints
        st.markdown("### Constraints")
        budget_constraint = st.select_slider("Budget level:", options=["Low", "Medium", "High"])
        time_constraint = st.select_slider("Implementation timeframe:", 
                                          options=["Short-term", "Medium-term", "Long-term"])
        
        analyze_button = st.button("Analyze & Suggest Interventions")
    
    with col2:
        if analyze_button:
            st.markdown("### Analysis Results")
            
            # Get recommendations
            location = (lat, lon)
            intervention_results = suggest_interventions(data, location)
            
            # Display local conditions
            st.markdown("#### Local Conditions")
            metric_cols = st.columns(4)
            with metric_cols[0]: st.metric("Temperature", intervention_results["local_temperature"], "+3.2°C")
            with metric_cols[1]: st.metric("Building Density", intervention_results["building_density"], "+0.15")
            with metric_cols[2]: st.metric("Vegetation Index", intervention_results["vegetation_index"], "-0.08")
            with metric_cols[3]: st.metric("Albedo", intervention_results["albedo"], "-0.12")
            
            # Area map
            st.markdown("#### Area Map")
            area_data = data[(data['latitude'] - lat)**2 + (data['longitude'] - lon)**2 <= (radius/111)**2]
            
            m = folium.Map(location=[lat, lon], zoom_start=14, tiles='CartoDB positron')
            heat_data = [[row['latitude'], row['longitude'], row['air_temperature']] for _, row in area_data.iterrows()]
            HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'},
                   min_opacity=0.5, blur=10).add_to(m)
            
            folium.Marker([lat, lon], popup=f"Selected Location<br>Temp: {intervention_results['local_temperature']}",
                         icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
            
            folium.Circle(location=[lat, lon], radius=radius * 1000, color='blue',
                         fill=True, fill_opacity=0.1).add_to(m)
            
            folium_static(m)
            
            # Intervention recommendations
            st.markdown("#### Recommended Interventions")
            
            if len(intervention_results["suggestions"]) > 0:
                # Create tabs for different intervention types
                intervention_types = list(set(s["type"] for s in intervention_results["suggestions"]))
                tabs = st.tabs(intervention_types + ["All Interventions"])
                
                for i, tab in enumerate(tabs):
                    with tab:
                        if i < len(intervention_types):
                            # Filter for this type
                            current_type = intervention_types[i]
                            type_suggestions = [s for s in intervention_results["suggestions"] 
                                               if s["type"] == current_type]
                            
                            for j, suggestion in enumerate(type_suggestions):
                                st.markdown(f"""
                                <div style="margin-bottom:15px; padding:15px; 
                                background-color:{'#e8f4f8' if j % 2 == 0 else '#f0f7fa'}; border-radius:5px;">
                                    <h4>{suggestion["description"]}</h4>
                                    <p><strong>Impact:</strong> {suggestion["impact"]}</p>
                                    <p><strong>Priority:</strong> {suggestion["priority"]}</p>
                                    <p><strong>Cost:</strong> {suggestion["cost_estimate"]}</p>
                                    <p><strong>Implementation:</strong> {suggestion["implementation_time"]}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            # Show all interventions
                            for j, suggestion in enumerate(intervention_results["suggestions"]):
                                st.markdown(f"""
                                <div style="margin-bottom:15px; padding:15px; 
                                background-color:{'#e8f4f8' if j % 2 == 0 else '#f0f7fa'}; border-radius:5px;">
                                    <h4>{suggestion["type"]}: {suggestion["description"]}</h4>
                                    <p><strong>Impact:</strong> {suggestion["impact"]}</p>
                                    <p><strong>Priority:</strong> {suggestion["priority"]}</p>
                                    <p><strong>Cost:</strong> {suggestion["cost_estimate"]}</p>
                                    <p><strong>Implementation:</strong> {suggestion["implementation_time"]}</p>
                                </div>
                                """, unsafe_allow_html=True)
            else:
                st.write("No specific interventions found for this location.")
            
            # Impact visualization
            if len(intervention_results["suggestions"]) > 0:
                st.markdown("#### Intervention Impact Visualization")
                
                # Select intervention to simulate
                intervention_options = [f"{s['type']}: {s['description']}" 
                                       for s in intervention_results["suggestions"]]
                selected_intervention = st.selectbox("Select intervention to visualize:", intervention_options)
                selected_type = selected_intervention.split(":")[0]
                
                # Intensity slider
                intensity = st.slider("Implementation intensity:", 0.1, 1.0, 0.5, 0.1,
                                     help="Higher values represent more extensive implementation")
                
                # Simulate impact
                new_data = simulate_intervention_impact(area_data, selected_type, intensity)
                
                # Impact statistics
                original_avg_temp = area_data['air_temperature'].mean()
                new_avg_temp = new_data['air_temperature'].mean()
                temp_reduction = original_avg_temp - new_avg_temp
                
                st.markdown(f"""
                #### Projected Impact
                **Temperature reduction:** {temp_reduction:.2f}°C  
                **Original average temperature:** {original_avg_temp:.2f}°C  
                **New average temperature:** {new_avg_temp:.2f}°C  
                """)
                
                # Before/after visualization
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=area_data['air_temperature'], name='Before Intervention',
                                          opacity=0.75, marker=dict(color='rgba(231, 76, 60, 0.7)')))
                fig.add_trace(go.Histogram(x=new_data['air_temperature'], name='After Intervention',
                                          opacity=0.75, marker=dict(color='rgba(46, 204, 113, 0.7)')))
                
                fig.update_layout(title=f'Temperature Distribution Before & After {selected_type} Implementation',
                                 xaxis_title='Temperature (°C)', yaxis_title='Frequency',
                                 barmode='overlay', bargap=0.1, height=400)
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Placeholder content
            st.markdown("""
            ### Intervention Planning
            
            Select a location and analysis parameters, then click "Analyze & Suggest Interventions" 
            to receive customized UHI mitigation recommendations.
            
            The system will:
            1. Analyze local urban characteristics
            2. Identify key contributors to UHI
            3. Recommend targeted interventions
            4. Visualize potential impact
            """)
            
            # Example strategies
            st.markdown("### Sample Intervention Strategies")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                #### Green Infrastructure
                - Tree planting programs
                - Green roofs and walls
                - Urban parks and green spaces
                - Vegetation corridors
                
                #### Cool Materials
                - High-albedo roofing
                - Reflective pavements
                - Cool building materials
                - Permeable surfaces
                """)
            
            with col2:
                st.markdown("""
                #### Urban Design
                - Building orientation
                - Street canyon modifications
                - Air flow optimization
                - Shade structures
                
                #### Water Features
                - Fountains and spray parks
                - Retention ponds
                - Urban streams restoration
                - Blue roofs
                """)

def show_optimization(data):
    """Display the optimization and simulation module""", '₹')}</p>
                                    <p><strong>Implementation:</strong> {suggestion["implementation_time"]}</p>
                                </div>
                                """, unsafe_allow_html=True)
            else:
                st.write("No specific interventions found for this location.")
            
            # Impact visualization
            if len(intervention_results["suggestions"]) > 0:
                st.markdown("#### Intervention Impact Visualization")
                
                # Select intervention to simulate
                intervention_options = [f"{s['type']}: {s['description']}" 
                                       for s in intervention_results["suggestions"]]
                selected_intervention = st.selectbox("Select intervention to visualize:", intervention_options)
                selected_type = selected_intervention.split(":")[0]
                
                # Intensity slider
                intensity = st.slider("Implementation intensity:", 0.1, 1.0, 0.5, 0.1,
                                     help="Higher values represent more extensive implementation")
                
                # Simulate impact
                new_data = simulate_intervention_impact(area_data, selected_type, intensity)
                
                # Impact statistics
                original_avg_temp = area_data['air_temperature'].mean()
                new_avg_temp = new_data['air_temperature'].mean()
                temp_reduction = original_avg_temp - new_avg_temp
                
                st.markdown(f"""
                #### Projected Impact
                **Temperature reduction:** {temp_reduction:.2f}°C  
                **Original average temperature:** {original_avg_temp:.2f}°C  
                **New average temperature:** {new_avg_temp:.2f}°C  
                """)
                
                # Before/after visualization
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=area_data['air_temperature'], name='Before Intervention',
                                          opacity=0.75, marker=dict(color='rgba(231, 76, 60, 0.7)')))
                fig.add_trace(go.Histogram(x=new_data['air_temperature'], name='After Intervention',
                                          opacity=0.75, marker=dict(color='rgba(46, 204, 113, 0.7)')))
                
                fig.update_layout(title=f'Temperature Distribution Before & After {selected_type} Implementation',
                                 xaxis_title='Temperature (°C)', yaxis_title='Frequency',
                                 barmode='overlay', bargap=0.1, height=400)
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Placeholder content
            st.markdown("""
            ### Intervention Planning
            
            Select a location and analysis parameters, then click "Analyze & Suggest Interventions" 
            to receive customized UHI mitigation recommendations for Nagpur.
            
            The system will:
            1. Analyze local urban characteristics
            2. Identify key contributors to UHI
            3. Recommend targeted interventions
            4. Visualize potential impact
            """)
            
            # Example strategies
            st.markdown("### Sample Intervention Strategies")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                #### Green Infrastructure
                - Tree planting programs
                - Green roofs and walls
                - Urban parks and green spaces
                - Vertical gardens on buildings
                
                #### Cool Materials
                - High-albedo roofing
                - Reflective pavements
                - Traditional white lime washing
                - Permeable surfaces
                """)
            
            with col2:
                st.markdown("""
                #### Urban Design
                - Building orientation for better airflow
                - Street canyon modifications
                - Traditional courtyard designs
                - Shade structures (chhatris)
                
                #### Water Features
                - Traditional step wells (baolis)
                - Fountains and spray parks
                - Retention ponds
                - Rainwater harvesting systems
                """)

def show_optimization(data):
    """Display the optimization and simulation module"""
    st.markdown('<h2 class="sub-header">Optimization & Simulation</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module uses optimization algorithms to find the most effective allocation of resources for UHI mitigation in Nagpur. It allows planners to simulate different scenarios and compare outcomes.</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Resource Optimization", "Scenario Simulation"])
    
    with tab1:
        st.markdown("### Resource Allocation Optimization")
        
        # Inputs
        budget_level = st.select_slider("Budget level:", options=["low", "medium", "high"])
        priority = st.selectbox("Optimization priority:", ["temperature", "cost", "implementation"])
        
        if st.button("Rundef show_uhi_detection(data):
    """Display the UHI detection and analysis module"""
    st.markdown('<h2 class="sub-header">UHI Detection & Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module uses ground temperature measurements and urban feature analysis to detect urban heat island hotspots in Nagpur.</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Temperature Analysis", "Cluster Analysis"])
    
    with tab1:
        st.markdown("### Ground Temperature Analysis")
        st.write("Select an area to analyze or use the demo data:")
        
        # Input controls
        col1, col2 = st.columns(2)
        with col1:
            location = st.selectbox("Select location:", ["Nagpur City Center", "Dharampeth", "Sadar", "Custom Location"])
            if location == "Nagpur City Center":
                lat, lon = 21.1458, 79.0882
            elif location == "Dharampeth":
                lat, lon = 21.1350, 79.0650
            elif location == "Sadar":
                lat, lon = 21.1530, 79.0800
            else:  # Custom Location
                lat = st.number_input("Latitude:", value=21.1458, format="%.4f")
                lon = st.number_input("Longitude:", value=79.0882, format="%.4f")
        
        with col2:
            analysis_date = st.date_input("Select date for analysis:", datetime.date(2025, 6, 1))
            measurement_type = st.selectbox("Measurement type:", ["Ground Temperature", "Surface Temperature"])
        
        # Run analysis
        if st.button("Run Temperature Analysis"):
            st.markdown("#### Analysis Results")
            
            # Metrics
            cols = st.columns(3)
            with cols[0]:
                ground_temp = get_ground_temperature_data(lat, lon, analysis_date)
                st.metric("Ground Temperature", f"{ground_temp:.1f}°C", "+3.2°C")
            with cols[1]:
                surface_temp = data[data['latitude'].between(lat-0.01, lat+0.01) & 
                                   data['longitude'].between(lon-0.01, lon+0.01)]['surface_temperature'].mean()
                st.metric("Surface Temperature", f"{surface_temp:.1f}°C", "+4.5°C")
            with cols[2]:
                building_density = data[data['latitude'].between(lat-0.01, lat+0.01) & 
                                       data['longitude'].between(lon-0.01, lon+0.01)]['building_density'].mean()
                st.metric("Building Density", f"{building_density:.2f}", "+0.04")
            
            # Heat map
            st.markdown("#### Temperature Map")
            area_data = data[data['latitude'].between(lat-0.03, lat+0.03) & data['longitude'].between(lon-0.03, lon+0.03)]
            m = folium.Map(location=[lat, lon], zoom_start=14, tiles='CartoDB positron')
            heat_data = [[row['latitude'], row['longitude'], row['surface_temperature']] for _, row in area_data.iterrows()]def show_optimization(data):
    """Display the optimization and simulation module"""
    st.markdown('<h2 class="sub-header">Optimization & Simulation</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module uses optimization algorithms to find the most effective allocation of resources for UHI mitigation. It allows planners to simulate different scenarios and compare outcomes.</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Resource Optimization", "Scenario Simulation"])
    
    with tab1:
        st.markdown("### Resource Allocation Optimization")
        
        # Inputs
        budget_level = st.select_slider("Budget level:", options=["low", "medium", "high"])
        priority = st.selectbox("Optimization priority:", ["temperature", "cost", "implementation"])
        
        if st.button("Run Optimization"):
            # Get optimization results
            results = optimize_interventions(data, budget_level, priority)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Budget", f"${results['budget']}")
            with col2: st.metric("Used Budget", f"${results['used_budget']}", 
                               f"{results['used_budget']/results['budget']*100:.1f}%")
            with col3: st.metric("Temperature Reduction", f"{results['estimated_temperature_reduction']:.2f}°C")
            
            # Resource allocation chart
            st.markdown("#### Resource Allocation")
            allocation_df = pd.DataFrame(results['allocation'])
            
            fig = px.bar(allocation_df, x='cost', y='name', orientation='h', color='temperature_reduction',
                        color_continuous_scale='Blues', title='Intervention Resource Allocation',
                        labels={'cost': 'Budget Allocation ($)', 'name': 'Intervention',
                               'temperature_reduction': 'Temp. Reduction (°C)'}, text='units')
            fig.update_traces(texttemplate='%{text} units', textposition='inside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Type distribution
            type_summary = allocation_df.groupby('type').agg({
                'cost': 'sum', 'temperature_reduction': 'sum', 'units': 'sum'
            }).reset_index()
            
            fig = px.pie(type_summary, values='cost', names='type',
                        title='Budget Distribution by Intervention Type', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cost-effectiveness analysis
            st.markdown("#### Cost-Effectiveness Analysis")
            allocation_df['cost_per_degree'] = allocation_df['cost'] / allocation_df['temperature_reduction']
            cost_effectiveness = allocation_df.sort_values('cost_per_degree')
            
            fig = px.bar(cost_effectiveness, x='name', y='cost_per_degree', color='type',
                        title='Cost per Degree of Cooling ($ / °C)',
                        labels={'cost_per_degree': 'Cost per °C Reduction ($)', 'name': 'Intervention'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Implementation timeline
            st.markdown("#### Implementation Timeline")
            
            # Create simple timeline data
            timeline_data = []
            start_date = datetime.date(2025, 7, 1)
            
            for i, row in allocation_df.iterrows():
                # Assign durations based on type
                if row['type'] == 'Green Infrastructure': duration = 90
                elif row['type'] == 'High-Albedo Surfaces': duration = 60
                elif row['type'] == 'Water Features': duration = 120
                else: duration = 30
                
                end_date = start_date + datetime.timedelta(days=duration)
                timeline_data.append({
                    'Task': row['name'], 'Start': start_date, 'Finish': end_date, 'Type': row['type']
                })
                start_date = start_date + datetime.timedelta(days=30)
            
            timeline_df = pd.DataFrame(timeline_data)
            
            fig = px.timeline(timeline_df, x_start='Start', x_end='Finish', y='Task',
                             color='Type', title='Implementation Timeline')
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Future Scenario Simulation")
        
        # Inputs
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Scenario Parameters")
            year = st.slider("Target Year:", 2025, 2050, 2030, 5)
            urban_growth = st.select_slider("Urban Growth Rate:", options=["Low", "Medium", "High"])
            climate_scenario = st.select_slider("Climate Change Scenario:", 
                                               options=["Optimistic", "Moderate", "Pessimistic"])
            mitigation_level = st.select_slider("UHI Mitigation Implementation:", 
                                              options=["Minimal", "Moderate", "Aggressive"])
            simulate_button = st.button("Run Simulation")
        
        with col2:
            if simulate_button:
                st.markdown("#### Simulation Results")
                
                # Calculate climate scenario impacts
                if climate_scenario == "Optimistic":
                    climate_increase = 0.5 * (year - 2025) / 5  # 0.5°C per 5 years
                elif climate_scenario == "Moderate":
                    climate_increase = 1.0 * (year - 2025) / 5  # 1.0°C per 5 years
                else:  # Pessimistic
                    climate_increase = 1.5 * (year - 2025) / 5  # 1.5°C per 5 years
                
                # Calculate urban growth impacts
                if urban_growth == "Low":
                    growth_factor = 0.2 * (year - 2025) / 5  # 0.2°C per 5 years
                elif urban_growth == "Medium":
                    growth_factor = 0.5 * (year - 2025) / 5  # 0.5°C per 5 years
                else:  # High
                    growth_factor = 0.8 * (year - 2025) / 5  # 0.8°C per 5 years
                
                # Calculate mitigation effects
                if mitigation_level == "Minimal":
                    mitigation_effect = 0.2 * (year - 2025) / 5  # 0.2°C reduction per 5 years
                elif mitigation_level == "Moderate":
                    mitigation_effect = 0.7 * (year - 2025) / 5  # 0.7°C reduction per 5 years
                else:  # Aggressive
                    mitigation_effect = 1.2 * (year - 2025) / 5  # 1.2°C reduction per 5 years
                
                # Calculate UHI change
                current_uhi = data['air_temperature'].mean() - 25  # Assuming 25°C is the baseline
                future_uhi = current_uhi + climate_increase + growth_factor - mitigation_effect
                
                # Display results
                st.markdown(f"""
                #### Projected UHI Intensity for {year}
                
                **Current UHI Intensity (2025):** {current_uhi:.2f}°C  
                **Projected UHI Intensity ({year}):** {future_uhi:.2f}°C  
                
                **Contributing Factors:**
                - Climate change impact: +{climate_increase:.2f}°C
                - Urban growth impact: +{growth_factor:.2f}°C
                - Mitigation effect: -{mitigation_effect:.2f}°C
                
                **Net Change:** {future_uhi - current_uhi:.2f}°C
                """)
                
                # Waterfall chart
                waterfall_data = pd.DataFrame({
                    'Factor': ['Current UHI', 'Climate Change', 'Urban Growth', 'Mitigation', f'UHI in {year}'],
                    'Value': [current_uhi, climate_increase, growth_factor, -mitigation_effect, future_uhi],
                    'Type': ['Total', 'Increase', 'Increase', 'Decrease', 'Total']
                })
                
                fig = go.Figure(go.Waterfall(
                    name="UHI Components", orientation="v", measure=waterfall_data['Type'],
                    x=waterfall_data['Factor'], y=waterfall_data['Value'],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": "#e74c3c"}},
                    decreasing={"marker": {"color": "#2ecc71"}},
                    totals={"marker": {"color": "#3498db"}}
                ))
                
                fig.update_layout(title=f"UHI Intensity Change from 2025 to {year}",
                                 yaxis_title="Temperature Change (°C)", height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Temperature distribution comparison
                st.markdown("#### Temperature Distribution Comparison")
                
                # Create synthetic distributions
                current_temps = data['air_temperature'].values
                future_temps = current_temps + (future_uhi - current_uhi)
                future_temps += np.random.normal(0, 0.5, size=len(future_temps))  # Add variability
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=current_temps, name='Current (2025)', opacity=0.75,
                                          marker=dict(color='rgba(52, 152, 219, 0.7)')))
                fig.add_trace(go.Histogram(x=future_temps, name=f'Projected ({year})', opacity=0.75,
                                          marker=dict(color='rgba(231, 76, 60, 0.7)')))
                
                fig.update_layout(title=f'Temperature Distribution: Current vs {year}',
                                 xaxis_title='Temperature (°C)', yaxis_title='Frequency',
                                 barmode='overlay', bargap=0.1, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Heat wave risk assessment
                st.markdown("#### Heat Wave Risk Assessment")
                
                # Determine risk level
                if future_uhi > 5:
                    risk_level, risk_color = "High", "#e74c3c"
                elif future_uhi > 3:
                    risk_level, risk_color = "Medium", "#f39c12"
                else:
                    risk_level, risk_color = "Low", "#2ecc71"
                
                # Calculate additional metrics
                extreme_heat_days_current = sum(current_temps > 30) / len(current_temps) * 365
                extreme_heat_days_future = sum(future_temps > 30) / len(future_temps) * 365
                
                st.markdown(f"""
                <div style="padding:20px; background-color:{risk_color}25; border-left:5px solid {risk_color}; margin-bottom:20px;">
                    <h4>Heat Wave Risk Level: <span style="color:{risk_color}">{risk_level}</span></h4>
                    <p>
                        <strong>Days over 30°C per year:</strong><br>
                        Current (2025): {extreme_heat_days_current:.1f} days<br>
                        Projected ({year}): {extreme_heat_days_future:.1f} days<br>
                        <strong>Increase: {extreme_heat_days_future - extreme_heat_days_current:.1f} days</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("#### Recommendations")
                
                if risk_level == "High":
                    st.markdown("""
                    To address the high heat wave risk projected in this scenario:
                    
                    1. **Implement comprehensive UHI mitigation plan** with emphasis on cooling interventions
                    2. **Develop heat emergency response protocols** for vulnerable populations
                    3. **Increase green infrastructure budget** to maximize cooling effect
                    4. **Revise building codes** to mandate cool roofs and energy-efficient designs
                    5. **Create cooling centers network** accessible within 10-minute walks citywide
                    """)
                elif risk_level == "Medium":
                    st.markdown("""
                    To address the medium heat wave risk projected in this scenario:
                    
                    1. **Gradually increase green cover** in hotspot areas
                    2. **Implement cool pavement program** during regular maintenance cycles
                    3. **Develop targeted interventions** for vulnerable neighborhoods
                    4. **Create incentives for green roofs** and cool building materials
                    5. **Monitor temperature trends** and adjust strategies accordingly
                    """)
                else:
                    st.markdown("""
                    To maintain the low heat wave risk projected in this scenario:
                    
                    1. **Continue current mitigation efforts** to maintain progress
                    2. **Preserve existing green spaces** and expand when possible
                    3. **Incorporate UHI considerations** in all future development
                    4. **Monitor temperature data** to detect any unexpected changes
                    5. **Document successful strategies** to share with other cities
                    """)
            else:
                # Placeholder content
                st.markdown("""
                ### Scenario Simulation
                
                Configure the parameters on the left and click "Run Simulation" to see projections of future UHI patterns
                based on different climate change, urban growth, and mitigation scenarios.
                
                The simulation will show:
                - Projected UHI intensity changes
                - Temperature distribution shifts
                - Heat wave risk assessment
                - Tailored recommendations based on outcomes
                """)
                
                # Sample projection chart
                years = list(range(2025, 2051, 5))
                no_action = [3.0 + 0.4 * i for i in range(len(years))]
                moderate_action = [3.0 + 0.3 * i - 0.1 * i**2 for i in range(len(years))]
                aggressive_action = [3.0 + 0.2 * i - 0.15 * i**2 for i in range(len(years))]
                
                scenario_df = pd.DataFrame({
                    'Year': years * 3,
                    'UHI Intensity (°C)': no_action + moderate_action + aggressive_action,
                    'Scenario': ['No Action'] * len(years) + ['Moderate Action'] * len(years) + 
                                ['Aggressive Action'] * len(years)
                })
                
                fig = px.line(scenario_df, x='Year', y='UHI Intensity (°C)', color='Scenario',
                             title='UHI Intensity Projections by Mitigation Scenario')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

def show_about():
    """Display the about page with project information"""
    st.markdown('<h2 class="sub-header">About This Project</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>Urban Heat Island Analysis & Mitigation System</h3>
        <p>This project aims to develop an integrated AI-based system that helps city planners and environmental 
        scientists detect, analyze, and mitigate urban heat island effects through data-driven decision making.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key features
    st.markdown("### Key Features")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Detection & Analysis
        - Satellite imagery analysis for UHI detection
        - Street-level temperature mapping
        - Temporal and spatial pattern identification
        - Cluster analysis of similar urban areas
        
        #### Prediction
        - Machine learning models for temperature prediction
        - Impact assessment of new developments
        - Future scenario simulation
        - Climate change integration
        """)
    
    with col2:
        st.markdown("""
        #### Intervention Planning
        - Customized intervention recommendations
        - Cost-benefit analysis of strategies
        - Implementation priority ranking
        - Visualization of potential impacts
        
        #### Optimization
        - Resource allocation optimization
        - Multi-objective decision support
        - Budget-constrained planning
        - Scenario comparison
        """)
    
    # Technical details
    st.markdown("### Technical Details")
    st.markdown("""
    This system integrates multiple technologies and data sources:
    
    - **Satellite Data**: Utilizes freely available Landsat, Sentinel-2, and MODIS data
    - **Machine Learning**: Employs random forest and gradient boosting models
    - **Optimization Algorithms**: Uses multi-objective optimization for planning
    - **GIS Integration**: Provides spatial analysis and mapping capabilities
    - **Simulation Models**: Enables scenario testing and future projections
    
    The application is built using Python and Streamlit, making it accessible through any web browser.
    No proprietary software or paid services are required to run the system.
    """)
    
    # Data sources
    st.markdown("### Data Sources")
    st.markdown("""
    The system can utilize data from various free sources:
    
    - NASA Earth Data (https://earthdata.nasa.gov/)
    - USGS Earth Explorer (https://earthexplorer.usgs.gov/)
    - Copernicus Open Access Hub (https://scihub.copernicus.eu/)
    - OpenStreetMap (https://www.openstreetmap.org/)
    - National Weather Service (https://www.weather.gov/)
    - Local municipal GIS data portals
    
    For demonstration purposes, this app uses synthetic data that simulates realistic urban temperature patterns.
    """)

# ---------- MAIN APPLICATION ----------
def main():
    """Main application entry point"""
    # Setup page
    setup_page()
    
    # Page header
    st.markdown('<h1 class="main-header">Urban Heat Island Analysis & Mitigation System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
        This AI-based system helps city planners and environmental scientists analyze urban heat island (UHI) effects 
        and develop data-driven strategies to mitigate their impact. Using satellite imagery, environmental data, 
        and machine learning, it provides insights for sustainable urban planning.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.image("https://www.epa.gov/sites/default/files/styles/medium/public/2020-07/urban-heat-island.jpg", 
                    use_container_width=True)
    st.sidebar.title("Navigation")
    
    # Page selection
    page = st.sidebar.radio("Select a Module", 
                           ["Dashboard", "UHI Detection", "Temperature Prediction", 
                            "Intervention Planning", "Optimization & Simulation", "About"])
    
    # Load sample data
    data = load_sample_data()
    
    # Display selected page
    if page == "Dashboard": show_dashboard(data)
    elif page == "UHI Detection": show_uhi_detection(data)
    elif page == "Temperature Prediction": show_temperature_prediction(data)
    elif page == "Intervention Planning": show_intervention_planning(data)
    elif page == "Optimization & Simulation": show_optimization(data)
    elif page == "About": show_about()
    
    # Footer
    st.markdown('<div class="footer">Urban Heat Island Analysis & Mitigation System © 2025</div>', 
               unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()"""
Urban Heat Island Analysis & Mitigation System - Streamlined Version
Author: Claude AI
Date: June 2025
"""

import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ---------- PAGE SETUP AND CONFIGURATION ----------
def setup_page():
    st.set_page_config(page_title="UHI Analysis System", page_icon="🌆", layout="wide")
    st.markdown("""
    <style>
        .main-header {font-size:2.5rem; color:#2c3e50; text-align:center; margin-bottom:1rem;}
        .sub-header {font-size:1.8rem; color:#34495e; margin-top:2rem; margin-bottom:1rem;}
        .card {background-color:#f8f9fa; border-radius:5px; padding:20px; margin-bottom:20px; box-shadow:0 4px 6px rgba(0,0,0,0.1);}
        .highlight {background-color:#e8f4f8; padding:10px; border-left:5px solid #3498db; margin-bottom:15px;}
        .footer {text-align:center; margin-top:3rem; color:#7f8c8d; font-size:0.8rem;}
        .metric-container {background-color:#f8f9fa; border-radius:5px; padding:15px; margin:10px 0; box-shadow:0 2px 4px rgba(0,0,0,0.1);}
        .metric-label {font-size:1rem; color:#7f8c8d;}
        .metric-value {font-size:1.8rem; font-weight:bold; color:#2c3e50;}
    </style>
    """, unsafe_allow_html=True)

# ---------- DATA GENERATION AND ANALYSIS FUNCTIONS ----------
def load_sample_data():
    """Generate synthetic UHI data for demonstration"""
    np.random.seed(42)
    lat_center, lon_center = 21.1458, 79.0882  # Nagpur, Maharashtra coordinates
    n_points = 500
    
    # Generate locations and urban features
    lats = lat_center + np.random.normal(0, 0.03, n_points)
    lons = lon_center + np.random.normal(0, 0.03, n_points)
    building_density = np.random.beta(2, 2, n_points)
    vegetation_index = 1 - np.random.beta(2, 1.5, n_points) * building_density
    albedo = np.random.beta(2, 5, n_points) * (1 - vegetation_index * 0.5)
    
    # Calculate distances and normalize
    dist_from_center = np.sqrt((lats - lat_center)**2 + (lons - lon_center)**2)
    dist_normalized = (dist_from_center - dist_from_center.min()) / (dist_from_center.max() - dist_from_center.min())
    
    # Calculate temperatures based on urban features
    base_temp = 32  # Higher base temperature for Nagpur
    uhi_effect = 5 * building_density - 3 * vegetation_index - 2 * albedo - 1 * dist_normalized
    temperature = base_temp + uhi_effect + np.random.normal(0, 0.5, n_points)
    surface_temp = temperature + 2 + 4 * building_density - 3 * vegetation_index + np.random.normal(0, 0.7, n_points)
    
    # Assign land use categories
    land_use = np.random.choice(
        ['Commercial', 'Residential', 'Industrial', 'Park', 'Water Body'],
        n_points, 
        p=[0.3, 0.4, 0.1, 0.15, 0.05]
    )
    
    # Create and return dataframe
    return pd.DataFrame({
        'latitude': lats, 'longitude': lons, 'building_density': building_density,
        'vegetation_index': vegetation_index, 'albedo': albedo, 'air_temperature': temperature,
        'surface_temperature': surface_temp, 'land_use': land_use, 'distance_from_center': dist_from_center
    })

def get_satellite_ndvi_data(lat, lon, date=None):
    """Mock function to simulate fetching NDVI data from satellite imagery"""
    np.random.seed(int(lat*100 + lon*100))
    lat_center, lon_center = 40.7128, -74.0060
    dist = np.sqrt((lat - lat_center)**2 + (lon - lon_center)**2)
    normalized_dist = min(dist / 0.1, 1)  # 0.1 is max_dist
    ndvi_value = -0.1 + normalized_dist * 0.7 + np.random.normal(0, 0.05)
    return max(-1, min(1, ndvi_value))  # Clamp to valid range

def get_temperature_prediction(features):
    """Predict temperature based on urban features"""
    return (25 + 5 * features['building_density'] - 4 * features['vegetation_index'] - 
            3 * features['albedo'] + np.random.normal(0, 0.2))

def create_cluster_map(data, n_clusters=5):
    """Create clusters of similar urban areas based on UHI characteristics"""
    # Prepare and cluster data
    features = data[['building_density', 'vegetation_index', 'albedo', 'air_temperature']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(scaled_features)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Create map
    colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F3']
    m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], 
                  zoom_start=12, tiles='CartoDB positron')
    
    # Add markers and legend
    for idx, row in data.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']], radius=5,
            color=colors[row['cluster']], fill=True, fill_color=colors[row['cluster']], fill_opacity=0.7,
            popup=f"Cluster: {row['cluster']}<br>Temp: {row['air_temperature']:.1f}°C<br>" +
                  f"Building Density: {row['building_density']:.2f}<br>" +
                  f"Vegetation: {row['vegetation_index']:.2f}<br>Land Use: {row['land_use']}"
        ).add_to(m)
    
    # Create legend
    legend_html = '''
    <div style="position:fixed; bottom:50px; right:50px; width:150px; height:160px; 
    border:2px solid grey; z-index:9999; font-size:14px; background-color:white;
    padding:10px; border-radius:5px;"><span style="font-weight:bold;">Clusters</span><br>
    '''
    
    for i, center in enumerate(centers):
        if center[0] > 0.6: cluster_desc = "Urban Core"
        elif center[1] > 0.6: cluster_desc = "Green Zone"
        elif center[3] > 27: cluster_desc = "Hot Spot"
        elif center[2] > 0.4: cluster_desc = "Reflective Area"
        else: cluster_desc = f"Cluster {i}"
        legend_html += f'<div style="background-color:{colors[i]}; width:20px; height:20px; display:inline-block;"></div> {cluster_desc}<br>'
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m, data

def suggest_interventions(data, location):
    """Suggest UHI mitigation strategies based on analysis"""
    # Get neighborhood data
    lat, lon = location
    distances = np.sqrt((data['latitude'] - lat)**2 + (data['longitude'] - lon)**2)
    neighborhood_data = data[distances < 0.01]  # ~1km radius
    
    if len(neighborhood_data) == 0:
        return {"message": "No data available for this location.", "suggestions": []}
    
    # Calculate neighborhood averages
    avg_temp = neighborhood_data['air_temperature'].mean()
    avg_building_density = neighborhood_data['building_density'].mean()
    avg_vegetation = neighborhood_data['vegetation_index'].mean()
    avg_albedo = neighborhood_data['albedo'].mean()
    suggestions = []
    
    # Generate suggestions based on conditions
    if avg_vegetation < 0.3:
        suggestions.append({
            "type": "Green Infrastructure", "priority": "High", "score": 5,
            "description": "Increase urban vegetation through tree planting, green roofs, or pocket parks.",
            "impact": f"Could reduce local temperature by {1.5 + np.random.uniform(0, 0.5):.1f}°C",
            "cost_estimate": "$$", "implementation_time": "Medium-term"
        })
    elif avg_vegetation < 0.5:
        suggestions.append({
            "type": "Green Infrastructure", "priority": "Medium", "score": 3,
            "description": "Enhance existing green spaces and add vegetation to streets.",
            "impact": f"Could reduce local temperature by {0.8 + np.random.uniform(0, 0.3):.1f}°C",
            "cost_estimate": "$", "implementation_time": "Short-term"
        })
    
    if avg_albedo < 0.2:
        suggestions.append({
            "type": "High-Albedo Surfaces", "priority": "High", "score": 5,
            "description": "Implement cool roofs and pavements to reflect more solar radiation.",
            "impact": f"Could reduce local temperature by {1.2 + np.random.uniform(0, 0.4):.1f}°C",
            "cost_estimate": "$$", "implementation_time": "Medium-term"
        })
    elif avg_albedo < 0.4:
        suggestions.append({
            "type": "High-Albedo Surfaces", "priority": "Medium", "score": 3,
            "description": "Gradually replace dark surfaces with lighter materials.",
            "impact": f"Could reduce local temperature by {0.7 + np.random.uniform(0, 0.3):.1f}°C",
            "cost_estimate": "$", "implementation_time": "Ongoing"
        })
    
    if avg_building_density > 0.7:
        suggestions.append({
            "type": "Urban Design", "priority": "High", "score": 4,
            "description": "Modify building arrangements to improve air flow and reduce heat trapping.",
            "impact": f"Could reduce local temperature by {1.0 + np.random.uniform(0, 0.5):.1f}°C",
            "cost_estimate": "$$$", "implementation_time": "Long-term"
        })
    elif avg_building_density > 0.5:
        suggestions.append({
            "type": "Urban Design", "priority": "Medium", "score": 2,
            "description": "Consider height variations in future developments to enhance ventilation.",
            "impact": f"Could reduce local temperature by {0.5 + np.random.uniform(0, 0.3):.1f}°C",
            "cost_estimate": "$$", "implementation_time": "Long-term"
        })
    
    if avg_temp > 28:
        suggestions.append({
            "type": "Water Features", "priority": "Medium", "score": 3,
            "description": "Incorporate water elements like fountains or retention ponds for cooling.",
            "impact": f"Could reduce local temperature by {0.8 + np.random.uniform(0, 0.3):.1f}°C",
            "cost_estimate": "$$", "implementation_time": "Medium-term"
        })
    
    # Sort suggestions by priority
    suggestions.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "message": "Analysis complete. Interventions suggested based on local conditions.",
        "local_temperature": f"{avg_temp:.1f}°C",
        "building_density": f"{avg_building_density:.2f}",
        "vegetation_index": f"{avg_vegetation:.2f}",
        "albedo": f"{avg_albedo:.2f}",
        "suggestions": suggestions
    }

def simulate_intervention_impact(current_data, intervention_type, intensity=0.5):
    """Simulate the impact of implementing a specific intervention"""
    new_data = current_data.copy()
    
    # Apply intervention effects
    if intervention_type == "Green Infrastructure":
        new_data['vegetation_index'] = new_data['vegetation_index'] + (1 - new_data['vegetation_index']) * intensity * 0.5
        new_data['vegetation_index'] = new_data['vegetation_index'].clip(upper=1.0)
    elif intervention_type == "High-Albedo Surfaces":
        new_data['albedo'] = new_data['albedo'] + (1 - new_data['albedo']) * intensity * 0.6
        new_data['albedo'] = new_data['albedo'].clip(upper=1.0)
    elif intervention_type == "Urban Design":
        new_data['building_density'] = new_data['building_density'] * (1 - intensity * 0.3)
    elif intervention_type == "Water Features":
        lat_center, lon_center = new_data['latitude'].mean(), new_data['longitude'].mean()
        distances = np.sqrt((new_data['latitude'] - lat_center)**2 + (new_data['longitude'] - lon_center)**2)
        max_dist = distances.max()
        cooling_effect = intensity * 0.5 * (1 - distances/max_dist)
        new_data['vegetation_index'] = new_data['vegetation_index'] + cooling_effect * 0.3
        new_data['vegetation_index'] = new_data['vegetation_index'].clip(upper=1.0)
    
    # Recalculate temperatures
    base_temp = 25
    uhi_effect = (5 * new_data['building_density'] - 3 * new_data['vegetation_index'] - 
                 2 * new_data['albedo'] - 1 * new_data['distance_from_center'].clip(0, 1))
    np.random.seed(42)
    new_data['air_temperature'] = base_temp + uhi_effect + np.random.normal(0, 0.2, len(new_data))
    
    return new_data

def optimize_interventions(data, budget_level='medium', priority='temperature'):
    """Optimize intervention strategy based on budget constraints and priorities"""
    # Define budget levels in Indian Rupees (₹)
    budget_map = {'low': 2000000, 'medium': 5000000, 'high': 10000000}  # ₹20 lakh, ₹50 lakh, ₹1 crore
    budget = budget_map.get(budget_level, 5000000)
    
    # Define intervention options with costs in INR
    interventions = [
        {'name': 'Tree Planting', 'type': 'Green Infrastructure', 'cost_per_unit': 200000, 
         'temp_reduction_per_unit': 0.05, 'max_units': 30},
        {'name': 'Cool Roofs', 'type': 'High-Albedo Surfaces', 'cost_per_unit': 300000, 
         'temp_reduction_per_unit': 0.08, 'max_units': 20},
        {'name': 'Cool Pavements', 'type': 'High-Albedo Surfaces', 'cost_per_unit': 400000, 
         'temp_reduction_per_unit': 0.06, 'max_units': 15},
        {'name': 'Green Roofs', 'type': 'Green Infrastructure', 'cost_per_unit': 500000, 
         'temp_reduction_per_unit': 0.1, 'max_units': 12},
        {'name': 'Water Features', 'type': 'Water Features', 'cost_per_unit': 600000, 
         'temp_reduction_per_unit': 0.12, 'max_units': 8}
    ]
    
    # Sort interventions by priority
    if priority == 'temperature':
        interventions.sort(key=lambda x: x['temp_reduction_per_unit'] / x['cost_per_unit'], reverse=True)
    elif priority == 'cost':
        interventions.sort(key=lambda x: x['cost_per_unit'])
    elif priority == 'implementation':
        interventions.sort(key=lambda x: x['cost_per_unit'])
    
    # Allocate budget using greedy algorithm
    allocation, remaining_budget = [], budget
    for intervention in interventions:
        affordable_units = min(intervention['max_units'], int(remaining_budget / intervention['cost_per_unit']))
        if affordable_units > 0:
            cost = affordable_units * intervention['cost_per_unit']
            temp_reduction = affordable_units * intervention['temp_reduction_per_unit']
            allocation.append({
                'name': intervention['name'], 'type': intervention['type'], 'units': affordable_units,
                'cost': cost, 'temperature_reduction': temp_reduction
            })
            remaining_budget -= cost
    
    # Calculate totals
    total_cost = sum(item['cost'] for item in allocation)
    total_reduction = sum(item['temperature_reduction'] for item in allocation)
    
    return {
        'budget': budget, 'used_budget': total_cost, 'remaining_budget': remaining_budget,
        'estimated_temperature_reduction': total_reduction, 'allocation': allocation
    }

# ---------- UI MODULES ----------
def show_dashboard(data):
    """Display the main dashboard with overview metrics and visualizations"""
    st.markdown('<h2 class="sub-header">UHI Dashboard</h2>', unsafe_allow_html=True)
    
    # Summary metrics
    cols = st.columns(4)
    with cols[0]: st.markdown(f'<div class="metric-container"><div class="metric-label">Average Temperature</div><div class="metric-value">{data["air_temperature"].mean():.1f}°C</div></div>', unsafe_allow_html=True)
    with cols[1]: st.markdown(f'<div class="metric-container"><div class="metric-label">Max Temperature</div><div class="metric-value">{data["air_temperature"].max():.1f}°C</div></div>', unsafe_allow_html=True)
    with cols[2]: st.markdown(f'<div class="metric-container"><div class="metric-label">Average Vegetation Index</div><div class="metric-value">{data["vegetation_index"].mean():.2f}</div></div>', unsafe_allow_html=True)
    with cols[3]: st.markdown(f'<div class="metric-container"><div class="metric-label">Temperature Anomaly</div><div class="metric-value">+{data["air_temperature"].max() - data["air_temperature"].min():.1f}°C</div></div>', unsafe_allow_html=True)
    
    st.selectbox("Select City", ["Nagpur, Maharashtra (Demo)"], index=0)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Heat Map", "Analysis", "Trends"])
    
    with tab1:
        st.markdown("### Urban Heat Map")
        st.write("This heat map shows the temperature distribution across Nagpur.")
        
        # Create heat map
        m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=12, tiles='CartoDB positron')
        heat_data = [[row['latitude'], row['longitude'], row['air_temperature']] for _, row in data.iterrows()]
        HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'}).add_to(m)
        folium_static(m)
    
    with tab2:
        st.markdown("### UHI Analysis")
        
        # Create charts
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(data, x='building_density', y='air_temperature', color='vegetation_index',
                            color_continuous_scale='Viridis', title='Temperature vs Building Density',
                            labels={'building_density': 'Building Density', 'air_temperature': 'Temperature (°C)',
                                   'vegetation_index': 'Vegetation Index'}, size_max=10, hover_data=['land_use'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            land_use_temp = data.groupby('land_use')['air_temperature'].mean().reset_index()
            fig = px.bar(land_use_temp, x='land_use', y='air_temperature', color='air_temperature',
                        color_continuous_scale='Thermal', title='Average Temperature by Land Use',
                        labels={'land_use': 'Land Use Type', 'air_temperature': 'Avg Temperature (°C)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.markdown("### Feature Correlations")
        corr_features = ['building_density', 'vegetation_index', 'albedo', 'air_temperature', 'surface_temperature']
        corr_matrix = data[corr_features].corr()
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                       title='Correlation Between Urban Features',
                       labels=dict(x='Features', y='Features', color='Correlation'))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Temperature Trends")
        
        # Generate temporal data (for demo)
        dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
        avg_temps = [data['air_temperature'].mean() + np.sin(i/5) + np.random.normal(0, 0.3) for i in range(30)]
        max_temps = [t + 2 + np.random.normal(0, 0.2) for t in avg_temps]
        time_df = pd.DataFrame({'date': dates, 'avg_temperature': avg_temps, 'max_temperature': max_temps})
        
        # Plot time series
        fig = px.line(time_df, x='date', y=['avg_temperature', 'max_temperature'],
                     title='Temperature Trends Over Time',
                     labels={'date': 'Date', 'value': 'Temperature (°C)', 'variable': 'Metric'})
        fig.update_layout(legend_title_text='')
        st.plotly_chart(fig, use_container_width=True)
        
        # Temperature forecast
        st.markdown("### Temperature Forecast")
        forecast_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=7, freq='D')
        forecast_temps = [avg_temps[-1] + 0.1*i + np.random.normal(0, 0.2) for i in range(7)]
        full_dates = dates.append(forecast_dates)
        full_temps = avg_temps + forecast_temps
        forecast_df = pd.DataFrame({
            'date': full_dates,
            'temperature': full_temps,
            'type': ['Historical']*30 + ['Forecast']*7
        })
        
        fig = px.line(forecast_df, x='date', y='temperature', color='type',
                     title='7-Day Temperature Forecast',
                     labels={'date': 'Date', 'temperature': 'Avg. Temperature (°C)'})
        st.plotly_chart(fig, use_container_width=True)

def show_uhi_detection(data):
    """Display the UHI detection and analysis module"""
    st.markdown('<h2 class="sub-header">UHI Detection & Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module uses satellite imagery and street-level data to detect urban heat island hotspots. Upload your own data or use our demo data to visualize UHI patterns.</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Satellite Analysis", "Cluster Analysis", "Temporal Analysis"])
    
    with tab1:
        st.markdown("### Satellite-Based UHI Detection")
        st.write("Select an area to analyze or use the demo data:")
        
        # Input controls
        col1, col2 = st.columns(2)
        with col1:
            location = st.selectbox("Select location:", ["New York City, NY", "Custom Location"])
            lat, lon = (40.7128, -74.0060) if location == "New York City, NY" else (
                st.number_input("Latitude:", value=40.7128, format="%.4f"),
                st.number_input("Longitude:", value=-74.0060, format="%.4f")
            )
        
        with col2:
            analysis_date = st.date_input("Select date for analysis:", datetime.date(2025, 6, 1))
            data_source = st.selectbox("Data source:", ["Landsat 9", "Sentinel-2", "MODIS"])
        
        # Run analysis
        if st.button("Run Satellite Analysis"):
            st.markdown("#### Analysis Results")
            
            # Metrics
            cols = st.columns(3)
            with cols[0]:
                ndvi = get_satellite_ndvi_data(lat, lon, analysis_date)
                st.metric("NDVI Index", f"{ndvi:.2f}", "-0.05")
            with cols[1]:
                surface_temp = data[data['latitude'].between(lat-0.01, lat+0.01) & 
                                    data['longitude'].between(lon-0.01, lon+0.01)]['surface_temperature'].mean()
                st.metric("Surface Temperature", f"{surface_temp:.1f}°C", "+2.3°C")
            with cols[2]:
                building_density = data[data['latitude'].between(lat-0.01, lat+0.01) & 
                                        data['longitude'].between(lon-0.01, lon+0.01)]['building_density'].mean()
                st.metric("Building Density", f"{building_density:.2f}", "+0.04")
            
            # Heat map
            st.markdown("#### Surface Temperature Map")
            area_data = data[data['latitude'].between(lat-0.03, lat+0.03) & data['longitude'].between(lon-0.03, lon+0.03)]
            m = folium.Map(location=[lat, lon], zoom_start=14, tiles='CartoDB positron')
            heat_data = [[row['latitude'], row['longitude'], row['surface_temperature']] for _, row in area_data.iterrows()]
            HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'},
                   min_opacity=0.5, blur=10).add_to(m)
            folium.Marker([lat, lon], popup=f"Selected Location<br>NDVI: {ndvi:.2f}<br>Temp: {surface_temp:.1f}°C",
                         icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
            folium_static(m)
            
            # NDVI vs Temperature
            st.markdown("#### NDVI vs Surface Temperature")
            fig = px.scatter(area_data, x='vegetation_index', y='surface_temperature', color='surface_temperature',
                            color_continuous_scale='Thermal', title='Vegetation Index vs Surface Temperature',
                            labels={'vegetation_index': 'Vegetation Index (NDVI)', 
                                   'surface_temperature': 'Surface Temperature (°C)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis summary
            st.markdown("#### Analysis Summary")
            avg_temp = area_data['surface_temperature'].mean()
            uhi_severity = "Severe" if avg_temp > 30 else "Moderate" if avg_temp > 27 else "Low"
            impact = "High" if avg_temp > 30 else "Medium" if avg_temp > 27 else "Low"
            
            st.markdown(f"""
            **UHI Severity:** {uhi_severity}  
            **Potential Impact:** {impact}  
            **Key Factors:**
            - Building density contributes approximately {(building_density * 100):.1f}% to the UHI effect
            - Vegetation cover is {(area_data['vegetation_index'].mean() * 100):.1f}% of the analyzed area
            - Average surface temperature is {avg_temp:.1f}°C, which is {avg_temp - 25:.1f}°C above the baseline temperature
            """)
    
    with tab2:
        st.markdown("### UHI Cluster Analysis")
        st.write("This analysis identifies similar urban areas based on their heat characteristics.")
        
        # Clustering
        n_clusters = st.slider("Number of clusters:", min_value=3, max_value=7, value=5)
        cluster_map, clustered_data = create_cluster_map(data, n_clusters)
        
        st.markdown("#### Urban Heat Clusters")
        folium_static(cluster_map)
        
        # Cluster characteristics
        st.markdown("#### Cluster Characteristics")
        cluster_means = clustered_data.groupby('cluster')[
            ['building_density', 'vegetation_index', 'albedo', 'air_temperature']
        ].mean().reset_index()
        
        # Parallel coordinates plot
        fig = px.parallel_coordinates(cluster_means,
                                     dimensions=['building_density', 'vegetation_index', 'albedo', 'air_temperature'],
                                     color='cluster',
                                     labels={'building_density': 'Building Density', 'vegetation_index': 'Vegetation',
                                            'albedo': 'Albedo', 'air_temperature': 'Temperature (°C)'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster summary
        st.markdown("#### Cluster Summary")
        cluster_descriptions = []
        # Cluster summary
        st.markdown("#### Cluster Summary")
        cluster_descriptions = []
        for _, row in cluster_means.iterrows():
            if row['air_temperature'] > 28 and row['building_density'] > 0.6:
                description = "Urban Core - Hot Spot"
            elif row['vegetation_index'] > 0.6:
                description = "Green Zone - Cool Area"
            elif row['albedo'] > 0.4:
                description = "High Reflectivity Zone"
            elif row['building_density'] > 0.5 and row['vegetation_index'] < 0.3:
                description = "Dense Urban - Moderate Heat"
            else:
                description = "Mixed Urban Zone"
            cluster_descriptions.append(description)
        
        # Display cluster summary
        cluster_means['description'] = cluster_descriptions
        display_df = cluster_means.copy()
        display_df.columns = ['Cluster', 'Building Density', 'Vegetation', 'Albedo', 'Temperature (°C)', 'Description']
        display_df = display_df[['Cluster', 'Description', 'Temperature (°C)', 'Building Density', 'Vegetation', 'Albedo']]
        st.dataframe(display_df.round(2))
    
    with tab3:
        st.markdown("### Temporal UHI Analysis")
        st.write("Analyze how UHI patterns change over time.")
        
        # Time period selection
        period = st.selectbox("Select analysis period:", ["Daily Cycle", "Seasonal Variation", "Annual Trend"])
        
        if period == "Daily Cycle":
            # Generate data for daily cycle
            hours = list(range(24))
            urban_temps = [25 + 5 * np.sin((h - 2) * np.pi / 24) for h in hours]
            rural_temps = [22 + 4 * np.sin((h - 2) * np.pi / 24) for h in hours]
            daily_df = pd.DataFrame({
                'hour': hours,
                'urban_temperature': urban_temps,
                'rural_temperature': rural_temps,
                'uhi_intensity': [u - r for u, r in zip(urban_temps, rural_temps)]
            })
            
            # Temperature comparison
            fig = px.line(daily_df, x='hour', y=['urban_temperature', 'rural_temperature'],
                         title='Daily Temperature Cycle: Urban vs Rural',
                         labels={'hour': 'Hour of Day', 'value': 'Temperature (°C)', 'variable': 'Location'})
            st.plotly_chart(fig, use_container_width=True)
            
            # UHI intensity
            fig = px.bar(daily_df, x='hour', y='uhi_intensity', title='Urban Heat Island Intensity Throughout the Day',
                        labels={'hour': 'Hour of Day', 'uhi_intensity': 'UHI Intensity (°C)'},
                        color='uhi_intensity', color_continuous_scale='Thermal')
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis
            max_uhi_hour = daily_df.loc[daily_df['uhi_intensity'].idxmax(), 'hour']
            max_uhi = daily_df['uhi_intensity'].max()
            
            st.markdown(f"""
            #### Key Findings:
            - Maximum UHI intensity of {max_uhi:.1f}°C occurs at {int(max_uhi_hour):02d}:00 hours
            - UHI effect is strongest during night and early morning hours
            - Minimum temperature difference observed during mid-day
            
            This pattern is typical of urban areas where built surfaces release stored heat during the night,
            while rural areas cool more rapidly after sunset.
            """)
        
        elif period == "Seasonal Variation":
            # Generate data for seasonal variation
            months = list(range(1, 13))
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            summer_peak = 7  # July
            urban_temps = [25 + 10 * np.sin((m - 1) * np.pi / 6) for m in months]
            rural_temps = [22 + 12 * np.sin((m - 1) * np.pi / 6) for m in months]
            uhi_intensity = [3 + 1.5 * np.sin((m - summer_peak) * np.pi / 6) for m in months]
            
            seasonal_df = pd.DataFrame({
                'month': month_names, 'month_num': months,
                'urban_temperature': urban_temps, 'rural_temperature': rural_temps,
                'uhi_intensity': uhi_intensity
            })
            
            # Temperature comparison
            fig = px.line(seasonal_df, x='month', y=['urban_temperature', 'rural_temperature'],
                         title='Seasonal Temperature Variation: Urban vs Rural',
                         labels={'month': 'Month', 'value': 'Temperature (°C)', 'variable': 'Location'})
            fig.update_xaxes(categoryorder='array', categoryarray=month_names)
            st.plotly_chart(fig, use_container_width=True)
            
            # UHI intensity
            fig = px.bar(seasonal_df, x='month', y='uhi_intensity', title='Urban Heat Island Intensity by Season',
                        labels={'month': 'Month', 'uhi_intensity': 'UHI Intensity (°C)'},
                        color='uhi_intensity', color_continuous_scale='Thermal')
            fig.update_xaxes(categoryorder='array', categoryarray=month_names)
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis
            max_uhi_month = seasonal_df.loc[seasonal_df['uhi_intensity'].idxmax(), 'month']
            max_uhi = seasonal_df['uhi_intensity'].max()
            
            st.markdown(f"""
            #### Key Findings:
            - Maximum UHI intensity of {max_uhi:.1f}°C occurs in {max_uhi_month}
            - Summer months generally show higher UHI intensity
            - Urban and rural temperature gap varies by season
            """)
        
        elif period == "Annual Trend":
            # Generate data for annual trend
            years = list(range(2020, 2026))
            base_uhi = [2.8, 3.0, 3.3, 3.5, 3.7, 4.0]
            uhi_trend = [b + np.random.normal(0, 0.1) for b in base_uhi]
            
            trend_df = pd.DataFrame({'year': years, 'uhi_intensity': uhi_trend})
            
            # Plot trend
            fig = px.line(trend_df, x='year', y='uhi_intensity',
                         title='Urban Heat Island Intensity Annual Trend',
                         labels={'year': 'Year', 'uhi_intensity': 'Average UHI Intensity (°C)'},
                         markers=True)
            
            # Add trendline
            fig.add_trace(px.scatter(trend_df, x='year', y='uhi_intensity', trendline='ols').data[1])
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis
            initial_uhi = trend_df.iloc[0]['uhi_intensity']
            final_uhi = trend_df.iloc[-1]['uhi_intensity']
            percent_change = ((final_uhi - initial_uhi) / initial_uhi) * 100
            
            st.markdown(f"""
            #### Key Findings:
            - UHI intensity has increased by {percent_change:.1f}% over the past 6 years
            - Average annual increase of {(final_uhi - initial_uhi) / 5:.2f}°C per year
            - If this trend continues, UHI intensity could reach {final_uhi + (final_uhi - initial_uhi) / 5 * 5:.1f}°C by 2030
            """)

def show_temperature_prediction(data):
    """Display the temperature prediction module"""
    st.markdown('<h2 class="sub-header">Temperature Prediction Model</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module uses machine learning to predict hyperlocal temperature variations based on urban features. Adjust the parameters to see how different urban configurations affect local temperatures.</div>', unsafe_allow_html=True)
    
    # Create layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Parameter inputs
        st.markdown("### Urban Feature Parameters")
        building_density = st.slider("Building Density:", 0.0, 1.0, 0.5, 0.01,
                                   help="Higher values indicate more dense urban development")
        vegetation_index = st.slider("Vegetation Index:", 0.0, 1.0, 0.3, 0.01,
                                   help="Higher values indicate more vegetation (trees, parks, etc.)")
        albedo = st.slider("Surface Albedo:", 0.0, 1.0, 0.2, 0.01,
                          help="Higher values indicate more reflective surfaces")
        
        st.markdown("### Additional Factors")
        land_use = st.selectbox("Land Use Type:", ["Residential", "Commercial", "Industrial", "Park", "Mixed Use"])
        time_of_day = st.select_slider("Time of Day:", 
                                      options=["Early Morning", "Morning", "Noon", "Afternoon", "Evening", "Night"])
        season = st.selectbox("Season:", ["Winter", "Spring", "Summer", "Fall"])
        predict_button = st.button("Predict Temperature")
    
    with col2:
        if predict_button:
            # Prepare prediction
            features = {'building_density': building_density, 'vegetation_index': vegetation_index, 'albedo': albedo}
            predicted_temp = get_temperature_prediction(features)
            
            # Adjustments
            time_factors = {"Early Morning": -2.0, "Morning": -0.5, "Noon": 1.5, 
                           "Afternoon": 2.0, "Evening": 0.0, "Night": -1.5}
            season_factors = {"Winter": -5.0, "Spring": 0.0, "Summer": 5.0, "Fall": 0.0}
            land_use_factors = {"Residential": 0.0, "Commercial": 1.0, "Industrial": 1.5, 
                               "Park": -2.0, "Mixed Use": 0.5}
            
            predicted_temp += time_factors[time_of_day] + season_factors[season] + land_use_factors[land_use]
            
            # Display prediction
            st.markdown(f"""
            <div style="text-align:center; margin:20px; padding:20px; background-color:#f8f9fa; 
            border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
                <h3>Predicted Temperature</h3>
                <div style="font-size:48px; font-weight:bold; color:#e74c3c;">{predicted_temp:.1f}°C</div>
                <p>Based on the urban features and conditions you specified</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparison to baseline
            baseline_temp = 25.0
            temp_difference = predicted_temp - baseline_temp
            st.markdown(f"""
            <div style="margin:20px 0;">
                <strong>Comparison to Baseline:</strong> {temp_difference:.1f}°C 
                {'warmer' if temp_difference > 0 else 'cooler'} than the baseline temperature
            </div>
            """, unsafe_allow_html=True)
            
            # Feature impact analysis
            st.markdown("### Feature Impact Analysis")
            building_impact = 5 * building_density
            vegetation_impact = -4 * vegetation_index
            albedo_impact = -3 * albedo
            
            impact_data = pd.DataFrame({
                'Feature': ['Building Density', 'Vegetation', 'Albedo', 'Time of Day', 'Season', 'Land Use'],
                'Impact': [building_impact, vegetation_impact, albedo_impact, 
                          time_factors[time_of_day], season_factors[season], land_use_factors[land_use]]
            })
            
            impact_data['Abs_Impact'] = impact_data['Impact'].abs()
            impact_data = impact_data.sort_values('Abs_Impact', ascending=False)
            
            # Impact visualization
            fig = px.bar(impact_data, y='Feature', x='Impact', orientation='h',
                        title='Feature Impact on Temperature', color='Impact',
                        color_continuous_scale='RdBu_r', labels={'Impact': 'Temperature Impact (°C)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### Recommendations")
            if predicted_temp > 28:
                st.markdown("""
                #### High Temperature Zone
                This configuration results in elevated temperatures that could contribute to urban heat island effects.
                Consider the following mitigation strategies:
                - **Increase vegetation coverage**: Adding trees, green roofs, or pocket parks could reduce temperature by 1-2°C
                - **Implement cool surfaces**: Replacing dark surfaces with high-albedo materials
                - **Redesign urban geometry**: Modify building arrangements to improve air flow
                """)
            elif predicted_temp > 25:
                st.markdown("""
                #### Moderate Temperature Zone
                This configuration shows a moderate urban heat island effect. Some improvements could help:
                - **Enhance existing green spaces**: Increase vegetation in available areas
                - **Gradual surface replacements**: Consider light-colored materials for upcoming renovations
                - **Water features**: Small water elements could provide localized cooling
                """)
            else:
                st.markdown("""
                #### Low Temperature Zone
                This configuration effectively minimizes urban heat island effects. To maintain:
                - **Preserve existing vegetation**: Protect and maintain current green spaces
                - **Continue high-albedo practices**: Maintain reflective surfaces when replacing materials
                - **Use as model**: Consider applying similar configurations to other urban areas
                """)
        else:
            # Placeholder content
            st.markdown("""
            ### Temperature Prediction
            Adjust the parameters on the left and click "Predict Temperature" to see results.
            
            The prediction model considers:
            - Building density
            - Vegetation coverage
            - Surface reflectivity (albedo)
            - Land use type
            - Time of day and seasonal factors
            """)
            
            # Example visualization
            building_range = np.linspace(0, 1, 50)
            vegetation_range = np.linspace(0, 1, 50)
            X, Y = np.meshgrid(building_range, vegetation_range)
            Z = 25 + 5 * X - 4 * Y  # Simplified temperature model
            
            fig = go.Figure(data=[go.Contour(
                z=Z, x=building_range, y=vegetation_range, colorscale='Thermal',
                contours=dict(start=22, end=30, size=0.5, showlabels=True),
                colorbar=dict(title='Temperature (°C)', titleside='right')
            )])
            
            fig.update_layout(title='Temperature Prediction by Building Density and Vegetation',
                             xaxis_title='Building Density', yaxis_title='Vegetation Index', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            The contour plot above shows how temperature varies with building density and vegetation coverage.
            - **Higher building density** tends to increase temperature (moving right on the x-axis)
            - **Higher vegetation** tends to decrease temperature (moving up on the y-axis)
            - The contour lines represent equal temperature values
            """)

def show_intervention_planning(data):
    """Display the intervention planning module"""
    st.markdown('<h2 class="sub-header">UHI Intervention Planning</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module helps planners identify the most effective UHI mitigation strategies for specific urban areas. Select a location and the system will analyze local conditions and recommend tailored interventions.</div>', unsafe_allow_html=True)
    
    # Create layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Select Location")
        location_method = st.radio("Selection method:", ["Map Selection", "Address Search"])
        
        # Location input
        if location_method == "Map Selection":
            st.write("Map will appear here in a real application.")
            lat = st.number_input("Latitude:", value=40.7128, format="%.4f")
            lon = st.number_input("Longitude:", value=-74.0060, format="%.4f")
        else:
            address = st.text_input("Enter address:", "Times Square, New York, NY")
            if address:
                st.write("Address geocoded to coordinates:")
                lat, lon = 40.7580, -73.9855  # Times Square coordinates
                st.write(f"Latitude: {lat}, Longitude: {lon}")
            else:
                lat, lon = 40.7128, -74.0060  # Default NYC coordinates
        
        # Parameters
        radius = st.slider("Analysis radius (km):", 0.5, 5.0, 1.0, 0.5)
        priority = st.selectbox("Optimization priority:", ["Temperature Reduction", "Cost Efficiency", 
                                                          "Implementation Speed", "Balanced Approach"])
        
        # Constraints
        st.markdown("### Constraints")
        budget_constraint = st.select_slider("Budget level:", options=["Low", "Medium", "High"])
        time_constraint = st.select_slider("Implementation timeframe:", 
                                          options=["Short-term", "Medium-term", "Long-term"])
        
        analyze_button = st.button("Analyze & Suggest Interventions")
    
    with col2:
        if analyze_button:
            st.markdown("### Analysis Results")
            
            # Get recommendations
            location = (lat, lon)
            intervention_results = suggest_interventions(data, location)
            
            # Display local conditions
            st.markdown("#### Local Conditions")
            metric_cols = st.columns(4)
            with metric_cols[0]: st.metric("Temperature", intervention_results["local_temperature"], "+3.2°C")
            with metric_cols[1]: st.metric("Building Density", intervention_results["building_density"], "+0.15")
            with metric_cols[2]: st.metric("Vegetation Index", intervention_results["vegetation_index"], "-0.08")
            with metric_cols[3]: st.metric("Albedo", intervention_results["albedo"], "-0.12")
            
            # Area map
            st.markdown("#### Area Map")
            area_data = data[(data['latitude'] - lat)**2 + (data['longitude'] - lon)**2 <= (radius/111)**2]
            
            m = folium.Map(location=[lat, lon], zoom_start=14, tiles='CartoDB positron')
            heat_data = [[row['latitude'], row['longitude'], row['air_temperature']] for _, row in area_data.iterrows()]
            HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'},
                   min_opacity=0.5, blur=10).add_to(m)
            
            folium.Marker([lat, lon], popup=f"Selected Location<br>Temp: {intervention_results['local_temperature']}",
                         icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
            
            folium.Circle(location=[lat, lon], radius=radius * 1000, color='blue',
                         fill=True, fill_opacity=0.1).add_to(m)
            
            folium_static(m)
            
            # Intervention recommendations
            st.markdown("#### Recommended Interventions")
            
            if len(intervention_results["suggestions"]) > 0:
                # Create tabs for different intervention types
                intervention_types = list(set(s["type"] for s in intervention_results["suggestions"]))
                tabs = st.tabs(intervention_types + ["All Interventions"])
                
                for i, tab in enumerate(tabs):
                    with tab:
                        if i < len(intervention_types):
                            # Filter for this type
                            current_type = intervention_types[i]
                            type_suggestions = [s for s in intervention_results["suggestions"] 
                                               if s["type"] == current_type]
                            
                            for j, suggestion in enumerate(type_suggestions):
                                st.markdown(f"""
                                <div style="margin-bottom:15px; padding:15px; 
                                background-color:{'#e8f4f8' if j % 2 == 0 else '#f0f7fa'}; border-radius:5px;">
                                    <h4>{suggestion["description"]}</h4>
                                    <p><strong>Impact:</strong> {suggestion["impact"]}</p>
                                    <p><strong>Priority:</strong> {suggestion["priority"]}</p>
                                    <p><strong>Cost:</strong> {suggestion["cost_estimate"]}</p>
                                    <p><strong>Implementation:</strong> {suggestion["implementation_time"]}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            # Show all interventions
                            for j, suggestion in enumerate(intervention_results["suggestions"]):
                                st.markdown(f"""
                                <div style="margin-bottom:15px; padding:15px; 
                                background-color:{'#e8f4f8' if j % 2 == 0 else '#f0f7fa'}; border-radius:5px;">
                                    <h4>{suggestion["type"]}: {suggestion["description"]}</h4>
                                    <p><strong>Impact:</strong> {suggestion["impact"]}</p>
                                    <p><strong>Priority:</strong> {suggestion["priority"]}</p>
                                    <p><strong>Cost:</strong> {suggestion["cost_estimate"]}</p>
                                    <p><strong>Implementation:</strong> {suggestion["implementation_time"]}</p>
                                </div>
                                """, unsafe_allow_html=True)
            else:
                st.write("No specific interventions found for this location.")
            
            # Impact visualization
            if len(intervention_results["suggestions"]) > 0:
                st.markdown("#### Intervention Impact Visualization")
                
                # Select intervention to simulate
                intervention_options = [f"{s['type']}: {s['description']}" 
                                       for s in intervention_results["suggestions"]]
                selected_intervention = st.selectbox("Select intervention to visualize:", intervention_options)
                selected_type = selected_intervention.split(":")[0]
                
                # Intensity slider
                intensity = st.slider("Implementation intensity:", 0.1, 1.0, 0.5, 0.1,
                                     help="Higher values represent more extensive implementation")
                
                # Simulate impact
                new_data = simulate_intervention_impact(area_data, selected_type, intensity)
                
                # Impact statistics
                original_avg_temp = area_data['air_temperature'].mean()
                new_avg_temp = new_data['air_temperature'].mean()
                temp_reduction = original_avg_temp - new_avg_temp
                
                st.markdown(f"""
                #### Projected Impact
                **Temperature reduction:** {temp_reduction:.2f}°C  
                **Original average temperature:** {original_avg_temp:.2f}°C  
                **New average temperature:** {new_avg_temp:.2f}°C  
                """)
                
                # Before/after visualization
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=area_data['air_temperature'], name='Before Intervention',
                                          opacity=0.75, marker=dict(color='rgba(231, 76, 60, 0.7)')))
                fig.add_trace(go.Histogram(x=new_data['air_temperature'], name='After Intervention',
                                          opacity=0.75, marker=dict(color='rgba(46, 204, 113, 0.7)')))
                
                fig.update_layout(title=f'Temperature Distribution Before & After {selected_type} Implementation',
                                 xaxis_title='Temperature (°C)', yaxis_title='Frequency',
                                 barmode='overlay', bargap=0.1, height=400)
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Placeholder content
            st.markdown("""
            ### Intervention Planning
            
            Select a location and analysis parameters, then click "Analyze & Suggest Interventions" 
            to receive customized UHI mitigation recommendations.
            
            The system will:
            1. Analyze local urban characteristics
            2. Identify key contributors to UHI
            3. Recommend targeted interventions
            4. Visualize potential impact
            """)
            
            # Example strategies
            st.markdown("### Sample Intervention Strategies")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                #### Green Infrastructure
                - Tree planting programs
                - Green roofs and walls
                - Urban parks and green spaces
                - Vegetation corridors
                
                #### Cool Materials
                - High-albedo roofing
                - Reflective pavements
                - Cool building materials
                - Permeable surfaces
                """)
            
            with col2:
                st.markdown("""
                #### Urban Design
                - Building orientation
                - Street canyon modifications
                - Air flow optimization
                - Shade structures
                
                #### Water Features
                - Fountains and spray parks
                - Retention ponds
                - Urban streams restoration
                - Blue roofs
                """)
def show_intervention_planning(data):
    """Display the intervention planning module"""
    st.markdown('<h2 class="sub-header">UHI Intervention Planning</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module helps planners identify the most effective UHI mitigation strategies for specific urban areas in Nagpur. Select a location and the system will analyze local conditions and recommend tailored interventions.</div>', unsafe_allow_html=True)
    
    # Create layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Select Location")
        location_method = st.radio("Selection method:", ["Map Selection", "Address Search"])
        
        # Location input
        if location_method == "Map Selection":
            st.write("Map will appear here in a real application.")
            lat = st.number_input("Latitude:", value=21.1458, format="%.4f")
            lon = st.number_input("Longitude:", value=79.0882, format="%.4f")
        else:
            address = st.text_input("Enter address:", "Sitabuldi, Nagpur")
            if address:
                st.write("Address geocoded to coordinates:")
                lat, lon = 21.1424, 79.0839  # Sitabuldi coordinates
                st.write(f"Latitude: {lat}, Longitude: {lon}")
            else:
                lat, lon = 21.1458, 79.0882  # Default Nagpur coordinates
        
        # Parameters
        radius = st.slider("Analysis radius (km):", 0.5, 5.0, 1.0, 0.5)
        priority = st.selectbox("Optimization priority:", ["Temperature Reduction", "Cost Efficiency", 
                                                          "Implementation Speed", "Balanced Approach"])
        
        # Constraints
        st.markdown("### Constraints")
        budget_constraint = st.select_slider("Budget level:", options=["Low", "Medium", "High"])
        time_constraint = st.select_slider("Implementation timeframe:", 
                                          options=["Short-term", "Medium-term", "Long-term"])
        
        analyze_button = st.button("Analyze & Suggest Interventions")
    
    with col2:
        if analyze_button:
            st.markdown("### Analysis Results")
            
            # Get recommendations
            location = (lat, lon)
            intervention_results = suggest_interventions(data, location)
            
            # Display local conditions
            st.markdown("#### Local Conditions")
            metric_cols = st.columns(4)
            with metric_cols[0]: st.metric("Temperature", intervention_results["local_temperature"], "+4.2°C")
            with metric_cols[1]: st.metric("Building Density", intervention_results["building_density"], "+0.15")
            with metric_cols[2]: st.metric("Vegetation Index", intervention_results["vegetation_index"], "-0.08")
            with metric_cols[3]: st.metric("Albedo", intervention_results["albedo"], "-0.12")
            
            # Area map
            st.markdown("#### Area Map")
            area_data = data[(data['latitude'] - lat)**2 + (data['longitude'] - lon)**2 <= (radius/111)**2]
            
            m = folium.Map(location=[lat, lon], zoom_start=14, tiles='CartoDB positron')
            heat_data = [[row['latitude'], row['longitude'], row['air_temperature']] for _, row in area_data.iterrows()]
            HeatMap(heat_data, radius=15, gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'},
                   min_opacity=0.5, blur=10).add_to(m)
            
            folium.Marker([lat, lon], popup=f"Selected Location<br>Temp: {intervention_results['local_temperature']}",
                         icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
            
            folium.Circle(location=[lat, lon], radius=radius * 1000, color='blue',
                         fill=True, fill_opacity=0.1).add_to(m)
            
            folium_static(m)
            
            # Intervention recommendations
            st.markdown("#### Recommended Interventions")
            
            if len(intervention_results["suggestions"]) > 0:
                # Create tabs for different intervention types
                intervention_types = list(set(s["type"] for s in intervention_results["suggestions"]))
                tabs = st.tabs(intervention_types + ["All Interventions"])
                
                for i, tab in enumerate(tabs):
                    with tab:
                        if i < len(intervention_types):
                            # Filter for this type
                            current_type = intervention_types[i]
                            type_suggestions = [s for s in intervention_results["suggestions"] 
                                               if s["type"] == current_type]
                            
                            for j, suggestion in enumerate(type_suggestions):
                                st.markdown(f"""
                                <div style="margin-bottom:15px; padding:15px; 
                                background-color:{'#e8f4f8' if j % 2 == 0 else '#f0f7fa'}; border-radius:5px;">
                                    <h4>{suggestion["description"]}</h4>
                                    <p><strong>Impact:</strong> {suggestion["impact"]}</p>
                                    <p><strong>Priority:</strong> {suggestion["priority"]}</p>
                                    <p><strong>Cost:</strong> {suggestion["cost_estimate"].replace('def show_uhi_detection(data):
    """Display the UHI detection and analysis module"""
    st.markdown('<h2 class="sub-header">UHI Detection & Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module uses ground temperature measurements and urban feature analysis to detect urban heat island hotspots in Nagpur.</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Temperature Analysis", "Cluster Analysis"])
    
    with tab1:
        st.markdown("### Ground Temperature Analysis")
        st.write("Select an area to analyze or use the demo data:")
        
        # Input controls
        col1, col2 = st.columns(2)
        with col1:
            location = st.selectbox("Select location:", ["Nagpur City Center", "Dharampeth", "Sadar", "Custom Location"])
            if location == "Nagpur City Center":
                lat, lon = 21.1458, 79.0882
            elif location == "Dharampeth":
                lat, lon = 21.1350, 79.0650
            elif location == "Sadar":
                lat, lon = 21.1530, 79.0800
            else:  # Custom Location
                lat = st.number_input("Latitude:", value=21.1458, format="%.4f")
                lon = st.number_input("Longitude:", value=79.0882, format="%.4f")
        
        with col2:
            analysis_date = st.date_input("Select date for analysis:", datetime.date(2025, 6, 1))
            measurement_type = st.selectbox("Measurement type:", ["Ground Temperature", "Surface Temperature"])
        
        # Run analysis
        if st.button("Run Temperature Analysis"):
            st.markdown("#### Analysis Results")
            
            # Metrics
            cols = st.columns(3)
            with cols[0]:
                ground_temp = get_ground_temperature_data(lat, lon, analysis_date)
                st.metric("Ground Temperature", f"{ground_temp:.1f}°C", "+3.2°C")
            with cols[1]:
                surface_temp = data[data['latitude'].between(lat-0.01, lat+0.01) & 
                                   data['longitude'].between(lon-0.01, lon+0.01)]['surface_temperature'].mean()
                st.metric("Surface Temperature", f"{surface_temp:.1f}°C", "+4.5°C")
            with cols[2]:
                building_density = data[data['latitude'].between(lat-0.01, lat+0.01) & 
                                       data['longitude'].between(lon-0.01, lon+0.01)]['building_density'].mean()
                st.metric("Building Density", f"{building_density:.2f}", "+0.04")
            
            # Heat map
            st.markdown("#### Temperature Map")
            area_data = data[data['latitude'].between(lat-0.03, lat+0.03) & data['longitude'].between(lon-0.03, lon+0.03)]
            m = folium.Map(location=[lat, lon], zoom_start=14, tiles='CartoDB positron')
            heat_data = [[row['latitude'], row['longitude'], row['surface_temperature']] for _, row in area_data.iterrows()]def show_optimization(data):
    """Display the optimization and simulation module"""
    st.markdown('<h2 class="sub-header">Optimization & Simulation</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">This module uses optimization algorithms to find the most effective allocation of resources for UHI mitigation. It allows planners to simulate different scenarios and compare outcomes.</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Resource Optimization", "Scenario Simulation"])
    
    with tab1:
        st.markdown("### Resource Allocation Optimization")
        
        # Inputs
        budget_level = st.select_slider("Budget level:", options=["low", "medium", "high"])
        priority = st.selectbox("Optimization priority:", ["temperature", "cost", "implementation"])
        
        if st.button("Run Optimization"):
            # Get optimization results
            results = optimize_interventions(data, budget_level, priority)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Budget", f"${results['budget']}")
            with col2: st.metric("Used Budget", f"${results['used_budget']}", 
                               f"{results['used_budget']/results['budget']*100:.1f}%")
            with col3: st.metric("Temperature Reduction", f"{results['estimated_temperature_reduction']:.2f}°C")
            
            # Resource allocation chart
            st.markdown("#### Resource Allocation")
            allocation_df = pd.DataFrame(results['allocation'])
            
            fig = px.bar(allocation_df, x='cost', y='name', orientation='h', color='temperature_reduction',
                        color_continuous_scale='Blues', title='Intervention Resource Allocation',
                        labels={'cost': 'Budget Allocation ($)', 'name': 'Intervention',
                               'temperature_reduction': 'Temp. Reduction (°C)'}, text='units')
            fig.update_traces(texttemplate='%{text} units', textposition='inside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Type distribution
            type_summary = allocation_df.groupby('type').agg({
                'cost': 'sum', 'temperature_reduction': 'sum', 'units': 'sum'
            }).reset_index()
            
            fig = px.pie(type_summary, values='cost', names='type',
                        title='Budget Distribution by Intervention Type', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cost-effectiveness analysis
            st.markdown("#### Cost-Effectiveness Analysis")
            allocation_df['cost_per_degree'] = allocation_df['cost'] / allocation_df['temperature_reduction']
            cost_effectiveness = allocation_df.sort_values('cost_per_degree')
            
            fig = px.bar(cost_effectiveness, x='name', y='cost_per_degree', color='type',
                        title='Cost per Degree of Cooling ($ / °C)',
                        labels={'cost_per_degree': 'Cost per °C Reduction ($)', 'name': 'Intervention'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Implementation timeline
            st.markdown("#### Implementation Timeline")
            
            # Create simple timeline data
            timeline_data = []
            start_date = datetime.date(2025, 7, 1)
            
            for i, row in allocation_df.iterrows():
                # Assign durations based on type
                if row['type'] == 'Green Infrastructure': duration = 90
                elif row['type'] == 'High-Albedo Surfaces': duration = 60
                elif row['type'] == 'Water Features': duration = 120
                else: duration = 30
                
                end_date = start_date + datetime.timedelta(days=duration)
                timeline_data.append({
                    'Task': row['name'], 'Start': start_date, 'Finish': end_date, 'Type': row['type']
                })
                start_date = start_date + datetime.timedelta(days=30)
            
            timeline_df = pd.DataFrame(timeline_data)
            
            fig = px.timeline(timeline_df, x_start='Start', x_end='Finish', y='Task',
                             color='Type', title='Implementation Timeline')
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Future Scenario Simulation")
        
        # Inputs
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Scenario Parameters")
            year = st.slider("Target Year:", 2025, 2050, 2030, 5)
            urban_growth = st.select_slider("Urban Growth Rate:", options=["Low", "Medium", "High"])
            climate_scenario = st.select_slider("Climate Change Scenario:", 
                                               options=["Optimistic", "Moderate", "Pessimistic"])
            mitigation_level = st.select_slider("UHI Mitigation Implementation:", 
                                              options=["Minimal", "Moderate", "Aggressive"])
            simulate_button = st.button("Run Simulation")
        
        with col2:
            if simulate_button:
                st.markdown("#### Simulation Results")
                
                # Calculate climate scenario impacts
                if climate_scenario == "Optimistic":
                    climate_increase = 0.5 * (year - 2025) / 5  # 0.5°C per 5 years
                elif climate_scenario == "Moderate":
                    climate_increase = 1.0 * (year - 2025) / 5  # 1.0°C per 5 years
                else:  # Pessimistic
                    climate_increase = 1.5 * (year - 2025) / 5  # 1.5°C per 5 years
                
                # Calculate urban growth impacts
                if urban_growth == "Low":
                    growth_factor = 0.2 * (year - 2025) / 5  # 0.2°C per 5 years
                elif urban_growth == "Medium":
                    growth_factor = 0.5 * (year - 2025) / 5  # 0.5°C per 5 years
                else:  # High
                    growth_factor = 0.8 * (year - 2025) / 5  # 0.8°C per 5 years
                
                # Calculate mitigation effects
                if mitigation_level == "Minimal":
                    mitigation_effect = 0.2 * (year - 2025) / 5  # 0.2°C reduction per 5 years
                elif mitigation_level == "Moderate":
                    mitigation_effect = 0.7 * (year - 2025) / 5  # 0.7°C reduction per 5 years
                else:  # Aggressive
                    mitigation_effect = 1.2 * (year - 2025) / 5  # 1.2°C reduction per 5 years
                
                # Calculate UHI change
                current_uhi = data['air_temperature'].mean() - 25  # Assuming 25°C is the baseline
                future_uhi = current_uhi + climate_increase + growth_factor - mitigation_effect
                
                # Display results
                st.markdown(f"""
                #### Projected UHI Intensity for {year}
                
                **Current UHI Intensity (2025):** {current_uhi:.2f}°C  
                **Projected UHI Intensity ({year}):** {future_uhi:.2f}°C  
                
                **Contributing Factors:**
                - Climate change impact: +{climate_increase:.2f}°C
                - Urban growth impact: +{growth_factor:.2f}°C
                - Mitigation effect: -{mitigation_effect:.2f}°C
                
                **Net Change:** {future_uhi - current_uhi:.2f}°C
                """)
                
                # Waterfall chart
                waterfall_data = pd.DataFrame({
                    'Factor': ['Current UHI', 'Climate Change', 'Urban Growth', 'Mitigation', f'UHI in {year}'],
                    'Value': [current_uhi, climate_increase, growth_factor, -mitigation_effect, future_uhi],
                    'Type': ['Total', 'Increase', 'Increase', 'Decrease', 'Total']
                })
                
                fig = go.Figure(go.Waterfall(
                    name="UHI Components", orientation="v", measure=waterfall_data['Type'],
                    x=waterfall_data['Factor'], y=waterfall_data['Value'],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": "#e74c3c"}},
                    decreasing={"marker": {"color": "#2ecc71"}},
                    totals={"marker": {"color": "#3498db"}}
                ))
                
                fig.update_layout(title=f"UHI Intensity Change from 2025 to {year}",
                                 yaxis_title="Temperature Change (°C)", height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Temperature distribution comparison
                st.markdown("#### Temperature Distribution Comparison")
                
                # Create synthetic distributions
                current_temps = data['air_temperature'].values
                future_temps = current_temps + (future_uhi - current_uhi)
                future_temps += np.random.normal(0, 0.5, size=len(future_temps))  # Add variability
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=current_temps, name='Current (2025)', opacity=0.75,
                                          marker=dict(color='rgba(52, 152, 219, 0.7)')))
                fig.add_trace(go.Histogram(x=future_temps, name=f'Projected ({year})', opacity=0.75,
                                          marker=dict(color='rgba(231, 76, 60, 0.7)')))
                
                fig.update_layout(title=f'Temperature Distribution: Current vs {year}',
                                 xaxis_title='Temperature (°C)', yaxis_title='Frequency',
                                 barmode='overlay', bargap=0.1, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Heat wave risk assessment
                st.markdown("#### Heat Wave Risk Assessment")
                
                # Determine risk level
                if future_uhi > 5:
                    risk_level, risk_color = "High", "#e74c3c"
                elif future_uhi > 3:
                    risk_level, risk_color = "Medium", "#f39c12"
                else:
                    risk_level, risk_color = "Low", "#2ecc71"
                
                # Calculate additional metrics
                extreme_heat_days_current = sum(current_temps > 30) / len(current_temps) * 365
                extreme_heat_days_future = sum(future_temps > 30) / len(future_temps) * 365
                
                st.markdown(f"""
                <div style="padding:20px; background-color:{risk_color}25; border-left:5px solid {risk_color}; margin-bottom:20px;">
                    <h4>Heat Wave Risk Level: <span style="color:{risk_color}">{risk_level}</span></h4>
                    <p>
                        <strong>Days over 30°C per year:</strong><br>
                        Current (2025): {extreme_heat_days_current:.1f} days<br>
                        Projected ({year}): {extreme_heat_days_future:.1f} days<br>
                        <strong>Increase: {extreme_heat_days_future - extreme_heat_days_current:.1f} days</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("#### Recommendations")
                
                if risk_level == "High":
                    st.markdown("""
                    To address the high heat wave risk projected in this scenario:
                    
                    1. **Implement comprehensive UHI mitigation plan** with emphasis on cooling interventions
                    2. **Develop heat emergency response protocols** for vulnerable populations
                    3. **Increase green infrastructure budget** to maximize cooling effect
                    4. **Revise building codes** to mandate cool roofs and energy-efficient designs
                    5. **Create cooling centers network** accessible within 10-minute walks citywide
                    """)
                elif risk_level == "Medium":
                    st.markdown("""
                    To address the medium heat wave risk projected in this scenario:
                    
                    1. **Gradually increase green cover** in hotspot areas
                    2. **Implement cool pavement program** during regular maintenance cycles
                    3. **Develop targeted interventions** for vulnerable neighborhoods
                    4. **Create incentives for green roofs** and cool building materials
                    5. **Monitor temperature trends** and adjust strategies accordingly
                    """)
                else:
                    st.markdown("""
                    To maintain the low heat wave risk projected in this scenario:
                    
                    1. **Continue current mitigation efforts** to maintain progress
                    2. **Preserve existing green spaces** and expand when possible
                    3. **Incorporate UHI considerations** in all future development
                    4. **Monitor temperature data** to detect any unexpected changes
                    5. **Document successful strategies** to share with other cities
                    """)
            else:
                # Placeholder content
                st.markdown("""
                ### Scenario Simulation
                
                Configure the parameters on the left and click "Run Simulation" to see projections of future UHI patterns
                based on different climate change, urban growth, and mitigation scenarios.
                
                The simulation will show:
                - Projected UHI intensity changes
                - Temperature distribution shifts
                - Heat wave risk assessment
                - Tailored recommendations based on outcomes
                """)
                
                # Sample projection chart
                years = list(range(2025, 2051, 5))
                no_action = [3.0 + 0.4 * i for i in range(len(years))]
                moderate_action = [3.0 + 0.3 * i - 0.1 * i**2 for i in range(len(years))]
                aggressive_action = [3.0 + 0.2 * i - 0.15 * i**2 for i in range(len(years))]
                
                scenario_df = pd.DataFrame({
                    'Year': years * 3,
                    'UHI Intensity (°C)': no_action + moderate_action + aggressive_action,
                    'Scenario': ['No Action'] * len(years) + ['Moderate Action'] * len(years) + 
                                ['Aggressive Action'] * len(years)
                })
                
                fig = px.line(scenario_df, x='Year', y='UHI Intensity (°C)', color='Scenario',
                             title='UHI Intensity Projections by Mitigation Scenario')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

def show_about():
    """Display the about page with project information"""
    st.markdown('<h2 class="sub-header">About This Project</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>Urban Heat Island Analysis & Mitigation System</h3>
        <p>This project aims to develop an integrated AI-based system that helps city planners and environmental 
        scientists detect, analyze, and mitigate urban heat island effects through data-driven decision making.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key features
    st.markdown("### Key Features")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Detection & Analysis
        - Satellite imagery analysis for UHI detection
        - Street-level temperature mapping
        - Temporal and spatial pattern identification
        - Cluster analysis of similar urban areas
        
        #### Prediction
        - Machine learning models for temperature prediction
        - Impact assessment of new developments
        - Future scenario simulation
        - Climate change integration
        """)
    
    with col2:
        st.markdown("""
        #### Intervention Planning
        - Customized intervention recommendations
        - Cost-benefit analysis of strategies
        - Implementation priority ranking
        - Visualization of potential impacts
        
        #### Optimization
        - Resource allocation optimization
        - Multi-objective decision support
        - Budget-constrained planning
        - Scenario comparison
        """)
    
    # Technical details
    st.markdown("### Technical Details")
    st.markdown("""
    This system integrates multiple technologies and data sources:
    
    - **Satellite Data**: Utilizes freely available Landsat, Sentinel-2, and MODIS data
    - **Machine Learning**: Employs random forest and gradient boosting models
    - **Optimization Algorithms**: Uses multi-objective optimization for planning
    - **GIS Integration**: Provides spatial analysis and mapping capabilities
    - **Simulation Models**: Enables scenario testing and future projections
    
    The application is built using Python and Streamlit, making it accessible through any web browser.
    No proprietary software or paid services are required to run the system.
    """)
    
    # Data sources
    st.markdown("### Data Sources")
    st.markdown("""
    The system can utilize data from various free sources:
    
    - NASA Earth Data (https://earthdata.nasa.gov/)
    - USGS Earth Explorer (https://earthexplorer.usgs.gov/)
    - Copernicus Open Access Hub (https://scihub.copernicus.eu/)
    - OpenStreetMap (https://www.openstreetmap.org/)
    - National Weather Service (https://www.weather.gov/)
    - Local municipal GIS data portals
    
    For demonstration purposes, this app uses synthetic data that simulates realistic urban temperature patterns.
    """)

# ---------- MAIN APPLICATION ----------
def main():
    """Main application entry point"""
    # Setup page
    setup_page()
    
    # Page header
    st.markdown('<h1 class="main-header">Urban Heat Island Analysis & Mitigation System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
        This AI-based system helps city planners and environmental scientists analyze urban heat island (UHI) effects 
        and develop data-driven strategies to mitigate their impact. Using satellite imagery, environmental data, 
        and machine learning, it provides insights for sustainable urban planning.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.image("https://www.epa.gov/sites/default/files/styles/medium/public/2020-07/urban-heat-island.jpg", 
                    use_container_width=True)
    st.sidebar.title("Navigation")
    
    # Page selection
    page = st.sidebar.radio("Select a Module", 
                           ["Dashboard", "UHI Detection", "Temperature Prediction", 
                            "Intervention Planning", "Optimization & Simulation", "About"])
    
    # Load sample data
    data = load_sample_data()
    
    # Display selected page
    if page == "Dashboard": show_dashboard(data)
    elif page == "UHI Detection": show_uhi_detection(data)
    elif page == "Temperature Prediction": show_temperature_prediction(data)
    elif page == "Intervention Planning": show_intervention_planning(data)
    elif page == "Optimization & Simulation": show_optimization(data)
    elif page == "About": show_about()
    
    # Footer
    st.markdown('<div class="footer">Urban Heat Island Analysis & Mitigation System © 2025</div>', 
               unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
