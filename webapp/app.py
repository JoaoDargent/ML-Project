import streamlit as st
import pandas as pd
import plotly.express as px
import json
import pickle
from datetime import datetime
import numpy as np
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, MinMaxScaler
from feature_engineering import engineer_features
from data_scaling_encoding import encode_and_scale, select_model_features
import zipfile
import os

# Load data
@st.cache_data
def load_data():
    # Check if the data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Check if the CSV file exists
    if not os.path.exists("data/merged_with_fips.csv"):
        # Look for zip file
        if os.path.exists("data.zip"):
            try:
                with zipfile.ZipFile("data.zip", 'r') as zip_ref:
                    zip_ref.extractall(".")
                st.success("Data files successfully extracted!")
            except Exception as e:
                st.error(f"Error extracting data files: {str(e)}")
                return None
        else:
            st.error("Neither data/merged_with_fips.csv nor data.zip found. Please ensure one of them exists.")
            return None
    
    try:
        df = pd.read_csv("data/merged_with_fips.csv", index_col=0)
        df['fips'] = df['fips'].apply(lambda x: f"{int(float(x)):05d}" if pd.notnull(x) else None)
        # Use Accident Year directly
        df['Year'] = df['Accident Year']
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load GeoJSON
@st.cache_data
def load_geojson():
    geojson_path = "data/counties.json"
    
    # Check if the JSON file exists
    if not os.path.exists(geojson_path):
        # Look for zip file
        if os.path.exists("data.zip"):
            try:
                with zipfile.ZipFile("data.zip", 'r') as zip_ref:
                    zip_ref.extractall(".")
                st.success("GeoJSON files successfully extracted!")
            except Exception as e:
                st.error(f"Error extracting GeoJSON files: {str(e)}")
                return None
        else:
            st.error(f"Neither {geojson_path} nor data.zip found. Please ensure one of them exists.")
            return None
    
    try:
        with open(geojson_path) as response:
            counties = json.load(response)
        return counties
    except Exception as e:
        st.error(f"Error loading GeoJSON: {str(e)}")
        return None

@st.cache_resource
def load_model():
    try:
        with open("models/rf_optimized.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model file exists in the models directory.")
        return None

@st.cache_resource
def load_encoders():
    encoders = {}
    encoder_files = {
        'onehot': 'models/scalers_encoders/one_hot_encoder.pkl',
        'target': 'models/scalers_encoders/target_encoder.pkl',
        'minmax': 'models/scalers_encoders/minmax_scaler.pkl'
    }
    
    missing_files = []
    for name, filepath in encoder_files.items():
        try:
            with open(filepath, 'rb') as file:
                encoders[name] = pickle.load(file)
        except FileNotFoundError:
            missing_files.append(filepath)
    
    if missing_files:
        st.error(f"Missing encoder files: {', '.join(missing_files)}")
        return None
    
    return encoders

def aggregate_by_fips(df, feature):
    # Map display names to actual column names
    feature_mapping = {
        "Accident Count": "accident_count",
        "Age at Injury": "Age at Injury",
        "Average Weekly Wage": "Average Weekly Wage",
        "Number of Dependents": "Number of Dependents"
    }
    
    actual_feature = feature_mapping[feature]
    
    if feature == "Accident Count":
        agg_data = df.groupby("fips").size().reset_index(name=actual_feature)
    else:
        agg_data = df.groupby("fips")[actual_feature].mean().reset_index()
    
    # Add county name and state information
    county_info = df[['fips', 'county_name', 'state_abbr']].drop_duplicates()
    agg_data = agg_data.merge(county_info, on='fips', how='left')
    
    # Create a location name combining county and state
    agg_data['location'] = agg_data['county_name'] + ', ' + agg_data['state_abbr']
    
    return agg_data

@st.cache_data
def create_mappings(df):
    # Create mapping for WCIO codes to descriptions
    nature_mapping = df[['WCIO Nature of Injury Code', 'WCIO Nature of Injury Description']].drop_duplicates()
    nature_mapping = dict(zip(nature_mapping['WCIO Nature of Injury Description'], 
                            nature_mapping['WCIO Nature of Injury Code']))
    
    cause_mapping = df[['WCIO Cause of Injury Code', 'WCIO Cause of Injury Description']].drop_duplicates()
    cause_mapping = dict(zip(cause_mapping['WCIO Cause of Injury Description'], 
                           cause_mapping['WCIO Cause of Injury Code']))
    
    body_mapping = df[['WCIO Part Of Body Code', 'WCIO Part Of Body Description']].drop_duplicates()
    body_mapping = dict(zip(body_mapping['WCIO Part Of Body Description'], 
                          body_mapping['WCIO Part Of Body Code']))
    
    industry_mapping = df[['Industry Code', 'Industry Code Description']].drop_duplicates()
    industry_mapping = dict(zip(industry_mapping['Industry Code Description'], 
                              industry_mapping['Industry Code']))
    
    return {
        'nature': nature_mapping,
        'cause': cause_mapping,
        'body': body_mapping,
        'industry': industry_mapping
    }

def get_county_center(counties_geojson, fips):
    """Get the center coordinates of a county using its FIPS code."""
    for feature in counties_geojson['features']:
        # GeoJSON stores FIPS in the GEO_ID property with a prefix
        geo_id = feature['properties']['GEO_ID']
        county_fips = geo_id.replace('0500000US', '')  # Remove the prefix to get just the FIPS
        
        if county_fips == fips:
            geometry = feature['geometry']
            coordinates = geometry['coordinates']
            
            # Initialize lists to store all coordinates
            all_lons = []
            all_lats = []
            
            # Handle both Polygon and MultiPolygon types
            if geometry['type'] == 'Polygon':
                coords = coordinates[0]  # Get the outer ring
                all_lons.extend([coord[0] for coord in coords])
                all_lats.extend([coord[1] for coord in coords])
            elif geometry['type'] == 'MultiPolygon':
                for polygon in coordinates:
                    coords = polygon[0]  # Get the outer ring of each polygon
                    all_lons.extend([coord[0] for coord in coords])
                    all_lats.extend([coord[1] for coord in coords])
            
            # Calculate center
            if all_lons and all_lats:
                center_lon = sum(all_lons) / len(all_lons)
                center_lat = sum(all_lats) / len(all_lats)
                return center_lat, center_lon
    return None

def main():
    st.set_page_config(page_title="Final Project App", layout="wide")
    
    # Initialize session state for selected county
    if 'selected_county' not in st.session_state:
        st.session_state.selected_county = None
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["EDA", "Prediction"])

    # Load data and check if successful
    df = load_data()
    if df is None:
        st.error("Unable to proceed without data. Please ensure data files are available.")
        return
    
    counties = load_geojson()
    if counties is None:
        st.error("Unable to proceed without GeoJSON data. Please ensure data files are available.")
        return
    
    # Load mappings here
    mappings = create_mappings(df)

    if page == "EDA":
        st.title("Exploratory Data Analysis")
        
        # Create columns for side-by-side selection boxes
        col1, col2, col3 = st.columns([1, 1, 1])  # Changed from 4 columns to 3
        
        with col1:
            numerical_features = [
                "Accident Count",
                "Age at Injury",
                "Average Weekly Wage",
                "Number of Dependents"
            ]
            selected_feature = st.selectbox(
                "Select feature to visualize:",
                numerical_features
            )
        
        with col2:
            available_years = ["All Years"] + sorted(df['Year'].unique().tolist())
            selected_year = st.selectbox(
                "Select year:",
                available_years
            )

        with col3:
            gender_options = ["All", "Male", "Female"]
            selected_gender = st.selectbox(
                "Select gender:",
                gender_options
            )

        # Filter data based on selections
        filtered_df = df.copy()
        
        # Apply year filter
        if selected_year != "All Years":
            filtered_df = filtered_df[filtered_df['Year'] == selected_year]
            
        # Apply gender filter
        if selected_gender != "All":
            filtered_df = filtered_df[filtered_df['Gender'] == (1 if selected_gender == "Male" else 0)]
            
        # Apply county filter if selected
        if st.session_state.selected_county:
            # Extract county name from the location (removes state abbreviation)
            county_name = st.session_state.selected_county.split(',')[0].strip()
            filtered_df = filtered_df[filtered_df['County of Injury'] == county_name]

        # Aggregate data based on selected feature and filters
        plot_data = aggregate_by_fips(filtered_df, selected_feature)
        
        # Set color label
        if selected_feature == "Accident Count":
            color_label = "Number of Accidents"
        else:
            color_label = f"Average {selected_feature}"
            
        # Add year, gender, and county to title if specific selections made
        if selected_year != "All Years":
            color_label += f" ({selected_year})"
        if selected_gender != "All":
            color_label += f" - {selected_gender}"
        if st.session_state.selected_county:
            color_label += f" - {st.session_state.selected_county}"

        # Create choropleth map
        map_center = {"lat": 37.0902, "lon": -95.7129}  # Default US center
        map_zoom = 3  # Default zoom level

        # Update center and zoom if county is selected
        if st.session_state.selected_county:
            selected_fips = plot_data[plot_data['location'] == st.session_state.selected_county]['fips'].iloc[0]
            county_center = get_county_center(counties, selected_fips)
            if county_center:
                map_center = {"lat": county_center[0], "lon": county_center[1]}
                map_zoom = 8.5  # Adjusted zoom level for better county view

        # Create the base map
        fig_map = px.choropleth_mapbox(
            plot_data,
            geojson=counties,
            locations="fips",
            color="accident_count" if selected_feature == "Accident Count" else selected_feature,
            color_continuous_scale=["#fee0d2", "#fc9272", "#de2d26", "#67000d"],
            range_color=(plot_data["accident_count" if selected_feature == "Accident Count" else selected_feature].min(), 
                        plot_data["accident_count" if selected_feature == "Accident Count" else selected_feature].max()),
            mapbox_style="carto-positron",
            zoom=map_zoom,
            center=map_center,
            opacity=0.5,
            labels={"accident_count" if selected_feature == "Accident Count" else selected_feature: color_label,
                   "location": "Location"},
            hover_data={"fips": False,
                       "location": True,
                       "accident_count" if selected_feature == "Accident Count" else selected_feature: True}
        )

        # Add selected county highlight if one is selected
        if st.session_state.selected_county:
            selected_fips = plot_data[plot_data['location'] == st.session_state.selected_county]['fips'].iloc[0]
            highlight_data = plot_data[plot_data['fips'] == selected_fips].copy()
            
            # Create hover text with the selected feature value
            feature_value = highlight_data["accident_count" if selected_feature == "Accident Count" else selected_feature].iloc[0]
            if selected_feature == "Accident Count":
                hover_text = f"{highlight_data['location'].iloc[0]}<br>Number of Accidents: {feature_value:,.0f}"
            else:
                hover_text = f"{highlight_data['location'].iloc[0]}<br>Average {selected_feature}: {feature_value:,.2f}"
            
            # Add highlighted county overlay
            highlight_trace = px.choropleth_mapbox(
                highlight_data,
                geojson=counties,
                locations="fips",
                color="accident_count" if selected_feature == "Accident Count" else selected_feature,
                color_continuous_scale=["#fee0d2", "#fc9272", "#de2d26", "#67000d"],
                range_color=(plot_data["accident_count" if selected_feature == "Accident Count" else selected_feature].min(), 
                           plot_data["accident_count" if selected_feature == "Accident Count" else selected_feature].max()),
                mapbox_style="carto-positron",
                opacity=1.0,
                hover_name="location",
                custom_data=["location"]
            ).data[0]
            
            # Update hover template for the highlighted county
            highlight_trace.hovertemplate = hover_text + "<extra></extra>"
            fig_map.add_trace(highlight_trace)
            
            # Add border trace
            fig_map.add_trace(px.choropleth_mapbox(
                highlight_data,
                geojson=counties,
                locations="fips",
                color_discrete_sequence=["#ffffff"],
                mapbox_style="carto-positron",
                opacity=0.3,
            ).data[0])

        # Update layout
        fig_map.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )

        # Display the map
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Create two columns for county selector and reset button
        col_select, col_reset = st.columns([4, 1])  # 4:1 ratio to make selector wider
        
        with col_select:
            # Add county selector dropdown
            unique_locations = sorted(plot_data['location'].unique())
            selected_location = st.selectbox(
                "Select a county:",
                ["All Counties"] + unique_locations,
                index=0 if not st.session_state.selected_county else 
                      unique_locations.index(st.session_state.selected_county) + 1,
                key='county_selector'
            )
        
        with col_reset:
            # Add reset button aligned with the dropdown
            if st.button("Reset County", key="reset_county_main"):
                st.session_state.selected_county = None
                st.rerun()
        
        # Update session state based on dropdown selection
        if selected_location != "All Counties":
            if st.session_state.selected_county != selected_location:
                st.session_state.selected_county = selected_location
                st.rerun()
        elif st.session_state.selected_county is not None:
            st.session_state.selected_county = None
            st.rerun()

        # Show data preview
        st.write(f"### Data Details ({selected_year}){' - ' + st.session_state.selected_county if st.session_state.selected_county else ''}")
        
        # Select relevant columns for display
        display_columns = [
            'Accident Year', 
            'County of Injury',
            'state_abbr',
            'Age at Injury',
            'Gender',
            'Average Weekly Wage',
            'Number of Dependents',
            'WCIO Nature of Injury Description',
            'WCIO Cause of Injury Description',
            'WCIO Part Of Body Description'
        ]
        
        # Create a display dataframe with filtered data
        display_df = filtered_df[display_columns].copy()
        
        # Format the data for better display
        display_df['Gender'] = display_df['Gender'].map({1: 'Male', 0: 'Female'})
        
        # Rename columns for better readability
        display_df.columns = [
            'Accident Year',
            'County',
            'State',
            'Age',
            'Gender',
            'Weekly Wage',
            'Dependents',
            'Nature of Injury',
            'Cause of Injury',
            'Body Part'
        ]
        
        # Show the dataframe with pagination
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Add download button for the filtered data
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Filtered Data",
            csv,
            "filtered_data.csv",
            "text/csv",
            key='download-csv'
        )
        
        # Create row for pie charts
        if st.session_state.selected_county:
            st.write(f"### {st.session_state.selected_county} - Injury Distribution Analysis")
        else:
            st.write("### United States - Injury Distribution Analysis")
        pie_col1, pie_col2, pie_col3 = st.columns(3)
        
        with pie_col1:
            # Nature of Injury pie chart
            nature_counts = filtered_df['WCIO Nature of Injury Description'].value_counts().head(10)
            fig_nature = px.pie(
                values=nature_counts.values,
                names=nature_counts.index,
                title='Top 10 Nature of Injuries',
                hole=0.4,  # Makes it a donut chart
                color_discrete_sequence=px.colors.sequential.Reds
            )
            fig_nature.update_traces(textposition='inside', textinfo='percent+label')
            fig_nature.update_layout(
                showlegend=False,
                title_x=0.5,
                title_y=0.95
            )
            st.plotly_chart(fig_nature, use_container_width=True)
            
        with pie_col2:
            # Cause of Injury pie chart
            cause_counts = filtered_df['WCIO Cause of Injury Description'].value_counts().head(10)
            fig_cause = px.pie(
                values=cause_counts.values,
                names=cause_counts.index,
                title='Top 10 Causes of Injuries',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Reds
            )
            fig_cause.update_traces(textposition='inside', textinfo='percent+label')
            fig_cause.update_layout(
                showlegend=False,
                title_x=0.5,
                title_y=0.95
            )
            st.plotly_chart(fig_cause, use_container_width=True)
            
        with pie_col3:
            # Body Part pie chart
            body_counts = filtered_df['WCIO Part Of Body Description'].value_counts().head(10)
            fig_body = px.pie(
                values=body_counts.values,
                names=body_counts.index,
                title='Top 10 Body Parts Injured',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Reds
            )
            fig_body.update_traces(textposition='inside', textinfo='percent+label')
            fig_body.update_layout(
                showlegend=False,
                title_x=0.5,
                title_y=0.95
            )
            st.plotly_chart(fig_body, use_container_width=True)
            
        # Add a note about the data
        st.caption("Note: Charts show top 10 categories for each injury characteristic based on the current filters.")

    elif page == "Prediction":
        st.title("Workers Compensation Claim Prediction")

        # Load model and check if successful
        model = load_model()
        encoders = load_encoders()
        
        if model is None or encoders is None:
            st.error("""
                Unable to load required model files. Please ensure all necessary files exist in the following structure:
                - models/rf_optimized.pkl
                - models/scalers_encoders/one_hot_encoder.pkl
                - models/scalers_encoders/target_encoder.pkl
                - models/scalers_encoders/minmax_scaler.pkl
            """)
            return

        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        # Personal Information Column
        with col1:
            st.subheader("Personal Information")
            age = st.number_input("Age at Injury", min_value=16, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            birth_year = st.number_input("Birth Year", min_value=1920, max_value=2010, value=1990)
            avg_weekly_wage = st.number_input("Average Weekly Wage ($)", min_value=0, max_value=10000, value=1000)
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
            ime4_count = st.number_input("IME-4 Count", min_value=0, value=0)
            
        # Claim Details Column    
        with col2:
            st.subheader("Claim Details")
            accident_date = st.date_input("Accident Date", datetime.now())
            c2_date = st.date_input("C-2 Date", datetime.now())
            c3_date = st.date_input("C-3 Date", datetime.now())
            assembly_date = st.date_input("Assembly Date", datetime.now())
            first_hearing = st.selectbox("First Hearing Already Occurred", ["Yes", "No"])
            attorney = st.selectbox("Attorney/Representative", ["Yes", "No"])
            adr = st.selectbox("Alternative Dispute Resolution", ["Yes", "No"])
            covid_indicator = st.selectbox("COVID-19 Related", ["Yes", "No"])
            
        # Location Information Column
        with col3:
            st.subheader("Location Information")
            county = st.selectbox("County of Injury", sorted(df['County of Injury'].unique()))
            medical_region = st.selectbox("Medical Fee Region", sorted(df['Medical Fee Region'].unique()))
            zip_code = st.text_input("Zip Code", "12345")
            
        # Create two more columns for additional details
        col4, col5 = st.columns(2)
        
        # WCIO Information Column
        with col4:
            st.subheader("WCIO Information")
            nature_of_injury = st.selectbox(
                "Nature of Injury",
                sorted(df['WCIO Nature of Injury Description'].unique())
            )
            cause_of_injury = st.selectbox(
                "Cause of Injury",
                sorted(df['WCIO Cause of Injury Description'].unique())
            )
            body_part = st.selectbox(
                "Part of Body",
                sorted(df['WCIO Part Of Body Description'].unique())
            )
            
        # Industry Information Column
        with col5:
            st.subheader("Industry Information")
            industry_code = st.selectbox(
                "Industry Code Description",
                sorted(df['Industry Code Description'].unique())
            )
            carrier_type = st.selectbox(
                "Carrier Type",
                sorted(df['Carrier Type'].unique())
            )
            carrier_name = st.selectbox(
                "Carrier Name",
                sorted(df['Carrier Name'].unique())
            )
            district = st.selectbox(
                "District",
                sorted(df['District Name'].unique())
            )
        
        # Add a predict button
        if st.button("Predict"):
            try:
                # Create initial DataFrame with all inputs
                input_data = pd.DataFrame({
                    'Age at Injury': [age],
                    'Gender': [1 if gender == "Male" else 0],
                    'Birth Year': [birth_year],
                    'Average Weekly Wage': [avg_weekly_wage],
                    'Number of Dependents': [dependents],
                    'IME-4 Count': [ime4_count],
                    'Accident Date': [accident_date.strftime('%Y-%m-%d')],
                    'C-2 Date': [c2_date.strftime('%Y-%m-%d')],
                    'C-3 Date': [c3_date.strftime('%Y-%m-%d')],
                    'Assembly Date': [assembly_date.strftime('%Y-%m-%d')],
                    'First Hearing Date': [1 if first_hearing == "Yes" else 0],
                    'Attorney/Representative': [1 if attorney == "Yes" else 0],
                    'Alternative Dispute Resolution': [1 if adr == "Yes" else 0],
                    'COVID-19 Indicator': [1 if covid_indicator == "Yes" else 0],
                    'County of Injury': [county],
                    'Medical Fee Region': [medical_region],
                    'Zip Code': [zip_code],
                    'WCIO Nature of Injury Code': [mappings['nature'][nature_of_injury]],
                    'WCIO Cause of Injury Code': [mappings['cause'][cause_of_injury]],
                    'WCIO Part Of Body Code': [mappings['body'][body_part]],
                    'Industry Code': [mappings['industry'][industry_code]],
                    'Carrier Type': [carrier_type],
                    'Carrier Name': [carrier_name],
                    'District Name': [district]
                })

                # # Show debug information
                # debug = pd.DataFrame(input_data).head()
                # st.write(debug)

                # Apply feature engineering
                input_data = engineer_features(input_data)

                # Show debug information after feature engineering
                # debug = pd.DataFrame(input_data).head()
                # st.write(debug)

                # Apply encoding and scaling
                input_data_scaled = encode_and_scale(input_data, encoders)

                # debug = pd.DataFrame(input_data).head()
                # st.write(debug)
                
                # Select features needed by model
                final_input = select_model_features(input_data_scaled)

                # debug = pd.DataFrame(input_data).head()
                # st.write("final input" + debug)
                
                # Make prediction
                prediction = model.predict(final_input)
                
                # Map prediction...
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                
            # Add information about the prediction
            with st.expander("About this prediction"):
                st.write("""
                This prediction tool uses a machine learning model trained on historical workers compensation claims data.
                The model takes into account various factors including personal information, claim details, and injury
                classification to make its prediction.
                
                Please note that this is a prediction tool only and should not be used as the sole basis for decision making.
                Always consult with appropriate professionals for official guidance.
                """)

if __name__ == "__main__":
    main()