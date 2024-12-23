import streamlit as st
import pandas as pd
import plotly.express as px
import json

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/merged_with_fips.csv", index_col=0)
    df['fips'] = df['fips'].apply(lambda x: f"{int(float(x)):05d}" if pd.notnull(x) else None)
    # Convert Accident Date to datetime and extract year
    df['Accident Date'] = pd.to_datetime(df['Accident Date'])
    df['Year'] = df['Accident Date'].dt.year
    return df

# Load GeoJSON
@st.cache_data
def load_geojson():
    with open("data/counties.json") as response:
        counties = json.load(response)
    return counties

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

def main():
    st.set_page_config(page_title="Final Project App", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["EDA", "Prediction"])

    df = load_data()
    counties = load_geojson()

    if page == "EDA":
        st.title("Exploratory Data Analysis")
        
        # Create columns for side-by-side selection boxes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Create list of features for visualization with display names
            numerical_features = [
                "Accident Count",
                "Age at Injury",
                "Average Weekly Wage",
                "Number of Dependents"
            ]
            
            # Feature selection
            selected_feature = st.selectbox(
                "Select feature to visualize:",
                numerical_features
            )
        
        with col2:
            # Year selection
            available_years = ["All Years"] + sorted(df['Year'].unique().tolist())
            selected_year = st.selectbox(
                "Select year:",
                available_years
            )

        with col3:
            # Gender selection
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

        # Aggregate data based on selected feature and filters
        plot_data = aggregate_by_fips(filtered_df, selected_feature)
        
        # Set color label
        if selected_feature == "Accident Count":
            color_label = "Number of Accidents"
        else:
            color_label = f"Average {selected_feature}"
            
        # Add year and gender to title if specific selections made
        if selected_year != "All Years":
            color_label += f" ({selected_year})"
        if selected_gender != "All":
            color_label += f" - {selected_gender}"

        # Create choropleth map
        fig_map = px.choropleth_mapbox(
            plot_data,
            geojson=counties,
            locations="fips",
            color="accident_count" if selected_feature == "Accident Count" else selected_feature,
            color_continuous_scale=["#fee0d2", "#fc9272", "#de2d26", "#67000d"],
            range_color=(plot_data["accident_count" if selected_feature == "Accident Count" else selected_feature].min(), 
                        plot_data["accident_count" if selected_feature == "Accident Count" else selected_feature].max()),
            mapbox_style="carto-positron",
            zoom=3,
            center={"lat": 37.0902, "lon": -95.7129},
            opacity=0.5,
            labels={"accident_count" if selected_feature == "Accident Count" else selected_feature: color_label,
                   "location": "Location"},
            hover_data={"fips": False,  # Hide FIPS in hover
                       "location": True,  # Show location name
                       "accident_count" if selected_feature == "Accident Count" else selected_feature: True}
        )
        fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig_map, use_container_width=True)

        # Show data preview
        st.write(f"### Data Details ({selected_year})")
        
        # Select relevant columns for display
        display_columns = [
            'Accident Date', 
            'County of Injury',
            'state_abbr',
            'Age at Injury',
            'Gender',
            'Average Weekly Wage',
            'Number of Dependents',
            'WCIO Nature of Injury Description',
            'WCIO Cause of Injury Description'
        ]
        
        # Create a display dataframe with filtered data
        display_df = filtered_df[display_columns].copy()
        
        # Format the data for better display
        display_df['Accident Date'] = pd.to_datetime(display_df['Accident Date']).dt.strftime('%Y-%m-%d')
        display_df['Gender'] = display_df['Gender'].map({1: 'Male', 0: 'Female'})
        
        # Rename columns for better readability
        display_df.columns = [
            'Accident Date',
            'County',
            'State',
            'Age',
            'Gender',
            'Weekly Wage',
            'Dependents',
            'Nature of Injury',
            'Cause of Injury'
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

    elif page == "Prediction":
        st.title("Prediction Page")
        st.write("Prediction functionality here.")

if __name__ == "__main__":
    main()