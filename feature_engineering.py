import pandas as pd
import numpy as np
from datetime import datetime

def engineer_features(df):
    """
    Applies all feature engineering steps from the notebook to a DataFrame
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # 1. Process Dates
    date_columns = ['Accident Date', 'Assembly Date', 'C-2 Date']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])
            data[f'{col} Year'] = data[col].dt.year
            data[f'{col} Month'] = data[col].dt.month

    # 2. Create Age Groups
    data['Age Group'] = pd.cut(
        data['Age at Injury'],
        bins=[0, 25, 35, 45, 55, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '55+']
    )

    # 3. Create Frequent Injury Cause
    if 'WCIO Cause of Injury Code' in data.columns:
        data['Frequent Injury Cause'] = data['WCIO Cause of Injury Code'].map(
            lambda x: 'Frequent' if x in [97, 31, 48, 25, 59] else 'Not Frequent'
        )

    # 4. Create Broad Body Part Categories
    def categorize_body_part(code):
        code = float(code)
        if code in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 26]:
            return 'Head and Neck'
        elif code in [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]:
            return 'Upper Extremities'
        elif code in [50, 51, 52, 53, 54, 55, 56, 57, 58]:
            return 'Lower Extremities'
        elif code in [40, 41, 42, 43, 44, 45, 46, 61, 62]:
            return 'Trunk'
        elif code in [21, 23, 43, 48, 49, 60, 63]:
            return 'Internal Organs'
        elif code in [90, 91]:
            return 'Multiple Body Parts'
        elif code in [66, 99]:
            return 'Whole Body'
        else:
            return 'Other'
    
    if 'WCIO Part Of Body Code' in data.columns:
        data['Broad Body Part'] = data['WCIO Part Of Body Code'].apply(categorize_body_part)

    # 5. Create Dependency-to-Income Ratio
    data['Dependency-to-Income Ratio'] = data['Number of Dependents'] / (data['Average Weekly Wage'] + 1e-9)
    data['Dependency-to-Income Ratio'] = data['Dependency-to-Income Ratio'].fillna(0).replace([float('inf'), -float('inf')], 0)

    # 6. Create Injury-Location Pair
    if all(col in data.columns for col in ['WCIO Part Of Body Code', 'WCIO Nature of Injury Code']):
        data['Injury-Location Pair'] = (
            data['WCIO Part Of Body Code'].astype(str) + " - " + 
            data['WCIO Nature of Injury Code'].astype(str)
        )

    # 7. Calculate Time Between Events
    if all(col in data.columns for col in ['Accident Date', 'Assembly Date']):
        data['Time Between Events'] = (data['Assembly Date'] - data['Accident Date']).dt.days
        data['Time Between Events'] = data['Time Between Events'].fillna(0)

    # 8. Create Accident on Weekday Feature
    if 'Accident Date' in data.columns:
        data['Accident on Weekday'] = data['Accident Date'].dt.dayofweek < 5

    # 9. Create Injury Complexity Score
    if all(col in data.columns for col in ['WCIO Nature of Injury Code', 'Number of Dependents']):
        data['Injury Complexity'] = data['WCIO Nature of Injury Code'] * (data['Number of Dependents'] + 1)

    # 10. Calculate Carrier Accident Density
    if 'Carrier Name' in data.columns:
        carrier_counts = data['Carrier Name'].value_counts()
        data['Carrier Accident Density'] = data['Carrier Name'].map(carrier_counts) / len(data)

    # 11. Create Season of Accident
    if 'Accident Date' in data.columns:
        month = data['Accident Date'].dt.month
        data['Season of Accident'] = month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })

    # 12. Add Risk Level Features
    data['Region Risk Level'] = 'Medium Risk'  # Default value
    data['Industry Risk Level'] = 'Medium Risk'  # Default value

    # 13. Create C2/C3 Form Indicators
    data['Both C2 and C3'] = (data['C-2 Date'].notna() & data['C-3 Date'].notna()).astype(int)
    data['Only C2'] = (data['C-2 Date'].notna() & data['C-3 Date'].isna()).astype(int)
    data['Only C3'] = (data['C-2 Date'].isna() & data['C-3 Date'].notna()).astype(int)
    data['No C2 or C3'] = (data['C-2 Date'].isna() & data['C-3 Date'].isna()).astype(int)

    # 14. Create Geo-Industry Risk (if needed)
    if all(col in data.columns for col in ['County of Injury', 'Industry Code']):
        data['Geo-Industry Risk'] = data['County of Injury'].astype(str) + '-' + data['Industry Code'].astype(str)

    return data