import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

def encode_and_scale(data, encoders):
    """
    Applies encoding and scaling steps from the notebook to a DataFrame
    """
    # Create a copy to avoid modifying the original
    df = data.copy()

    # Define feature groups
    numeric_features = [
        'Age at Injury', 'Average Weekly Wage', 'Birth Year',
        'Number of Dependents', 'Accident Year', 'C-2 Date Year',
        'Assembly Year', 'Dependency-to-Income Ratio', 'Time Between Events',
        'Carrier Accident Density', 'Industry Claim Percentage', 
        'Region Risk Percentage', 'Geo-Industry Risk'
    ]

    low_cardinality_features = [
        'Carrier Type', 'District Name', 'Medical Fee Region',
        'Age Group', 'Broad Body Part', 'Season of Accident',
        'Region Risk Level', 'Industry Risk Level'
    ]

    high_cardinality_features = [
        'Carrier Name','County of Injury', 'Industry Code',
        'WCIO Cause of Injury Code', 'WCIO Nature of Injury Code',
        'WCIO Part Of Body Code', 'Zip Code', 'Injury-Location Pair',
        'Injury Complexity'
    ]

    binary_features = [
        'IME-4 Count', 'Alternative Dispute Resolution',
        'Attorney/Representative', 'COVID-19 Indicator',
        'First Hearing Date', 'Gender', 'Frequent Injury Cause',
        'Accident on Weekday', 'Both C2 and C3', 'C-3 Date',
        'Only C2', 'Only C3', 'No C2 or C3'
    ]

    cyclic_features = ['Accident Month', 'Assembly Month', 'C-2 Date Month']

    # 1. HANDLE CYCLIC FEATURES
    for col in cyclic_features:
        if col in df.columns:
            # Convert months to sine and cosine components
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col]/12)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col]/12)
            df.drop(col, axis=1, inplace=True)

    # 2. ONE HOT ENCODE LOW CARDINALITY CATEGORICAL FEATURES
    ohc = encoders['onehot']
    low_card_cols = [col for col in low_cardinality_features if col in df.columns]
    if low_card_cols:
        ohc_columns = ohc.transform(df[low_card_cols])
        ohc_feature_names = ohc.get_feature_names_out(low_card_cols)
        df_encoded = pd.DataFrame(
            ohc_columns,
            columns=ohc_feature_names,
            index=df.index
        )
        # Remove original categorical columns and add encoded ones
        df = df.drop(low_card_cols, axis=1)
        df = pd.concat([df, df_encoded], axis=1)

    # 3. TARGET ENCODE HIGH CARDINALITY FEATURES
    target_encoder = encoders['target']
    high_card_cols = [col for col in high_cardinality_features if col in df.columns]
    
    if high_card_cols:
        # Create a copy of df without high cardinality columns
        remaining_cols = [col for col in df.columns if col not in high_card_cols]
        df_encoded = df[remaining_cols].copy()
        
        # Add each encoded column
        for col in high_card_cols:
            encoded_values = target_encoder.transform(df[[col]])
            df_encoded[f'{col}_encoded'] = encoded_values
        
        df = df_encoded

    # 4. SCALE ALL NUMERIC FEATURES
    # Get numeric columns that exist in the dataframe
    numeric_cols = [col for col in numeric_features if col in df.columns]
    
    if numeric_cols:
        minmax_scaler = encoders['minmax']
        scaled_features = minmax_scaler.transform(df[numeric_cols])
        df[numeric_cols] = scaled_features

    # 5. ENSURE BINARY FEATURES ARE 0/1
    binary_cols = [col for col in binary_features if col in df.columns]
    for col in binary_cols:
        df[col] = df[col].astype(int)

    return df

def select_model_features(df_scaled, model_features=None):
    """
    Selects and orders features required by the model
    """
    if model_features is None:
        model_features = [
            'Age at Injury', 'Average Weekly Wage', 'Birth Year',
            'Dependency-to-Income Ratio', 'Carrier Accident Density',
            'Injury-Location Pair_1. CANCELLED', 'Injury-Location Pair_2. NON-COMP',
            'Injury-Location Pair_3. MED ONLY', 'Injury-Location Pair_4. TEMPORARY',
            'Injury-Location Pair_5. PPD SCH LOSS', 'IME-4 Count',
            'Attorney/Representative', 'First Hearing Date', 'C-3 Date'
        ]

    # Create DataFrame with all required features, filling missing ones with 0
    final_input = pd.DataFrame(0, index=df_scaled.index, columns=model_features)
    
    # Fill in available features
    for col in model_features:
        if col in df_scaled.columns:
            final_input[col] = df_scaled[col]
    
    return final_input