from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px

# Initialize the app
app = Dash(__name__)

# Load dataset
csv_file = '/Users/joaodargent/Library/CloudStorage/OneDrive-NOVAIMS/IMS/Machine Learning/Project/Project2/train_data.csv'
data = pd.read_csv(csv_file)

# App layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # For tracking the current page
    html.Div([
        html.H1("Interactive ML Dashboard", style={'text-align': 'center'}),
        html.Div([
            dcc.Link('EDA', href='/eda', style={'margin-right': '20px'}),
            dcc.Link('Predict an Observation', href='/predict'),
        ], style={'text-align': 'center', 'margin-bottom': '30px'})
    ], className="header"),

    # Content will be rendered here based on the URL
    html.Div(id='page-content')
])


# EDA Page Layout
def eda_page_layout():
    num_observations = len(data)
    target_counts = data['Claim Injury Type'].value_counts()

    return html.Div([
        html.H2("Exploratory Data Analysis", style={'text-align': 'center'}),
        html.P(f"Number of Observations: {num_observations}"),
        html.Div([
            html.H3("Counts for each Claim Injury Type:"),
            html.Ul([html.Li(f"{target}: {count}") for target, count in target_counts.items()])
        ])
    ])


# Prediction Page Layout
def prediction_page_layout():
    return html.Div([
        html.H2("Observation Prediction", style={'text-align': 'center'}),
        html.Div([
            html.H3("Enter Observation Details:"),
            html.Div([dcc.Input(id=f'predict-{col}', type='text', placeholder=f'Enter {col}') for col in data.columns[:-1]]),
            html.Button('Predict!', id='predict-button', style={'margin-top': '20px'}),
            html.Div(id='prediction-output', style={'margin-top': '20px'})
        ], style={'margin': '0 auto', 'width': '50%'})
    ])


# Callback to render the correct page
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/predict':
        return prediction_page_layout()
    else:  # Default to EDA page
        return eda_page_layout()


# Callback for prediction logic
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State(f'predict-{col}', 'value') for col in data.columns[:-1]]
)
def make_prediction(n_clicks, *input_values):
    if n_clicks:
        # Dummy prediction logic (replace with actual model prediction)
        input_data = {col: val for col, val in zip(data.columns[:-1], input_values)}
        return f"Prediction: [Your model result here]"


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)