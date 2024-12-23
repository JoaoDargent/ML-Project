# Workers Compensation Claims Analysis

This project provides an interactive web application for analyzing workers compensation claims data and predicting claim outcomes.

## Setup & Installation

1. Make sure you have Python 3.8 or newer installed.

2. Clone or download this repository to your local machine

3. Open a terminal/command prompt and navigate to the project's webapp directory:
   ```bash
   cd ML-Project/webapp
   ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

The application should automatically open in your default web browser. If it doesn't, you can manually open the URL shown in the terminal (typically http://localhost:8501).

## Features

The application has two main sections:

1. **Exploratory Data Analysis (EDA)**
   - Interactive choropleth map of workers compensation claims
   - Filtering by year, gender, and county
   - Distribution visualizations of injury types, causes, and affected body parts

2. **Prediction**
   - Input form for predicting claim outcomes
   - Uses machine learning model trained on historical data

## Project Structure