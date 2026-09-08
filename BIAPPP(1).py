import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.stattools import adfuller  # Make sure to import adfuller


# Set the Streamlit page title and configuration
st.set_page_config(page_title='Insects vs Weather Analysis', page_icon='📊')

# Helper Functions
def load_and_clean_metro(files):
    dfs = []
    for file in files:
        df = pd.read_excel(file, skiprows=0)
        df.columns = df.columns.str.strip()
        expected_columns = {
            "Dati meteo storici": "DateTime",
            "Unnamed: 1": "Temperature",
            "Unnamed: 2": "LowTemperature",
            "Unnamed: 3": "HighTemperature",
            "Unnamed: 4": "Humidity"
        }
        df = df.rename(columns={col: expected_columns.get(col, col) for col in df.columns})
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['DateTime'])
        numeric_columns = ["Temperature", "LowTemperature", "HighTemperature", "Humidity"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].str.replace(',', '.').astype(float)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def load_and_clean_capture(files):
    dfs = []
    for file in files:
        df = pd.read_excel(file, skiprows=0)
        df.columns = df.columns.str.strip()
        df = df.rename(columns={
            "Grafico delle catture": "DateTime",
            "Unnamed: 1": "NumOfInstect",
            "Unnamed: 2": "NewCapture",
            "Unnamed: 3": "Reviewed",
            "Unnamed: 4": "Event"
        })
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['DateTime'])
        for col in ["NumOfInstect", "NewCapture"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Visualization Functions
def plot_captures(captures_per_day):
    captures_per_day['Date'] = pd.to_datetime(captures_per_day['Date']).dt.strftime('%Y-%m-%d')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=captures_per_day['Date'], y=captures_per_day['TotalInsects'], name='Total Insects', marker_color='blue'))
    fig.add_trace(go.Bar(x=captures_per_day['Date'], y=captures_per_day['NewCaptures'], name='New Captures', marker_color='orange'))
    fig.update_layout(
        title='Total Insects and New Captures per Day',
        xaxis_title='Date',
        yaxis_title='Count',
        barmode='group',
        xaxis=dict(tickangle=-45),
        template='plotly_white',
        legend=dict(title='Legend')
    )
    st.plotly_chart(fig)

def plot_metro_data_filtered(metro_data):
    # Convert DateTime column to pandas datetime format
    metro_data['DateTime'] = pd.to_datetime(metro_data['DateTime'])
    
    # Filter data for July to September
    metro_data_filtered = metro_data[metro_data['DateTime'].dt.month.isin([7, 8, 9])]

    # Plot Temperature Over Time
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=metro_data_filtered['DateTime'],
        y=metro_data_filtered['Temperature'],
        mode='lines+markers',
        name='Temperature',
        line=dict(color='red'),
        marker=dict(size=6)
    ))
    fig_temp.update_layout(
        title='Temperature Over Time (July to September)',
        xaxis_title='DateTime',
        yaxis_title='Temperature (°C)',
        template='plotly_white',
        xaxis=dict(tickangle=-45)
    )
    st.plotly_chart(fig_temp)

    # Plot Humidity Over Time
    fig_humidity = go.Figure()
    fig_humidity.add_trace(go.Scatter(
        x=metro_data_filtered['DateTime'],
        y=metro_data_filtered['Humidity'],
        mode='lines+markers',
        name='Humidity',
        line=dict(color='blue'),
        marker=dict(size=6)
    ))
    fig_humidity.update_layout(
        title='Humidity Over Time (July to September)',
        xaxis_title='DateTime',
        yaxis_title='Humidity (%)',
        template='plotly_white',
        xaxis=dict(tickangle=-45)
    )
    st.plotly_chart(fig_humidity)
def plot_average_temperature_humidity(metro_data):
    # Convert DateTime column to datetime type
    metro_data['DateTime'] = pd.to_datetime(metro_data['DateTime'])
    
    # Extract the Date part
    metro_data['Date'] = metro_data['DateTime'].dt.date
    
    # Filter data for the months of July to September
    metro_data['Month'] = metro_data['DateTime'].dt.month
    metro_data = metro_data[metro_data['Month'].between(7, 9)]
    
    # Group by Date and calculate mean Temperature and Humidity
    grouped_data = metro_data.groupby('Date', as_index=False).agg({'Temperature': 'mean', 'Humidity': 'mean'})
    
    # Reshape data for plotting
    long_data = grouped_data.melt(id_vars='Date', value_vars=['Temperature', 'Humidity'], 
                                  var_name='Metric', value_name='Value')
    
    # Create a bar plot
    fig = px.bar(long_data, x='Date', y='Value', color='Metric', barmode='group', 
                 title='Average Temperature and Humidity by Date (July to September)',
                 labels={'Date': 'Date', 'Value': 'Value', 'Metric': 'Measure'}, 
                 template='plotly_white')
    
    # Customize the layout
    fig.update_layout(xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
# Function to create the line graph
def plot_captured_insects_over_time(combined_data):
    # Ensure 'Date' is in datetime format
    combined_data['Date'] = pd.to_datetime(combined_data['Date'])

    # Create the interactive line graph
    fig = px.line(
        combined_data,
        x='Date',
        y='NewCaptures',
        title='Number of Captured Insects Over Time',
        labels={'Date': 'Date', 'NewCaptures': 'New Captures'},
        template='plotly_white'
    )

    # Customize the line and markers
    fig.update_traces(mode='lines+markers', line=dict(width=2, color='blue'), marker=dict(size=6))
    fig.update_layout(xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))

    # Display the plot in Streamlit
    st.plotly_chart(fig)
def plot_insects_over_time(combined_data):
    # Ensure 'Date' is in datetime format
    combined_data['Date'] = pd.to_datetime(combined_data['Date'])

    # Create the interactive line graph
    fig = px.line(
        combined_data,
        x='Date',
        y='TotalInsects',
        title='Number of Insects Over Time',
        labels={'Date': 'Date', 'TotalInsects': 'Insects'},
        template='plotly_white'
    )

    # Customize the line and markers
    fig.update_traces(mode='lines+markers', line=dict(width=2, color='blue'), marker=dict(size=6))
    fig.update_layout(xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))

    # Display the plot in Streamlit
    st.plotly_chart(fig)
# Function to plot the bar graph for temperature and humidity
# Function to plot the bar graph for temperature and humidity
def plot_temperature_humidity_bar(combined_data):
    # Create a bar plot for Temperature and Humidity
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting temperature and humidity
    ax.bar(combined_data['Date'], combined_data['Temperature'], width=0.4, label='Temperature', align='center')
    ax.bar(combined_data['Date'], combined_data['Humidity'], width=0.4, label='Humidity', align='edge')

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Temperature and Humidity Comparison')
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)


def plot_boxplots(combined_data):
    columns = ['TotalInsects', 'NewCaptures', 'Temperature', 'Humidity']
    long_df = combined_data.melt(id_vars=["Date"], value_vars=columns, var_name="Column", value_name="Value")
    fig = px.box(long_df, x='Column', y='Value', title="Boxplots of Selected Columns", labels={"Value": "Value", "Column": "Columns"}, points="all")
    fig.update_layout(xaxis_title="Columns", yaxis_title="Values", showlegend=False, plot_bgcolor="rgba(0,0,0,0)", title_font=dict(size=18, family="Verdana, sans-serif", color="darkblue"), xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False), yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False), margin=dict(l=50, r=50, t=50, b=50))
    st.plotly_chart(fig)


# Function to plot heatmap
def plot_heatmap(dataframe):
    # Dropping the 'Date' column as it's non-numeric
    df_numeric = dataframe.drop(columns=['Date'])

    # Calculate correlation matrix
    corr_matrix = df_numeric.corr()

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)

    # Set the title and labels
    plt.title('Correlation Heatmap')
    
    # Display the plot in Streamlit
    st.pyplot(plt)

# Clustering Function
def kmeans_clustering(combined_data, n_clusters=3):
    # Select relevant columns
    data_for_clustering = combined_data[['TotalInsects', 'NewCaptures', 'Temperature', 'Humidity']]
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_clustering)
    
    # Elbow Method to determine optimal number of clusters (WCSS)
    wcss = []  # Within-cluster sum of squares
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    # Plot the elbow curve
    st.subheader('Elbow Method for Optimal Number of Clusters')
    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    st.pyplot(fig)

    # Apply K-Means with the optimal number of clusters (use user input or default n_clusters=3)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    combined_data['Cluster'] = kmeans.fit_predict(scaled_data)

    # Display the results
    st.write("### Clustered Data Preview")
    st.dataframe(combined_data[['Date', 'TotalInsects', 'NewCaptures', 'Temperature', 'Humidity', 'Cluster']].head())

    # Create an interactive scatter plot with Plotly (Temperature vs. Humidity)
    st.subheader(f'Interactive Cluster Visualization (Temperature vs. Humidity) - {n_clusters} Clusters')
    fig = px.scatter(
        combined_data,
        x='Temperature',
        y='Humidity',
        color='Cluster',
        hover_data=['Date', 'TotalInsects', 'NewCaptures'],
        title=f'Cluster Visualization (Temperature vs. Humidity) - {n_clusters} Clusters',
        labels={'Cluster': 'Cluster ID'}
    )
    st.plotly_chart(fig)

    # 3D Scatter plot visualization (Temperature, Humidity, TotalInsects)
    st.subheader(f'3D Interactive Cluster Visualization - {n_clusters} Clusters')
    fig_3d = px.scatter_3d(
        combined_data,
        x='Temperature',
        y='Humidity',
        z='TotalInsects',
        color='Cluster',
        hover_data=['Date', 'NewCaptures'],
        title=f'3D Cluster Visualization - {n_clusters} Clusters',
        labels={'Cluster': 'Cluster ID'}
    )
    st.plotly_chart(fig_3d)
def linear_regression_model(combined_data):
    # Features and Targets
    X = combined_data[['Temperature', 'Humidity']]
    y_total_insects = combined_data['TotalInsects']
    y_new_captures = combined_data['NewCaptures']

    # Train-Test Split
    X_train, X_test, y_train_total, y_test_total = train_test_split(X, y_total_insects, test_size=0.2, random_state=42)
    _, _, y_train_new, y_test_new = train_test_split(X, y_new_captures, test_size=0.2, random_state=42)

    # Initialize Linear Regression Model
    model = LinearRegression()

    # Fit the model for TotalInsects
    model.fit(X_train, y_train_total)
    y_pred_total = model.predict(X_test)

    # Evaluate TotalInsects Model
    mse_total = mean_squared_error(y_test_total, y_pred_total)
    r2_total = r2_score(y_test_total, y_pred_total)

    st.subheader("Linear Regression for TotalInsects")
    st.write(f"Mean Squared Error: {mse_total:.2f}")
    st.write(f"R-squared: {r2_total:.2f}")

    # Scatter plot for Actual vs Predicted TotalInsects
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_total, y_pred_total, color='blue', alpha=0.6)
    plt.title("Actual vs Predicted: TotalInsects")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.plot([y_test_total.min(), y_test_total.max()], [y_test_total.min(), y_test_total.max()], color='red', linewidth=2)
    st.pyplot(plt)  # Show plot in Streamlit

    # Fit the model for NewCaptures
    model.fit(X_train, y_train_new)
    y_pred_new = model.predict(X_test)

    # Evaluate NewCaptures Model
    mse_new = mean_squared_error(y_test_new, y_pred_new)
    r2_new = r2_score(y_test_new, y_pred_new)

def arima_forecast_model(combined_data):
    # Select the target variable
    total_insects = combined_data['TotalInsects']

    # Check stationarity using ADF test
    adf_test = adfuller(total_insects)
    st.subheader("ADF Test Results")
    st.write(f"ADF Statistic: {adf_test[0]}")
    st.write(f"p-value: {adf_test[1]}")

    # If the p-value > 0.05, data is non-stationary, apply differencing
    if adf_test[1] > 0.05:
        st.warning("Data is non-stationary. Applying differencing...")
        total_insects_diff = total_insects.diff().dropna()
    else:
        total_insects_diff = total_insects
        st.info("Data is stationary. Proceeding without differencing.")

    # Fit ARIMA model (example: p=1, d=1, q=1)
    model = ARIMA(total_insects, order=(1, 1, 1))
    model_fit = model.fit()

    # Display ARIMA summary
    st.subheader("ARIMA Model Summary")
    st.text(model_fit.summary())

    # Forecast future values (e.g., 7 days)
    forecast_steps = 7
    forecast = model_fit.forecast(steps=forecast_steps)
    st.subheader(f"Forecasted Values for Next {forecast_steps} Days")
    st.write(forecast)

    # Plot original and forecasted data
    plt.figure(figsize=(10, 6))
    plt.plot(total_insects, label='Original Data', color='blue')
    plt.plot(range(len(total_insects), len(total_insects) + forecast_steps), forecast, label='Forecasted Data', linestyle='--', color='red')
    plt.legend()
    plt.title('ARIMA Forecast for TotalInsects')
    st.pyplot(plt)  # Display plot in Streamlit
    
    
    
    
# Streamlit UI
st.title("Environmental and Insect Capture Analysis")
st.sidebar.title("Navigation")
options = ["Load Data", "Visualizations", "Clustering", "Predictive Analysis", "ARIMA Forecast"]
choice = st.sidebar.radio("Choose an Option", options)

# File Uploads
files_metro = st.sidebar.file_uploader("Upload Metro Data", accept_multiple_files=True)
files_capture = st.sidebar.file_uploader("Upload Capture Data", accept_multiple_files=True)

if files_metro and files_capture:
    metro_data = load_and_clean_metro(files_metro)
    capture_data = load_and_clean_capture(files_capture)

    capture_data['Date'] = capture_data['DateTime'].dt.date
    captures_per_day = capture_data.groupby('Date').agg({'NumOfInstect': 'sum', 'NewCapture': 'sum'}).reset_index()
    captures_per_day.rename(columns={'NumOfInstect': 'TotalInsects', 'NewCapture': 'NewCaptures'}, inplace=True)
    average_metro_data = metro_data.groupby(metro_data['DateTime'].dt.date)[['Temperature', 'Humidity']].mean().reset_index()
    average_metro_data.rename(columns={'DateTime': 'Date'}, inplace=True)

    combined_data = pd.merge(captures_per_day, average_metro_data, on='Date', how='inner')

    if choice == "Load Data":
        st.write("### Metro Data Sample")
        st.dataframe(metro_data.head())
        st.write("### Capture Data Sample")
        st.dataframe(capture_data.head())
        st.write("### Combined Data Sample")
        st.dataframe(combined_data.head())

    elif choice == "Visualizations":
        plot_metro_data_filtered(metro_data)
        plot_average_temperature_humidity(metro_data)
        plot_captures(captures_per_day)
        plot_captured_insects_over_time(combined_data)
        plot_insects_over_time(combined_data)
        plot_temperature_humidity_bar(combined_data)
        plot_boxplots(combined_data)
        plot_heatmap(combined_data)

    elif choice == "Clustering":
        kmeans_clustering(combined_data, n_clusters=3)

    elif choice == "Predictive Analysis":
        linear_regression_model(combined_data)

    elif choice == "ARIMA Forecast":
        arima_forecast_model(combined_data)
else:
    st.warning("Please upload the required data files to proceed.")
