import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
import csv
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

def redirect(url):
    st.markdown(f"""
        <meta http-equiv="refresh" content="0; url={url}" />
    """, unsafe_allow_html=True)

# Function to run the ML model and save bounding boxes
def run_model(date_str):
    try:
        # Load the data from the CSV file
        data = pd.read_csv(f'people_motion_{date_str}.csv')

        # Group the data by person ID
        grouped_data = data.groupby('Person ID')

        # Dictionary to store the results: Person ID and their most frequent area's bounding box
        person_most_frequent_area = {}

        def get_bounding_box(points):
            """Calculate the bounding box from the given points."""
            min_x = np.min(points[:, 0])
            max_x = np.max(points[:, 0])
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])
            return min_x, min_y, max_x, max_y

        # Iterate over each group (person ID)
        for person_id, group in grouped_data:
            # Extract the coordinates (X, Y) for each person
            coordinates = group[['X', 'Y']].values

            # Use KMeans to cluster the data (we'll use 1 cluster to find the centroid)
            kmeans = KMeans(n_clusters=1, random_state=42)
            kmeans.fit(coordinates)

            # The cluster center is the most frequent region (centroid)
            cluster_center = kmeans.cluster_centers_[0]

            # Get the coordinates of the points in the main cluster
            labels = kmeans.labels_
            cluster_points = coordinates[labels == 0]

            # Determine the bounding box for the main cluster
            min_x, min_y, max_x, max_y = get_bounding_box(cluster_points)

            # Store the person's ID and their most frequent area's bounding box coordinates
            person_most_frequent_area[person_id] = (int(min_x), int(min_y), int(max_x), int(max_y))

        # Save the prediction to a new CSV file
        output_file = f"person_most_frequent_area_{date_str}.csv"
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the headers
            writer.writerow(['Person ID', 'Min X', 'Min Y', 'Max X', 'Max Y'])

            # Write each person's most frequent area's bounding box coordinates
            for person_id, (min_x, min_y, max_x, max_y) in person_most_frequent_area.items():
                writer.writerow([person_id, min_x, min_y, max_x, max_y])

        st.success(f"Model executed and results saved to {output_file}")
    
    except FileNotFoundError:
        st.error(f"File 'people_motion_{date_str}.csv' not found. Please provide the correct date.")

# Function to plot bounding boxes with unique colors
def plot_bounding_boxes(df):
    date_str = st.session_state.date.strftime("%d-%m-%Y")

    # Create a blank image to plot the bounding boxes
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set the axis limits
    ax.set_xlim(0, 1000)  # Adjust these limits according to your needs
    ax.set_ylim(0, 1000)  # Adjust these limits according to your needs
    ax.invert_yaxis()  # Invert y axis to match the image coordinate system
    
    # Generate unique colors for each person ID
    unique_ids = df['Person ID'].unique()
    num_ids = len(unique_ids)
    
    # Use a color map to generate unique colors
    colormap = plt.get_cmap('tab20')  # Use 'tab20' colormap for up to 20 unique colors
    if num_ids > 20:
        # If more than 20 unique IDs, use a continuous color map
        colormap = plt.get_cmap('hsv')
    colors = colormap(np.linspace(0, 1, num_ids))
    color_map = dict(zip(unique_ids, colors))
    
    for index, row in df.iterrows():
        # Extract coordinates
        min_x, min_y, max_x, max_y = row['Min X'], row['Min Y'], row['Max X'], row['Max Y']
        person_id = row['Person ID']
        
        # Get color for this person ID
        color = color_map[person_id]
        
        # Create a Rectangle patch
        rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, linewidth=2, edgecolor=color, facecolor='none')
        
        # Add the patch to the plot
        ax.add_patch(rect)
        
        # Label the rectangle with the person ID
        plt.text(min_x, min_y, f'ID: {person_id}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor=color))
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Most Frequent Areas for Each Person')
    
    return fig

# Function to plot time series data
def plot_time_series(df):
    # Create a Plotly figure
    fig = go.Figure()
    
    # Generate a color map for unique person IDs
    unique_ids = df['Person ID'].unique()
    num_ids = len(unique_ids)
    color_map = px.colors.qualitative.Plotly  # Using Plotly's qualitative color map
    if num_ids > len(color_map):
        color_map = px.colors.qualitative.Set1  # Use a different color map if more colors are needed
    
    # Plot each person's time series with combined X and Y data
    for idx, person_id in enumerate(unique_ids):
        person_color = color_map[idx % len(color_map)]  # Cycle through the color map
        
        person_data = df[df['Person ID'] == person_id]
        
        fig.add_trace(go.Scatter(
            x=person_data['Time (seconds)'],
            y=person_data['X'],
            mode='lines+markers',
            name=f'Person ID {person_id} (X, Y)',
            line=dict(color=person_color),  # Use unique color for X data
            marker=dict(size=6)  # Customize marker size if needed
        ))
        fig.add_trace(go.Scatter(
            x=person_data['Time (seconds)'],
            y=person_data['Y'],
            mode='lines+markers',
            line=dict(color=person_color, dash='dash'),  # Use same color but dashed line for Y data
            marker=dict(size=6),  # Customize marker size if needed
            showlegend=False  # Hide the legend for this trace
        ))

    # Update layout
    fig.update_layout(
        title='Time Series Analysis of People Motion',
        xaxis_title='Time (seconds)',
        yaxis_title='Coordinate Value',
        legend_title='Legend',
        template='plotly_white'
    )
    
    return fig

# Streamlit app
st.title('Person Motion Analysis')

# Date picker for the user to select the date
date = st.date_input("Select Date", datetime.now())

# Store selected date in session state
st.session_state.date = date

# Button to show plot for bounding boxes
if st.button('Show Plot for Bounding Boxes'):
    date_str = st.session_state.date.strftime("%d-%m-%Y")

    # Run the ML model and save the results
    run_model(date_str)
    
    # Load the data from the CSV file
    try:
        df = pd.read_csv(f'person_most_frequent_area_{date_str}.csv')
        
        # Ensure correct column names
        if 'Person ID' in df.columns and 'Min X' in df.columns and 'Min Y' in df.columns and 'Max X' in df.columns and 'Max Y' in df.columns:
            # Plot bounding boxes
            fig = plot_bounding_boxes(df)
            
            # Display the plot in Streamlit
            st.pyplot(fig)
        else:
            st.error("CSV file must contain 'Person ID', 'Min X', 'Min Y', 'Max X', and 'Max Y' columns.")
    except FileNotFoundError:
        st.error(f"File 'person_most_frequent_area_{date_str}.csv' not found. Please ensure the date is correct and the file exists.")

# Button to show time series plot
if st.button('Show Time Series Plot'):
    date_str = st.session_state.date.strftime("%d-%m-%Y")

    # Load the data from the time series CSV file
    try:
        df = pd.read_csv(f'people_motion_{date_str}.csv')
        
        # Ensure correct column names
        if 'Person ID' in df.columns and 'Time (seconds)' in df.columns and 'X' in df.columns and 'Y' in df.columns:
            # Plot time series data
            fig = plot_time_series(df)
            
            # Display the plot in Streamlit
            st.plotly_chart(fig)
        else:
            st.error("CSV file must contain 'Person ID', 'Time (seconds)', 'X', and 'Y' columns.")
    except FileNotFoundError:
        st.error(f"File 'people_motion_{date_str}.csv' not found. Please ensure the date is correct and the file exists.")

if st.button('Exit'):
    redirect("http://localhost:8502")
