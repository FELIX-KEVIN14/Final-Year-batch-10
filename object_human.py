import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
def redirect(url):
    st.markdown(f"""
        <meta http-equiv="refresh" content="0; url={url}" />
    """, unsafe_allow_html=True)
# Function to load and filter the CSV file based on the format
def load_and_filter_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None

    # Filter out columns where all values are zero
    filtered_data = data.loc[:, (data != 0).any(axis=0)]
    
    return filtered_data

# Streamlit app
st.title("Object Counts Graph")

# Create a sidebar with date picker and buttons
st.sidebar.header("Options")

# Date input using the calendar to select the CSV file
selected_date = st.sidebar.date_input("Select Date", datetime.today())
formatted_date = selected_date.strftime("%d-%m-%Y")
formatted_date_for_human_count = selected_date.strftime("%Y-%m-%d")

# Buttons to trigger data loading and plotting
if st.sidebar.button("Resource Count"):
    file_path = f"object_counts_{formatted_date}.csv"
    df = load_and_filter_data(file_path)

    if df is not None:
        st.write("Filtered Data (Resource Count):")
        st.dataframe(df)

        # Select the columns to plot
        columns_to_plot = df.columns[1:]  # Assuming the first column is time or similar index

        if len(columns_to_plot) > 0:
            # Plot the filtered columns using Plotly
            fig = px.line(df, x=df.columns[0], y=columns_to_plot, title="Resource Counts Over Time")
            st.plotly_chart(fig)
        else:
            st.warning("No columns with non-zero values found to plot.")

if st.sidebar.button("Human Count"):
    file_path = f"object_counts_{formatted_date_for_human_count}.csv"
    df = load_and_filter_data(file_path)

    if df is not None:
        st.write("Filtered Data (Human Count):")
        st.dataframe(df)

        # Select the columns to plot
        columns_to_plot = df.columns[1:]  # Assuming the first column is time or similar index

        if len(columns_to_plot) > 0:
            # Plot the filtered columns using Plotly
            fig = px.line(df, x=df.columns[0], y=columns_to_plot, title="Human Counts Over Time")
            st.plotly_chart(fig)
        else:
            st.warning("No columns with non-zero values found to plot.")

# Add Exit button to sidebar
if st.sidebar.button("Exit"):
    redirect("http://localhost:8502")
    
    # Clear the Streamlit app interface by stopping execution
    st.stop()
