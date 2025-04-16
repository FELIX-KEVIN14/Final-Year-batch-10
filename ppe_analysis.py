import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
def redirect(url):
    st.markdown(f"""
        <meta http-equiv="refresh" content="0; url={url}" />
    """, unsafe_allow_html=True)

def load_data(file_path):
    """Load CSV data from the specified file path."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found. Please select a correct date.")
    
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()  # Return empty DataFrame if there's an error

def main():
    st.title("PPE Object Count Over Time")

    # Exit button
    if st.button("Exit"):
        redirect("http://localhost:8502") # Stops the Streamlit script from running further

    # Calendar date input
    selected_date = st.date_input("Select date")
    if selected_date:
        timestamp = selected_date.strftime('%d_%m_%Y')
        csv_filename = f'ppe_prediction_{timestamp}.csv'
        
        # Attempt to load data from the constructed file path
        try:
            df = load_data(csv_filename)
        except FileNotFoundError as e:
            st.error(f"File 'ppe_prediction_{timestamp}.csv'not found. Please ensure the date is correct and the file exists.")
            return
        
        # Check if the necessary columns are present
        if len(df.columns) < 3:
            st.error("CSV file must have at least 3 columns.")
            return

        # Ensure the necessary columns are present
        if 'Hour-Min-Sec' not in df.columns:
            st.error("CSV file must contain 'Hour-Min-Sec' column.")
            return

        # Prepare data for plotting
        x_column = 'Hour-Min-Sec'
        y_columns = [col for col in df.columns[2:] if col != 'Distant Class']

        # Dropdown menu for selecting a column to plot
        selected_column = st.selectbox("Select column to plot", y_columns)

        if selected_column:
            # Plot the selected column
            fig = px.line(df, x=x_column, y=selected_column, title=f'{selected_column} Over Time')
            fig.update_xaxes(title_text='Hour-Min-Sec')
            fig.update_yaxes(title_text=selected_column)
            
            # Display the plot
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
