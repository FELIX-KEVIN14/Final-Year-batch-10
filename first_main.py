import streamlit as st
import pandas as pd
import os
import streamlit_construction

def redirect(url):
    st.markdown(f"""
        <meta http-equiv="refresh" content="0; url={url}" />
    """, unsafe_allow_html=True)


# File to store user login details
csv_file = 'login_details.csv'

# Function to load the CSV file with login details
def load_login_data():
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    else:
        return pd.DataFrame(columns=['Username', 'Password'])

# Function to save new user details in CSV
def save_user(username, password):
    df = load_login_data()
    new_user = pd.DataFrame({'Username': [username], 'Password': [password]})
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(csv_file, index=False)

# Function to check if the user exists and password matches
def authenticate_user(username, password):
    df = load_login_data()
    user = df[(df['Username'] == username) & (df['Password'] == password)]
    return not user.empty

# Streamlit app layout
st.title("Login / Register App")
st.image("OptiBuild.png", caption="OptiBuild", width=500) 
placeholder = st.empty()
# Menu selection: Login or Register
menu = st.selectbox("Choose Login or Register", ["Login", "Register"])

if menu == "Login":
    st.subheader("Login")
    login_username = st.text_input("Username")
    login_password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate_user(login_username, login_password):
            st.success(f"Welcome {login_username}!")
            st.balloons()  # Balloons for successful login
            # Code to redirect to next page or perform an action after successful login
            st.write("You are now logged in!")
            redirect("http://localhost:8502")

            
        else:
            st.error("Incorrect username or password")

elif menu == "Register":
    st.subheader("Register")
    register_username = st.text_input("Choose a Username")
    register_password = st.text_input("Choose a Password", type="password")

    if st.button("Register"):
        df = load_login_data()
        if register_username in df['Username'].values:
            st.warning("Username already exists! Please choose another.")
        else:
            save_user(register_username, register_password)
            st.success("You have successfully registered! Now you can login.")
            st.balloons()  # Balloons for successful registration

# Ensure the file exists when the program runs
if not os.path.exists(csv_file):
    load_login_data().to_csv(csv_file, index=False)
