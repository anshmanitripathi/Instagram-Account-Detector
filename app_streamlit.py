import streamlit as st
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- Load Trained Model and Encoder ---
# Make sure these files are in the same directory as your script
try:
    model = joblib.load('instagram_account_detector.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
except FileNotFoundError:
    st.error("Model or encoder not found! Please run the training notebook first to generate the .joblib files.")
    st.stop() # Stop the app from running further

# --- Page Configuration ---
st.set_page_config(
    page_title="Instagram Account Detector",
    page_icon="",
    layout="centered"
)

# --- UI Elements ---
st.title("Instagram Account Detector")
st.write(
    "This app uses a **Random Forest model** to predict whether an Instagram account "
    "is likely to be **Real, a Bot, a Scam, or Spam** based on its profile information."
)
st.markdown("---")


# --- User Input Form ---
with st.form("prediction_form"):
    st.header("Enter Account Details")

    # Create two columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        followers = st.number_input("Followers", min_value=0, value=0, step=100)
        following = st.number_input("Following", min_value=0, value=0, step=50)
        posts = st.number_input("Posts", min_value=0, value=0, step=10)
        mutual_friends = st.number_input("Mutual Friends", min_value=0, value=0, step=1)

    with col2:
        bio = st.selectbox("Has Bio?", ("select","yes", "n"))
        profile_picture = st.selectbox("Has Profile Picture?", ("select","yes", "n"))
        external_link = st.selectbox("Has External Link?", ("select", "n", "yes"))
        threads = st.selectbox("Active on Threads?", ("select","n", "yes"))

    # Submit button for the form
    submit_button = st.form_submit_button(label="Analyze Account")


# --- Prediction Logic ---
if submit_button:
    # 1. Create a dictionary from user input
    user_input = {
        'Followers': followers,
        'Following': following,
        'Posts': posts,
        'Mutual Friends': mutual_friends,
        'Bio': bio,
        'Profile Picture': profile_picture,
        'External Link': external_link,
        'Threads': threads
    }

    # 2. Preprocess the input
    input_df = pd.DataFrame([user_input])
    
    # Calculate ratio columns (add a small epsilon to avoid division by zero)
    epsilon = 1e-9
    input_df['Following/Followers'] = input_df['Following'] / (input_df['Followers'] + epsilon)
    input_df['Posts/Followers'] = input_df['Posts'] / (input_df['Followers'] + epsilon)

    # Map binary columns from 'yes'/'n' to 1/0
    binary_cols = ['Bio', 'Profile Picture', 'External Link', 'Threads']
    for col in binary_cols:
        input_df[col] = input_df[col].map({'n': 0, 'yes': 1})

    # Ensure the column order is the same as the training data
    expected_cols = model.named_steps['preprocessor'].transformers_[0][2]
    input_df = input_df[expected_cols]

    # 3. Make a prediction
    with st.spinner("Analyzing..."):
        numeric_prediction = model.predict(input_df)
        final_prediction = label_encoder.inverse_transform(numeric_prediction)[0]

    # 4. Display the result
    st.markdown("---")
    st.header("Prediction Result")
    
    if final_prediction == 'Real':
        st.success(f"✅ The account is predicted to be: **{final_prediction}**")
        st.info("This account shows patterns consistent with genuine human behavior.")
    elif final_prediction == 'Scam':
        st.error(f"🚨 The account is predicted to be a: **{final_prediction}**")
        st.warning("This account exhibits traits commonly associated with fraudulent activities.")
    else: # Bot or Spam
        st.warning(f"⚠️ The account is predicted to be a: **{final_prediction}**")
        st.info("This account shows signs of automation or spam-like activity.")

