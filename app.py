import streamlit as st
 
st.title("👋 Hello User App")
 
# Get integer input from user

user_id = st.number_input("Enter your ID (as a number):", min_value=1, step=1)
 
# Display greeting

if user_id:

    st.success(f"Hello User #{int(user_id)}! 👋")

 