# fully updated application as of (07-jan-2025) and deploying it into azure app services
import streamlit as st
import pandas as pd

import numpy as np
import json
import logging
import os
from streamlit_extras.stylable_container import stylable_container
import psycopg2
import logging
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
import base64 

import openai
from datetime import datetime,timedelta
import time,uuid

from azure.storage.blob import BlobServiceClient
import base64
from io import BytesIO
# ==================================to set baground pic for login page implemented=======

logging.basicConfig(level=logging.INFO)

DB_NAME = 'metaldb'
DB_USER = 'ajuservpostgresql'
DB_PASSWORD = 'Project1'
DB_HOST = 'aj-flexible-server-postgre.postgres.database.azure.com' 
DB_PORT = '5432'

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'active_page' not in st.session_state:
    st.session_state['active_page'] = 'Home'
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None

# Connect to the PostgreSQL database

def get_db_info():
    cnxn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        sslmode='require'  # Ensure SSL mode is set to require
    )
    return cnxn

def read_credentials():
    cnxn = get_db_info()
    cursor = cnxn.cursor()
    cursor.execute("SELECT userid, password, id FROM credentials")
    credentials = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
    cursor.close()
    cnxn.close()
    return credentials



# =====================save signup credentials===================

def save_credentials(userid, password):
    cnxn = get_db_info()
    cursor = cnxn.cursor()
    id = uuid.uuid4()  
    cursor.execute("INSERT INTO credentials (id,userid, password) VALUES (%s,%s, %s)", (str(id),userid, password))
    cnxn.commit()
    cursor.close()
    cnxn.close()
 # =================
def set_background_image(image_file):
        with open(image_file, "rb") as image:
            encoded_string = base64.b64encode(image.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
                background-attachment: no-scroll;
                background-position: center;
                width: 100%;
                hieght: 100vh;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

st.markdown("""
        <style>
           body {
                 background-image: url('https://i.pinimg.com/originals/6b/5d/4e/6b5d4e0fbd7aad3c5110222e3d32d642.jpg'); 
                 background-size: cover; 
                 background-repeat: no-repeat; 
                 background-position: center; 
                 height: 100vh; 
                 margin: 0; 
            }
                
            .login-title {
                text-align: center;
                font-size: 60px;
                font-weight: bold;
                color: darkblue;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
            }
            .login-title1 {
                text-align: center;
                font-size: 28px;
                font-weight: bold;
                color: darkblue;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
            }
            .login-title span {
                color: hsl(122, 39.40%, 49.20%);
            }
            .login-subtitle {
                text-align: left;
                font-size: 20px;
                color: #000000;
                margin-top: -15px;
            }
            .login-button {
                display: block;
                width: 100%;
                padding: 10px 15px;
                background-color:hsl(122, 39.40%, 49.20%);
                color: blue;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                text-align: center;
                cursor: pointer;
                margin-top: 15px;
                transition: background-color 0.3s ease;
            }
            .left-column {
                display: flex;
                flex-direction: column;
                justify-content: center;
                padding: 50px;
            }   
            .login-button:hover {
                background-color: #45a049;
            }
        </style>
        """, unsafe_allow_html=True)     

# ========================signup page===================

def signup_page():
    set_background_image('static/sign-up.jpeg')
    with stylable_container(
            key="main_container", 
            css_styles="""
                {
                    
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 10px;
                    padding: 40px;
                    left: 45%; /* Moves the container towards the right */
                    
                    background-color: rgba(200, 200, 200, 0.8);
                }
            """
        ):
            # Layout using columns
        col1, col2 = st.columns([2, 1.5])  # Adjust the ratio as needed

            # Left Section: MetalIntel Branding
        # Vertically center the content in col1
        with col1:
            # st.image('logo.png')
            st.markdown("<div class='left-column'>", unsafe_allow_html=True)
            st.image("static/BHELLOGO.png", use_container_width=False, width=350)
            st.markdown("</div>", unsafe_allow_html=True)

            # Right Section: Login Form
            # Right column for login form
            
        with col2:
                    
            user_inputs_font_css = """
            <style>


            div[class*="stTextInput"] label p {
            font-size: 18px;
            color: darkblue;
            }
            div[class*="stCheckbox"] label p {
            font-size: 18px;
            color: darkblue;
            }
            </style>
            """

            st.write(user_inputs_font_css, unsafe_allow_html=True)

            st.markdown("<h3 class='login-title1'>User Sign Up</h3>", unsafe_allow_html=True)
            new_userid = st.text_input("UserID", key="new_userid", placeholder="Enter UserID")
            new_password = st.text_input("Password", type="password", key="new_password", placeholder="Enter Password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password", placeholder="Confirm Password")
            
           
            col1, col2 = st.columns([1, 2])

            with col1:
                if st.button("Signup"):
                    if not new_userid.strip():
                        st.error("UserID is required!")
                    elif not new_password.strip():
                        st.error("Password is required!")
                    elif not confirm_password.strip():
                        st.error("Confirm Password is required!")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match!")
                    else:
                        credentials = read_credentials()
                        if new_userid in credentials:
                            st.error("UserID already exists! Please choose another.")
                        else:
                            save_credentials(new_userid, new_password)
                            st.success("Signup successful! Redirecting to login page...")
                            st.session_state['active_page'] = 'Login'
                            st.rerun()

            with col2:
                st.markdown("<p>already have account</p>", unsafe_allow_html=True)
                if st.button("Back to Login"):
                    st.session_state['active_page'] = 'Login'
                    st.rerun()
                                

    
   

# ===================================================================================        
def login_page():  
    # Set the background image for the app (replace with your image file path)
    set_background_image('static/bhel_login_page.jpg')

    with stylable_container(
            key="main_container", 
            css_styles="""
                {
                    
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 10px;
                    padding: 40px;
                    left: 45%; /* Moves the container towards the right */
                    
                    background-color: rgba(200, 200, 200, 0.5);
                }
            """
        ):
            # Layout using columns
        col1, col2 = st.columns([2, 1.5])  # Adjust the ratio as needed

            # Left Section: MetalIntel Branding
        # Vertically center the content in col1
        with col1:
            # st.image('logo.png')
            st.markdown("<div class='left-column'>", unsafe_allow_html=True)
            st.image("static/BHELLOGO.png", use_container_width=False, width=350)
            st.markdown("</div>", unsafe_allow_html=True)

            # Right Section: Login Form
            # Right column for login form
        with col2:
                    
            user_inputs_font_css = """
            <style>


            div[class*="stTextInput"] label p {
            font-size: 18px;
            color: darkblue;
            }
            div[class*="stCheckbox"] label p {
            font-size: 18px;
            color: darkblue;
            }
            </style>
            """

            st.write(user_inputs_font_css, unsafe_allow_html=True)

            st.markdown("<h3 class='login-title1'>User Login</h3>", unsafe_allow_html=True)
            # Add User ID and Password input fields
            userid = st.text_input("UserID", key="userid",placeholder="Enter UserID")
            password = st.text_input("Password", type="password", key="password",placeholder="Enter Password")
            # Add "Remember me" checkbox
            remember_me = st.checkbox("Remember me")

           
            if st.button("Login"):  # Replace the markdown button with a Streamlit button
                
                credentials = read_credentials()
                # Check if the entered user ID and password match
                if userid in credentials and credentials[userid][0] == password:
                    # Set session state variables
                    st.session_state['logged_in'] = True
                    st.session_state['user_id'] = credentials[userid][1]
                    st.session_state['active_page'] = 'Home'
                    st.success("Login successful!")
                    st.rerun()
                else:
                    # Display error only on incorrect login attempt
                    st.error("Incorrect User ID or Password")
                
            # Signup link
            if st.button("Sign Up here"):
                st.session_state['active_page'] = 'Signup'
                st.rerun()


# Only render the page that matches the active state
if st.session_state['active_page'] == 'Signup':
    signup_page()
    st.stop()  # Ensure no other content loads
elif st.session_state['active_page'] == 'Login':
    login_page()
    st.stop()  # Stop execution after loading login page

# ============================================bhel bot page=================================================

def Home_page():
    # Sidebar navigation
    with st.sidebar:
        selected_page = option_menu(
            "",
            options=[
                "Home", "BHEL Genie", "Tenders",
                "Logout"
            ],
            icons=["house", "robot", "book", "box-arrow-right"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#EFEBEB"},
                "icon": {"color": "blue", "font-size": "22px"},
                "nav-link": {
                    "font-size": "12px",
                    "text-align": "left",
                    "margin": "0",
                    "color": "black"
                },
                "nav-link-selected": {"background-color": "#DFDCDC", "color": "blue"},
                "nav-link:hover": {"background-color": "#e2e6ea", "color": "#007bff"}
            }
        )

    # BHEL Genie Chatbot Page
    if selected_page == "BHEL Genie":
        # st.set_page_config(page_title="BHEL Chat", layout="wide")
        st.image("static/BHELLOGO.png", width=200)

        def get_base64_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()

        assistant_logo_base64 = get_base64_image("static/bhelchaticon.jpg")

        st.markdown(
            """
            <style>
            .title {
                text-align: center; /* Center align the text */
                font-size: 3rem; /* Adjust font size */
                font-weight: bold; /* Make it bold */
                
                background: linear-gradient(45deg, #00ff87, #3d82c0, #00ff87, #3d82c0, #3d82c0);
                
                
                
                -webkit-background-clip: text; /* Clip gradient to text */
                -webkit-text-fill-color: transparent; /* Fill text with gradient */
                color: transparent; /* Ensure fallback color is transparent */
                margin-top: 20px;
                margin-bottom: 20px;
            }
            </style>
            <h1 class="title">Bharat Heavy Electricals Limited</h1>

            <style>
            /* Set the background color of the entire page */
            .stApp {
                background-color: #fcfcfc; /* Light grey background */
            }

            /* Chat container styling */
            .chat-container {
                padding: 10px;
            }

            /* Styling for chat messages */
            .message {
                display: flex;
                align-items: center;
                margin: 10px 0;
            }
            .message img {
                width: 30px;
            }
            .message-left {
                justify-content: flex-start;
            }
            .message-right {
                justify-content: flex-end;
            }

            /* User query styling: #3d82c0 */
            .message-right .message-text {
                background-color: #3d82c0; /* Blue shade for user query */
                color: #ffffff; /* White text for contrast */
                text-align: right;
                border-radius: 8px;
                padding: 10px;
                max-width: 80%;
                box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
            }

            /* Assistant reply styling: whiter shade */
            .message-left .message-text {
                background-color: #ffffff; /* Whiter shade */
                color: #333333; /* Dark grey for text */
                text-align: left;
                border-radius: 8px;
                padding: 10px;
                max-width: 80%;
                box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
            }
            </style>
            """, unsafe_allow_html=True )           
            

        # Initialize chat session
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for text, align in st.session_state.messages:
            if align == "left":
                st.markdown(
                    f"""<div class="message message-left">
                            <img src="data:image/png;base64,{assistant_logo_base64}">
                            <div class="message-text">{text}</div>
                        </div>""", unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div class="message message-right">
                            <div class="message-text">{text}</div>
                        </div>""", unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)
        # ================================================================
        import os
        import json
        import pickle
        import numpy as np
        import faiss
        import openai

        # Azure OpenAI Configuration
        openai.api_type = "azure"
        openai.api_base = "https://aj-rec-dev-si-ai.openai.azure.com/"
        openai.api_version ="2024-05-01-preview"
        openai.api_key = "78QQiYXPCJ5elvTWkWD3f3td484PVVBGIHq4JuvP1CcCekKapQnAJQQJ99BCAC77bzfXJ3w3AAABACOGnqSZ"

        # FAISS and metadata file paths
        FAISS_INDEX_PATH = "test_faiss_index.bin"
        CHUNKS_FILE_PATH = "test_chunks.pkl"
        EMBEDDINGS_FILE_PATH = "test_embeddings.pkl"
        METADATA_FILE_PATH = "test_metadata.json"


            


        # Check if required files exist before loading
        if not all(os.path.exists(path) for path in [FAISS_INDEX_PATH, CHUNKS_FILE_PATH, METADATA_FILE_PATH]):
            raise FileNotFoundError("❌ Required FAISS index, chunks, or metadata files are missing!")

        # Load FAISS index and stored embeddings
        index = faiss.read_index(FAISS_INDEX_PATH)

        with open(CHUNKS_FILE_PATH, "rb") as f:
            chunks = pickle.load(f)

        with open(METADATA_FILE_PATH, "r") as f:
            metadata = json.load(f)  

        print(f"✅ Metadata loaded. Total records: {len(metadata)}")  # Debugging metadata size

        def get_embedding(text):
            """Generate an embedding for the given text using Azure OpenAI."""
            try:
                response = openai.Embedding.create(
                    input=text,
                    engine="text-embedding-3-large"
                )
                return np.array(response["data"][0]["embedding"]).astype("float32")
            except Exception as e:
                print(f"❌ Error generating embedding: {e}")
                return None

        def search_faiss(query, k=5, similarity_threshold=0.3):
            """Search the FAISS index for similar documents with a similarity threshold."""
            query_embedding = get_embedding(query)
            
            if query_embedding is None:
                return []

            query_embedding = query_embedding.reshape(1, -1)  # Convert to NumPy array
            D, I = index.search(query_embedding, k=k)  # Retrieve top k matches

            results = []
            for score, idx in zip(D[0], I[0]):
                if idx != -1 and idx < len(chunks) and score > similarity_threshold:  # Filter based on similarity
                    meta_entry = metadata[idx] if idx < len(metadata) else {"pdf_name": "Unknown", "url": "#"}
                    print(f"🔎 Found match: Index {idx}, Score: {score}, Metadata: {meta_entry}")  # Debugging FAISS search
                    results.append((chunks[idx], meta_entry))

            return results

        def truncate_context(context, max_tokens=3000):
            """Truncate context to fit within max token limit."""
            token_count = 0
            truncated_context = []
            
            for doc, _ in context:
                if not isinstance(doc, str):
                    print(f"⚠️ Skipping non-string document: {doc}")  # Debugging step
                    continue  # Skip non-string entries
                
                doc_tokens = doc.split()
                if token_count + len(doc_tokens) > max_tokens:
                    break
                truncated_context.append(doc)
                token_count += len(doc_tokens)
            
            return "\n".join(truncated_context)

        def generate_response(final_prompt):
            """Generate a response using Azure OpenAI."""
            try:
                response = openai.ChatCompletion.create(
                    engine="gpt-35-turbo",
                    messages=[
                        {"role": "system", "content": " You are an expert research assistant. Answer queries concisely using the provided document context.If unsure, state that you don't know. Be factual and clear. Use the provided context to answer queries accurately."},
                        {"role": "user", "content": final_prompt}
                    ],
                    max_tokens=1500
                )
                return response["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"❌ Error generating response: {e}")
                return "An error occurred while generating the response."
            




        def rewrite_query_with_history(chat_history, query):
            """Use Azure OpenAI to rewrite the user query by considering chat history."""
                
            # Azure OpenAI Configuration
            openai.api_type = "azure"
            openai.api_base = "https://aj-rec-dev-si-ai.openai.azure.com/"
            openai.api_version ="2024-05-01-preview"
            openai.api_key = "78QQiYXPCJ5elvTWkWD3f3td484PVVBGIHq4JuvP1CcCekKapQnAJQQJ99BCAC77bzfXJ3w3AAABACOGnqSZ"


            # Format chat history into a readable structure
            formatted_history = "\n".join(
                [f"User: {msg.get('user', '')}\nBot: {msg.get('bot', '')}" for msg in chat_history if isinstance(msg, dict)]
            ) if chat_history else "No prior conversation."

            prompt = f"""
            The user has been having a conversation with you. Rewrite the user's latest query so it makes sense given the conversation history.


            Chat History:
            {formatted_history}

            Current User Query: "{query}"

            Rewrite the query in a way that it is self-contained and clear:
            """

            try:
                response = openai.ChatCompletion.create(
                    engine="gpt-35-turbo",  # Adjust based on your available Azure OpenAI model
                    messages=[
                        {"role": "system", "content": "You are an AI specialized in query refinement."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                refined_query = response["choices"][0]["message"]["content"].strip()
                return refined_query

            except Exception as e:
                print(f"⚠️ Azure OpenAI Error: {e}")
                return query  # Fallback to original query if there's an error

        def handle_query(query, chat_history):
            """Handle user query by retrieving relevant data and generating a response, considering chat history."""

            # Rewrite the query using Azure OpenAI
            refined_query = rewrite_query_with_history(chat_history, query)

            # Retrieve relevant context
            context = search_faiss(refined_query, k=5, similarity_threshold=0.3)

            if not context:
                return "No relevant documents found.", chat_history

            # Truncate context (assume you have your own logic here)
            truncated_context = truncate_context(context)

            # Final prompt for answer generation
            final_prompt = f"""
            Context: {truncated_context}
            User Query: {refined_query}
            Answer:
            """

            response = generate_response(final_prompt)

            # 💡 Ask LLM to judge the quality of the response
            validation_prompt = f"""
            User Query: {refined_query}
            Response: {response}

            Does this response contain meaningful information that addresses the user's query? Reply with only "Yes" or "No".
            """

            judgment = generate_response(validation_prompt).strip().lower()

            if "no" in judgment:
                return "Response: No information found.\n\nReferences:\nNo references available", chat_history

            # Prepare references
            referenced_files = list({
                meta["pdf_name"]: f'<a href="{meta["url"]}" target="_blank">{meta["pdf_name"]}</a>'
                for _, meta in context
            }.values())

            numbered_references = "\n".join(
                [f"{i+1}. {file}" for i, file in enumerate(referenced_files)]
            )

            chat_history.append({"user": query, "bot": response})

            result = f"Response: {response}\n\nReferences:\n{numbered_references if referenced_files else 'No references available'}"

            return result, chat_history

        # ==================================================================

        def handle_enter_input():
            user_input = st.session_state.get("user_input", "").strip()
            if user_input:
                st.session_state.messages.append((user_input, "right"))
                response, st.session_state.chat_history = handle_query(user_input, st.session_state.get('chat_history', []))
                st.session_state.messages.append((response, "left"))
                st.session_state["chat_history"].append({"user": user_input, "response": response})
                st.session_state['user_input'] = ""
        st.markdown(
            """
            <style>
            textarea {
                width: 100% !important;
                min-height: 68px !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        user_input = st.text_area("Type your Query...", key="user_input", height=68, on_change=handle_enter_input)
        st.markdown("<div style='height: 300px;'></div>", unsafe_allow_html=True)  # Spacer
        st.markdown("""
            <div class="footer" style="position:relative; width:100%; text-align:center; padding:15px;
                                        background-color:#000; color:#fff; font-size:14px; border-top:1px solid #444;">
                <p>
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Accessibility Statement</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Copyright Policy</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Disclaimer</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Feedback</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Site Map</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Hyperlink Policy</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Privacy Policy</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Terms & Conditions</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Terms of Use</a> |
                    <a href="#" style="color:#fff; text-decoration:none;">Help</a>
                </p>
                
                <p>Copyright &copy; 2025 - All Rights Reserved - Ajuserv Technologies Private Limited</p>
                

                <p>Note: Content on this website is published and managed by Ajuserv Technologies Private Limited.</p>
                <p>For any query regarding this website, please contact the web information manager at</p>
                <p>Email ID: contactus at ajuserv.com</p>

                <p>Maintained by Ajuserv</p>
            </div>
        """, unsafe_allow_html=True)

    

# ====================================
    if selected_page == 'Home':
        st.markdown("<h1 style='text-align: center; color: darkblue;'>Welcome to BHEL Portal</h1>", unsafe_allow_html=True)
        
        # Display an image banner
        # st.image("static/home_banner.jpg", use_container_width=True)

        # Add sections for navigation
        st.markdown("""
            <style>
                .home-section {
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 10px;
                    padding: 20px;
                    background-color: rgba(200, 200, 200, 0.5);
                    margin-top: 20px;
                    text-align: center;
                }
                .home-section h2 {
                    color: darkblue;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 300px;'></div>", unsafe_allow_html=True)  # Spacer
        st.markdown("""
            <div class="footer" style="position:relative; width:100%; text-align:center; padding:15px;
                                        background-color:#000; color:#fff; font-size:14px; border-top:1px solid #444;">
                <p>
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Accessibility Statement</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Copyright Policy</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Disclaimer</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Feedback</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Site Map</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Hyperlink Policy</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Privacy Policy</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Terms & Conditions</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Terms of Use</a> |
                    <a href="#" style="color:#fff; text-decoration:none;">Help</a>
                </p>
                
                <p>Copyright &copy; 2025 - All Rights Reserved - Ajuserv Technologies Private Limited</p>
                

                <p>Note: Content on this website is published and managed by Ajuserv Technologies Private Limited.</p>
                <p>For any query regarding this website, please contact the web information manager at</p>
                <p>Email ID: contactus at ajuserv.com</p>

                <p>Maintained by Ajuserv</p>
            </div>
        """, unsafe_allow_html=True)



# ======================================================================================
    if selected_page == 'Logout':
        st.session_state['logged_in'] = False
        st.session_state['user_id'] = None
        st.session_state['active_page'] = 'none'
        st.success("Logout successful!")
        st.rerun()
# =========================================




# ====================================================================================
    # Other Pages
    elif selected_page in ["Tenders"]:
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            st.image('static/ajuserv_logo.png', width=155)
        with col2:
            st.markdown(f'<h1 style="text-align:center; color:gray; font-size:28px;">{selected_page}</h1>', unsafe_allow_html=True)
        with col3:
            st.image('static/BHELLOGO.png', width=155)

        
        if selected_page == "Tenders":
            st.write("Content for Tenders page...")         
       
        

        st.markdown("<div style='height: 300px;'></div>", unsafe_allow_html=True)  # Spacer
        st.markdown("""
            <div class="footer" style="position:relative; width:100%; text-align:center; padding:15px;
                                        background-color:#000; color:#fff; font-size:14px; border-top:1px solid #444;">
                <p>
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Accessibility Statement</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Copyright Policy</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Disclaimer</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Feedback</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Site Map</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Hyperlink Policy</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Privacy Policy</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Terms & Conditions</a> |
                    <a href="#" style="color:#fff; text-decoration:none; margin-right:10px;">Terms of Use</a> |
                    <a href="#" style="color:#fff; text-decoration:none;">Help</a>
                </p>
                
                <p>Copyright &copy; 2025 - All Rights Reserved - Ajuserv Technologies Private Limited</p>
                

                <p>Note: Content on this website is published and managed by Ajuserv Technologies Private Limited.</p>
                <p>For any query regarding this website, please contact the web information manager at</p>
                <p>Email ID: contactus at ajuserv.com</p>

                <p>Maintained by Ajuserv</p>
            </div>
        """, unsafe_allow_html=True)
# =========================================

# Ensure 'logged_in' is initialized in session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    Home_page()
else:
    login_page()  # Define this function separately
