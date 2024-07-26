# import streamlit as st
# import pandas as pd
# from pandasai import SmartDataframe
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# import os


# # Loading environment variables from .env file
# load_dotenv() 

# # Function to chat with CSV data
# def chat_with_csv(df,query):
#     # Loading environment variables from .env file
#     load_dotenv() 

#     # Function to initialize conversation chain with GROQ language model
#     groq_api_key = os.environ['GROQ_API_KEY']

#     # Initializing GROQ chat with provided API key, model name, and settings
#     llm = ChatGroq(
#     groq_api_key=groq_api_key, model_name="llama3-70b-8192",
#     temperature=0.2)
#     # Initialize SmartDataframe with DataFrame and LLM configuration
#     pandas_ai = SmartDataframe(df, config={"llm": llm})
#     # Chat with the DataFrame using the provided query
#     result = pandas_ai.chat(query)
#     return result

# # Set layout configuration for the Streamlit page
# st.set_page_config(layout='wide')
# # Set title for the Streamlit application
# st.title("PDF Chat Hackathon 🧑‍💻")

# # Upload multiple CSV files
# input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

# # Check if CSV files are uploaded
# if input_csvs:
#     # Select a CSV file from the uploaded files using a dropdown menu
#     selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
#     selected_index = [file.name for file in input_csvs].index(selected_file)

#     #load and display the selected csv file 
#     st.info("CSV uploaded successfully")
#     data = pd.read_csv(input_csvs[selected_index])
#     st.dataframe(data.head(3),use_container_width=True)

#     #Enter the query for analysis
#     st.info("Chat Below")
#     input_text = st.text_area("Enter the query")

#     #Perform analysis
#     if input_text:
#         if st.button("Chat with csv"):
#             st.info("Your Query: "+ input_text)
#             result = chat_with_csv(data,input_text)
#             st.success(result)


import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import torch

# Loading environment variables from .env file
load_dotenv()

# Load the CSV file and precompute embeddings
def load_data_and_embeddings():
    data = pd.read_csv(r'docs\aMF857DsyF.pdf')  # Replace 'data.csv' with the path to your CSV file
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    embeddings = model.encode(data.astype(str).values.tolist(), convert_to_tensor=True, show_progress_bar=True)
    return data, embeddings, model

# Function to find the most relevant rows based on the query
def find_relevant_rows(query, data, embeddings, model, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k].cpu().numpy()
    return data.iloc[top_results]

# Function to chat with CSV data
def chat_with_csv(df, query, embeddings, model):
    relevant_rows = find_relevant_rows(query, df, embeddings, model)
    
    # Prepare the context for the LLM based on the relevant rows
    context = relevant_rows.to_string(index=False)
    
    # Load environment variables
    groq_api_key = os.environ['GROQ_API_KEY']
    
    # Initialize GROQ chat with provided API key, model name, and settings
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", temperature=0.2)
    
    # Chat with the context
    result = llm(context + "\n\nQ: " + query + "\nA:")
    return result

# Load data and embeddings once when the script runs
data, embeddings, model = load_data_and_embeddings()

# Set layout configuration for the Streamlit page
st.set_page_config(layout='wide')
# Set title for the Streamlit application
st.title("CSV Chat Hackathon 🧑‍💻")

# Display the CSV file
st.info("CSV loaded successfully")
st.dataframe(data.head(3), use_container_width=True)

# Enter the query for analysis
st.info("Chat Below")
input_text = st.text_area("Enter the query")

# Perform analysis
if input_text:
    if st.button("Chat with CSV"):
        st.info("Your Query: " + input_text)
        result = chat_with_csv(data, input_text, embeddings, model)
        st.success(result)
