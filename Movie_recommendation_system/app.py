import os
import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import base64

#setup
BASE_DIR = os.path.dirname(__file__)
st.set_page_config(page_icon=os.path.join(BASE_DIR,"assets","clapperboard.png"),page_title="Movie Recommendation System",layout="wide")
st.sidebar.markdown("<h2 style= color:#FF2056;>Movie Recommendation System</h2>",unsafe_allow_html=True)
st.sidebar.image(os.path.join(BASE_DIR,"assets","clapperboard.png"),width=200)
st.sidebar.text("Never Run Out of Movies")

#background image
def set_bg_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-color: rgba(255,255,255,0.1);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_local(os.path.join(BASE_DIR,"assets","semi-transparent.jpg"))

#load API's
load_dotenv()
API_KEY =os.getenv("API_KEY")
BASE_URL =os.getenv("BASE_URL")

if not API_KEY or not BASE_URL:
    st.error("API_KEY or BASE_URL not found. Check your .env file.")
    st.stop()



#load data
movies = pd.read_csv(os.path.join(BASE_DIR,"Datasets","movies_cleaned.csv"))

#vectorising the text using tfidf
vectorizer = TfidfVectorizer(stop_words="english",max_features=5000)
vectors = vectorizer.fit_transform(movies["tags"]).toarray()

#find cosine similarity
cosine_sim = cosine_similarity(vectors)

@st.cache_data(show_spinner=False)
def get_poster(movieid):
    url = f"{BASE_URL}/{movieid}"
    params = {"api_key": API_KEY}
    try:
        response = requests.get(url=url,params=params,timeout=5)
        if response.status_code!=200:
            return "icons/empty.png"
        
        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
        else:
            return os.path.join(BASE_DIR,"assets","empty.png")
        
    except Exception as e:
        return os.path.join(BASE_DIR,"assets","empty.png")
   
  
    
def recommend_movie(title):
    index = movies[movies['title']==title].index[0]
    sorted_list = sorted(enumerate(cosine_sim[index]),key=lambda x:x[1],reverse=True)
    recommended_movies = []
    for idx in sorted_list[1:11]:
        movieid = movies.iloc[idx[0]]["movie_id"]
        title = movies.iloc[idx[0]]["title"]
        with st.spinner("Loading.."):
            url = get_poster(movieid)
  
        recommended_movies.append([url,title])
    return recommended_movies

col1,col2 = st.columns([8,1])
title = col1.selectbox("which movie do you like? ",options=movies["title"].unique())
col2.write(" ")
col2.text(" ")
if col2.button("Search"):
    if title:
        recommended_movies = recommend_movie(title)
        c1,c2,c3,c4,c5 = st.columns(5,border=True,gap="small")
        b1,b2,b3,b4,b5 = st.columns(5,border=True)
        cols = [c1,c2,c3,c4,c5,b1,b2,b3,b4,b5]
        for idx,movie in enumerate(recommended_movies):
            with cols[idx]:
                st.image(movie[0],use_container_width=True)
                st.write(movie[1])
    