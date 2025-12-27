import os
import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from datetime import date
import base64

#setup
BASE_DIR = os.path.dirname(__file__)
st.set_page_config(page_icon=os.path.join(BASE_DIR,"assets","clapperboard.png"),page_title="Movie Recommendation System",layout="wide")
st.sidebar.markdown("<h2 style= color:#FF2056;>Movie Recommendation System</h2>",unsafe_allow_html=True)
st.sidebar.image(os.path.join(BASE_DIR,"assets","clapperboard.png"),width=200)
st.sidebar.text("Never Run Out of Movies")


#CONSTANTS
language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Odia": "or",
    "Assamese": "as",
    "Urdu": "ur",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Russian": "ru",
    "Portuguese": "pt",
    "Turkish": "tr",
    "Arabic": "ar"
}


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
   
def popular_movies(n=10):
    df = movies.sort_values(by="popularity",ascending=False).head(n)
    popular_movies = []
    for movieid,title in zip(df["movie_id"],df["title"]):
        with st.spinner("Loading.."):
            url = get_poster(movieid)
        popular_movies.append([url,title,movieid])
    
    return popular_movies
    
def recommend_movie(title):
    index = movies[movies['title']==title].index[0]
    sorted_list = sorted(enumerate(cosine_sim[index]),key=lambda x:x[1],reverse=True)
    recommended_movies = []
    for idx in sorted_list[1:11]:
        movieid = movies.iloc[idx[0]]["movie_id"]
        title = movies.iloc[idx[0]]["title"]
        with st.spinner("Loading.."):
            url = get_poster(movieid)
        recommended_movies.append([url,title,movieid])
    return recommended_movies

@st.dialog(" ",width="medium")
def disp_overview(movieid=False,poster="",latest_movie=False):
    c1,c2 = st.columns([1,2])
    with c1:
        st.header(" ")
        st.image(poster)
    if movieid:
        with c2:
            df = movies[movies["movie_id"]==movieid].reset_index()
            st.write("## ",df["title"][0])
            st.write("**Overview:** ")
            st.caption(" ".join(eval(df["overview"][0])))
            st.write("**Genre:** ")
            st.caption(", ".join(eval(df["genres"][0])),)
            st.write("**Director:** ")
            st.caption(", ".join(eval(df["crew"][0])))
            st.write("**Cast:** ")
            st.caption(", ".join(eval(df["cast"][0])))
    else:
        with c2:
            st.write("## ",latest_movie[0])
            st.write("**Overview:** ")
            st.caption(latest_movie[1])

            col1,col2=st.columns([1,2])
            with col1:
                st.write("**Release date:** ")
                st.caption(latest_movie[2])
                st.write("**Rating:** ")
                st.caption(f"{latest_movie[3]:.2f} /10")
            with col2:
                st.write("**Genre:** ")
                st.caption(f"{latest_movie[4]["genres"]}")
                st.write("**Director:** ")
                st.caption(f"{latest_movie[4]["director"]}")
                st.write("**Cast:** ")
                st.caption(f"{latest_movie[4]["cast"]}")

def display_posters(state,button_name):
    c1,c2,c3,c4,c5 = st.columns(5,border=True,gap="small")
    b1,b2,b3,b4,b5 = st.columns(5,border=True)
    cols = [c1,c2,c3,c4,c5,b1,b2,b3,b4,b5]
    for idx,movie in enumerate(state):
        with cols[idx]:
            st.image(movie[0],use_container_width=True)
            if st.button(movie[1],type="tertiary",key=f"{button_name}_{idx}"):
                st.session_state.poster = movie[0]
                st.session_state.movieid = movie[2]
                st.session_state.show_dialog = True
                st.rerun()

@st.cache_data(show_spinner=False)
def fetch_movies(url,params):
    try:
        response = requests.get(url,params=params,timeout=10)
        if response.status_code==200:
            return response.json().get("results",[])
        return []
    except:
        return []
    
def latest_movies(language):
    url = "https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key":API_KEY,
        "language":"en-US",
        "region":"IN",
        "with_original_language":language_codes[language],
        "sort_by":"primary_release_date.desc",
        "release_date.lte":date.today().isoformat,
        "vote_count.gte":10, #filters unreleased movies
        "page":1
    }
    movies = fetch_movies(url,params)
    return movies

@st.cache_data(show_spinner=False)
def latest_movie_additional(movie_id):
    try: 
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {
            "api_key":API_KEY,
             "append_to_response": "credits"
             }
        response = requests.get(url,params=params,timeout=10)
        if response.status_code==200:
            data = response.json()
            genres = ", ".join([gen["name"] for gen in data["genres"]])
            cast = ", ".join([i["name"] for i in data["credits"]["cast"][:3]])
            director = ", ".join([i["name"] for i in data["credits"]["crew"] if i["job"]=="Director"])
            return {"genres":genres,"cast":cast,"director":director}
        
        return {}
    except:
        return {}

if "movieid" not in st.session_state:
    st.session_state.movieid=False
if "poster" not in st.session_state:
    st.session_state.poster=False

if "recommended_movies" not in st.session_state:
    st.session_state.recommended_movies = None

if "popular_movies" not in st.session_state:
    st.session_state.popular_movies = None

tab1,tab2,tab3 = st.tabs(tabs=["Popular","Recommend","Latest Movies"])
with tab1:
    #popular movies
    st.subheader("Popular movies")
    st.session_state.popular_movies = popular_movies()
    display_posters(st.session_state.popular_movies,"pop_movies")

with tab2:
    col1,col2 = st.columns([8,1])
    title = col1.selectbox("which movie do you like? ",options=movies["title"].unique())
    col2.write(" ")
    col2.text(" ")

    if col2.button("Search"):
        st.session_state.recommended_movies = recommend_movie(title)
        
    if st.session_state.recommended_movies:
        display_posters(st.session_state.recommended_movies,"rec_movies")

with tab3:
    c1,c2 = st.columns([3,1])
    c1.subheader("Latest Movies")
    language = c2.selectbox("Language: ",options=language_codes)
    latest = latest_movies(language)
    for i in range(0,len(latest),5):
        cols = st.columns(5,border=True)
        for col,data in zip(cols,latest[i:i+5]):
            details = latest_movie_additional(data["id"])
            with col:
                title = data["title"]
                overview = data["overview"]
                release_date = data["release_date"]
                rating = data["vote_average"]
                poster_path = "https://image.tmdb.org/t/p/w500"+data["poster_path"]
                st.image(poster_path)
                if st.button(title,type="tertiary"):
                    disp_overview(poster=poster_path,latest_movie=[title,overview,release_date,rating,details])
if st.session_state.movieid:
    disp_overview(st.session_state.movieid,st.session_state.poster)
    st.session_state.show_dialog = False
    st.session_state.movieid = False
    