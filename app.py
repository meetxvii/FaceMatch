import streamlit as st
from elasticsearch import Elasticsearch
from deepface import DeepFace
import os
import shutil
from zipfile import ZipFile

@st.cache_resource
def load_es():
    return Elasticsearch(hosts=[{"host": "localhost", "port": 9200,'scheme':'http'}])

def search_image_page():
    

    if os.path.exists("temp/files.txt"):
        show_results()
        return
    st.title("Search Image")

    cols = st.columns(3)
    try:
        if len(os.listdir('temp')) == 0:
            st.write("No Images Uploaded")
            return
    except:
        st.write("No Images Uploaded")
        return
    for idx,file in enumerate(os.listdir("temp")):
        if "txt" in file:
            continue
        with cols[idx%3]:
            st.image(os.path.join("temp",file), width=200)
            if st.button("Seach", key=file):
                embeddings = DeepFace.represent(os.path.join("temp",file), model_name='Facenet', enforce_detection=False)
                body = {
                    "min_score": 1.5,
                    "query": {
                        "script_score": {
                            "query": {
                                "match_all": {}
                            },
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {
                                    "query_vector": embeddings[0]['embedding']
                                }
                            }
                        }
                    }
                }
                res = es.search(index='facematch',body=body)
                
                with open("temp/files.txt", "w") as f:
                    for hit in res['hits']['hits']:
                        f.write(hit['_source']['image']+"\n")
                st.experimental_rerun()
                        

    if st.button("Delete:wastebasket:", key="Delete"):
        clear_progress = st.progress(0)
        for cl_prog,file in enumerate(os.listdir("temp")):
            es.delete_by_query(index='facematch',body={"query": {"match": {"image": file}}})
            clear_progress.progress((cl_prog+1)/len(os.listdir("temp")))
        shutil.rmtree("temp")
        st.experimental_rerun()

def show_results():
                        
    st.title("Results")
    
    with open("temp/files.txt", "r") as f:
        files = f.readlines()
    cols = st.columns(3)
    with ZipFile('temp/files.zip', 'w') as zipObj:
        
        for idx,file in enumerate(files):
            with cols[idx%3]:
                st.image(os.path.join("temp",file.strip()), width=200)
                zipObj.write(os.path.join("temp",file.strip()))
    
    with open("temp/files.zip", "rb") as fp:
        st.download_button(
            label="Download ZIP",
            data=fp,
            file_name="myfile.zip",
            mime="application/zip"
        )
    

    if st.button("Go Back:back:"):
        try:
            os.remove("temp/files.txt")
            os.remove("temp/files.zip")
        except:
            pass
        st.experimental_rerun()


    

def upload_image(files):
    progress_bar = st.progress(0)

    os.makedirs("temp", exist_ok=True)
    for idx,file in enumerate(files):
        if os.path.exists(os.path.join("temp",file.name)):
            progress_bar.progress((idx+1)/len(files))
            continue
        shutil.copyfileobj(file, open(os.path.join("temp",file.name), "wb"))
        embeddings = DeepFace.represent(os.path.join("temp",file.name), model_name='Facenet', enforce_detection=False)

        for embedding in embeddings:
            body = {
                "embedding": embedding['embedding'],
                "image": file.name
            }
            es.index(index='facematch',body=body)
        progress_bar.progress((idx+1)/len(files))
    st.balloons()

def upload_image_page():
    st.title("Upload Images")
    files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    if files:
        if st.button("Upload:arrow_up:"):
            upload_image(files)
    
if __name__ == "__main__":

    es = load_es()

    PAGE = st.sidebar.selectbox("Select Page", ["Home", "Upload Images", "Search Image"])
    st_page = st.empty()
    if PAGE == "Home":
        st.title("Face Recognition")
        
        st.image("banner.jpeg", width=400)
   
        st.markdown("FaceSort is perfect for event photographers, security personnel, and anyone who needs to organize large collections of images. By using this system, you can easily identify and group images based on individual faces, which can save you a lot of time and effort.")
        
    if PAGE == "Upload Images":
        upload_image_page()

    if PAGE == "Search Image":
        search_image_page()
