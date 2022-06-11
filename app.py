import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title('Havo suv va yer transportlari klassifikatsiyasi')

file = st.file_uploader('Rasmni yuklash', 
                 type = ['jpg', 
                         'gif', 
                         'jpeg', 
                         'svg'])

if file:
    st.image(file)
    img = PILImage.create(file)
    
    model = load_learner('transport_model.pkl')
    prediction, prob_id, probs = model.predict(img)
    st.success(f'Bashorat: {prediction}')
    st.info(f'Ehtimollik: {probs[prob_id]*100:.1f} %')
    
    fig = px.bar(x = probs*100, y = model.dls.vocab)
    st.plotly_chart(fig)