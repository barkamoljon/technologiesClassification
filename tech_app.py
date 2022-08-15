import streamlit as st
import pathlib
pip install plotly
import plotly.express as px
matplotlib==1.5.1
import platform
from fastai.vision.all import *

plt = platform.system()
if plt == 'Linux' : pathlib.WindowsPath = pathlib.PosixPath

# title
st.title("Texnologiyalarni(telefonlar, soatlar va qurol-aslahalarni) klassifikatsiya qiluvchi model")

# rasmni joylash
file = st.file_uploader('Rasm yuklash', type=(['png','jpg','jpeg','gif','svg']))

if file is not None:
    img = Image.open(file)
    st.image(img, caption = 'Yuklangan rasm')
    if file:
      #st.image(file)
      #PIL convert
      img = PILImage.create(file)
      #model
      model = load_learner('technologies_model.pkl')

      # prediction
      pred, pred_id, probs = model.predict(img)
      st.success(f'Prognoz:{pred}')
      st.info(f'Ehtimollik:{probs[pred_id]*100: .1f}%')

      #plotting
      fig = px.bar(x=probs*100, y=model.dls.vocab)
      st.plotly_chart(fig)

