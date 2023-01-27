import streamlit as st
import pathlib
import plotly.express as px
import platform
from fastai.vision.all import *

plt = platform.system()
if plt == 'Linux' : pathlib.WindowsPath = pathlib.PosixPath

# title
st.title("A model for classifying technologies (phones, watches and weapons).")

# rasmni joylash
file = st.file_uploader('Upload image', type=(['png','jpg','jpeg','gif','svg','webp']))

if file is not None:
    img = Image.open(file)
    st.image(img, caption = 'Uploaded image')
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
else:
    print("Iltimos, modelga  mos rasm yuklang")
