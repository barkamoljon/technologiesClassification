import streamlit as st
import pathlib
import plotly.express as px
import platform
from fastai.vision.all import *
import streamlit as st

# Everything is accessible via the st.secrets dict:

st.write("DB username:", st.secrets["db_username"])
st.write("DB password:", st.secrets["db_password"])
st.write("My cool secrets:", st.secrets["my_cool_secrets"]["things_i_like"])

# And the root-level secrets are also accessible as environment variables:

import os

st.write(
    "Has environment variables been set:",
    os.environ["db_username"] == st.secrets["db_username"],
)

plt = platform.system()
if plt == 'Linux' : pathlib.WindowsPath = pathlib.PosixPath

# title
st.title("Texnologiyalarni(telefonlar, soatlar va qurol-aslahalarni) klassifikatsiya qiluvchi model")

# rasmni joylash
file = st.file_uploader('Rasm yuklash', type=(['png','jpg','jpeg','gif','svg','webp']))

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
else:
    print("Iltimos, modelga  mos rasm yuklang")
