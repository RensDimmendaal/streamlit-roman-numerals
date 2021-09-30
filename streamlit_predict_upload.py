import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import tempfile
import pandas as pd

from pathlib import Path
import altair as alt

from rens import load_img_32, create_model

ROMAN_NUMERALS = "i ii iii iv v vi vii viii ix x".upper().split()
REVERSED_NUMERALS = {r: i for i, r in enumerate(ROMAN_NUMERALS, start=1)}


@st.cache
def load_model(directory):
    return create_model(compile=True, trained_weights=f"{directory}/best_model")


model = load_model(".")


file = st.file_uploader("upload img")
# Do something interesting with the image data and paths
if file is not None:
    st.write(file)
    # # save
    # st.write(canvas_result.image_data / 255)
    image = Image.open(file)

    with tempfile.TemporaryDirectory() as td:
        fpath = Path(td) / "myfile.png"
        image.save(fpath)
        # _ = plt.imsave("delme.png", canvas_result.image_data / 255, cmap="gray")

        # # loadcorrect
        img = load_img_32(fpath)
        fig, ax = plt.subplots()
        ax.imshow(img)
        st.pyplot(fig)
        pred = model(img[np.newaxis, ...]).numpy().flatten()

        proba = np.exp(pred) / np.sum(np.exp(pred))
        df = pd.DataFrame({"label": ROMAN_NUMERALS, "proba": proba})
        fig = alt.Chart(df).mark_bar().encode(x="label", y="proba")
        st.altair_chart(fig)
