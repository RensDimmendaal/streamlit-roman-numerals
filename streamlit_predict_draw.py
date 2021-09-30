import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from rens import load_img_32, create_model


ROMAN_NUMERALS = "i ii iii iv v vi vii viii ix x".upper().split()
REVERSED_NUMERALS = {r: i for i, r in enumerate(ROMAN_NUMERALS, start=1)}


@st.cache
def load_model(directory):
    return create_model(compile=True, trained_weights=f"{directory}/best_model")


model = load_model(".")


# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)


# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=None,
    update_streamlit=realtime_update,
    width=150,
    height=150,
    drawing_mode=drawing_mode,
    key="canvas",
)


# Do something interesting with the image data and paths
if st.button("predict"):
    if canvas_result.image_data is not None:
        # save
        # st.write(canvas_result.image_data / 255)
        _ = plt.imsave("delme.png", canvas_result.image_data / 255, cmap="gray")

        # loadcorrect
        img = load_img_32("delme.png")
        # fig, ax = plt.subplots()
        # ax.imshow(img)
        # st.pyplot(fig)
        pred = model(img[np.newaxis, ...]).numpy().flatten()
        proba = np.exp(pred) / np.sum(np.exp(pred))
        df = pd.DataFrame({"label": ROMAN_NUMERALS, "proba": proba})
        fig = alt.Chart(df).mark_bar().encode(x="label", y="proba")
        st.altair_chart(fig)
