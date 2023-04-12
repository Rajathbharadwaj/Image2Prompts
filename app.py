import streamlit as st
from im2prompt import blip2Prompt, blipBase, sentenceSimilarity, translate

st.set_page_config(layout="wide", page_title="Image to Prompt Generator")

st.write("## Image 2 Prompt")
st.write(
    ":dog: Try uploading an image to get the prompt of it :grin:"
)
st.sidebar.write("## Upload :gear:")

# Download the fixed image


my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

generate = st.button('Click to Generate the prompt')

if generate:
    if my_upload is not None:
        with st.spinner("Generating Prompt from BLIP2"):
            prompt1 = blip2Prompt(url=my_upload)
            st.success('Done BLIP2')
        with st.spinner('Generating prompt from BLIP Base'):
            prompt2 = blipBase(url=my_upload)
            st.success('Done BLIP Base')
        with st.spinner('Checking Similarity'):
            cosine_sim = sentenceSimilarity(prompt1, prompt2)
            if bool(cosine_sim>0.45):
                st.write(f'The closest match prompt is {prompt2}')
        with st.spinner('Translating to French'):
            st.write(f' The translated text from the prompt is {translate(prompt2)}')

    else:
        st.write("Upload an image")
