import streamlit as st
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

@st.cache_data
def blip2Prompt(url):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
    )
    model.to(device)
    image = Image.open(url)
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return generated_text

@st.cache_data
def blipBase(url):

    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    raw_image = Image.open(url)

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


def sentenceSimilarity(text1, text2):

    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer('all-mpnet-base-v2')
    sentences = [text1, text2]
    embeddings = model.encode(sentences, convert_to_tensor=True)
    cos_sim = util.cos_sim(embeddings, embeddings)

    # Add all pairs to a list with their cosine similarity score
    all_sentence_combinations = []
    for i in range(len(cos_sim) - 1):
        for j in range(i + 1, len(cos_sim)):
            all_sentence_combinations.append({'index': [i, j], 'score': cos_sim[i][j]})

    # Sort list by the highest cosine similarity score
    all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x['score'], reverse=True)
    return all_sentence_combinations[0]['score']
