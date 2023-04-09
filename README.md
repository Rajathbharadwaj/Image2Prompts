
# Image 2 Prompts

A streamlit application that generates prompts when an Image is uploaded. 
It uses two different models to generate prompts (BLIP2 and BLIP Base), after which it is then piped to a sentence transformer to calculate cosine similarity of the generated prompts by encoding the embeddings. Then the closest match is outputed.


## How to run the project




## Installation


```bash
  git clone https://github.com/Rajathbharadwaj/Image2Prompts.git im2p
  cd im2p
  pip install -r requirements.txt
```

## Running

In the same directory run

```bash
streamlit run app.py
```
    