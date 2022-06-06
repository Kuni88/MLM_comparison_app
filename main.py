import streamlit as st
import plotly.graph_objects as go
import os
import shutil
from transformers import pipeline
from huggingface_hub import HfApi, ModelFilter

CACHE_PATH = '/transformers/cache/'
os.environ['TRANSFORMERS_CACHE'] = CACHE_PATH


## Config
st.set_page_config(
    page_title="MLM Comparison app",
    page_icon="🚀",
)
st.header("MLM comparison")

@st.cache()
def run_and_visualize(model, text, topk):
    pipe = pipeline('fill-mask', model=model)
    text = text.replace('[MASK]', pipe.tokenizer.mask_token)
    res = pipe(text, top_k=topk)
    fig = go.Figure(
        go.Bar(
            x=[r['score'] for r in res[::-1]],
            y=[r['token_str'] for r in res[::-1]],
            orientation='h',
        )
    )
    return fig

TEMPLATES = {
    'en': 'Paris is the [MASK] of France.',
    'ja': '大学で[MASK]の研究をしています。',
}

lang = st.selectbox("Select a language", list(TEMPLATES.keys()))
filter = ModelFilter(language=lang, task='fill-mask')
api = HfApi()
hf_models = [model.id for model in api.list_models(filter=filter)]

models = st.multiselect("Choose two models", options=hf_models, default=hf_models[0:2])
text = st.text_input("Input texts (Mask token: [MASK])", TEMPLATES[lang])
topk = st.number_input("Topk", min_value=1, max_value=10, value=5, step=1)

if len(models) != 2:
    st.error("Please select two models to compare them")

col1, col2 = st.columns(2)
if st.button("Run"):
    with col1:
        st.title(models[0])
        with st.container():
            fig1 = run_and_visualize(models[0], text, topk)
            st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.title(models[1])
        with st.container():
            fig2 = run_and_visualize(models[1], text, topk)
            st.plotly_chart(fig2, use_container_width=True)

    stat = shutil.disk_usage("/")
    st.write(stat)
