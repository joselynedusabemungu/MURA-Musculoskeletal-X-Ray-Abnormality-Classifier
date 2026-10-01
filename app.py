from pathlib import Path
import sys
import streamlit as st
from PIL import Image
import pandas as pd

ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/'src'))
from mura_utils import load_bundle,predict_image
MODEL=ROOT/'models'/'mura_hog_logistic.joblib'
st.set_page_config(page_title='MURA X-ray Review',layout='wide')
st.title('MURA Musculoskeletal X-ray Review')
st.caption('Lightweight scikit-learn coursework prototype: Normal vs Abnormal')
st.warning('Not a medical device. Do not use this output for diagnosis or treatment decisions.')
if not MODEL.exists():
    st.error('Trained model not found. Run the notebook through the export section first.')
    st.stop()
bundle=load_bundle(MODEL); model=bundle['model']; meta=bundle.get('metadata',{})
with st.sidebar:
    st.header('Model details'); st.write('Features: resized 128×128 grayscale pixels'); st.write('Classifier: balanced logistic regression'); st.write('Input: one PNG/JPG radiograph'); st.json(meta)
up=st.file_uploader('Upload a musculoskeletal X-ray',type=['png','jpg','jpeg'])
if up:
    img=Image.open(up).convert('RGB'); result=predict_image(model,img)
    a,b=st.columns(2)
    with a: st.subheader('Uploaded image'); st.image(img,use_container_width=True)
    with b:
        st.subheader('Prediction'); st.metric('Class',result['label'],f"{result['confidence']:.1%} confidence")
        st.bar_chart(pd.Series(result['probabilities']))
        if result['confidence']<.65: st.warning('Low confidence: treat this result as inconclusive.')
        st.write(f"The model predicts **{result['label'].lower()}** for this radiograph. This is an automated coursework estimate, not a diagnosis.")
else: st.info('Upload an image to receive a real prediction.')
