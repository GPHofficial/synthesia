import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import time
from model.main import predictApp
from model.model import WordLSTM

input = {1: [['v', 43, 2]], 3: [['v', 43, 2]], 5: [['v', 48, 2]], 7: [['v', 43, 3]], 11: [['v', 48, 2]], 13: [['v', 43, 2]], 15: [['v', 48, 3]], 19: [['v', 43, 3]], 22: [['v', 48, 1]], 23: [['v', 41, 1]], 24: [['v', 39, 1]], 25: [['v', 38, 1]], 26: [['v', 36, 1]], 27: [['v', 34, 1]], 29: [['v', 34, 1]], 31: [['v', 41, 2]], 33: [['v', 46, 1]], 35: [['v', 41, 1]], 37: [['v', 38, 1]], 39: [['v', 39, 6]], 45: [['v', 41, 1]], 47: [['v', 44, 4]], 51: [['v', 46, 4]], 55: [['v', 45, 12]], 58: [['v', 46, 1]], 59: [['v', 43, 1]], 60: [['v', 46, 1]], 61: [['v', 43, 1]], 62: [['v', 41, 1]], 63: [['v', 44, 1]], 64: [['v', 43, 1]], 65: [['v', 41, 1]], 66: [['v', 43, 1]], 67: [['v', 41, 1]], 68: [['v', 43, 1]], 69: [['v', 44, 1]], 70: [['v', 43, 1]], 71: [['v', 41, 1]], 73: [['v', 43, 1]], 74: [['v', 41, 1]], 75: [['v', 43, 1]], 77: [['v', 39, 1]], 78: [['v', 41, 1]], 79: [['v', 43, 2]], 82: [['v', 43, 1]], 83: [['v', 44, 1]], 84: [['v', 43, 1]], 85: [['v', 41, 1]], 86: [['v', 39, 1]], 87: [['v', 38, 1]], 88: [['v', 36, 1]], 89: [['v', 38, 4]], 95: [['v', 39, 12]]}

_selectable_data_table = components.declare_component("selectable_data_table", path="frontend/build")

dimensions = [10,62]
inputDict = {'dimensions': dimensions}
st.title("Synthesia - Generated Music")
st.markdown('''
    Welcome!  
    The following grid is a sequence of midi notes that you can select:  
    Red indicates Violin (Click Once)  
    Blue indicates Piano (Click Twice)  
    Magenta indicates both Violin and Piano (Click Thrice)
 ''')
st.subheader("Start Sequence")
rows = _selectable_data_table(data=inputDict, default=[], key="Synthesia")

if rows:
    input_sequence = ' '.join(rows)
    print(input_sequence)
    st.subheader("Generated Sequence")
    st.markdown('''
        Possible Parameters that can be adjusted during generation:  
        Number of notes to generate  
        Time co-efficent  
        (Currently still edited in code)
    ''')
    with st.spinner("Playing the violin and piano simultaneously"):
        input = predictApp(seed_prompt = input_sequence,tokens_to_generate = 512, time_coefficient = 4, top_k_coefficient = 12)
        
    dimensions[0] = 28
    inputDict = {'dimensions': dimensions, 'input': input}
    rows = _selectable_data_table(data=inputDict, default=[], key="synthesia-output")

    audio_file = open('./model/output/output.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{
                max-width: {1920}px;
                
            }}
        </style>
        """,
                unsafe_allow_html=True,
)