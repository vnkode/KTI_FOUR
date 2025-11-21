import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import base64
import requests

load_dotenv(".env")

class GeminiAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={self.api_key}"

    def image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_response(self, image_path, prompt):
        base64_image = self.image_to_base64(image_path)
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/png", "data": base64_image}}
                ]
            }]
        }

        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status() 
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        except requests.exceptions.RequestException as e:
            return f"Error koneksi API Gemini: Gagal terhubung atau menerima respons. {e}"
        except (KeyError, IndexError):
            return "Error API Gemini: Respons tidak terduga dari model. Coba gambar lebih jelas."


api_key = os.getenv('GEMINI_API_KEY') 
if not api_key:
    st.error("❌ GEMINI_API_KEY tidak ditemukan. Silakan set environment variable atau buat file .env") 
    st.stop()

gemini_api = GeminiAPI(api_key)

new_title = '<p style="font-family:tahoma; color:#a67a16; font-size: 42px;"><b>⚛︎ CalcSketch | KTI Demo</b></p>'
st.markdown(new_title, unsafe_allow_html=True)

if 'canvas' not in st.session_state:
    st.session_state.canvas = np.zeros((480, 640, 3), dtype="uint8")

canvas_result = st_canvas(
    fill_color="rgba(229, 171, 19, 0.8)", 
    stroke_width=2,
    stroke_color="#08081a",
    background_color="#baa488",
    width=800, 
    height=480,
    drawing_mode="freedraw",
    key="canvas",
    update_streamlit=True,
)

st.stylable_container(
    key="my_button_container",
    css_styles="""
    button {
        background-color: #FF0000;
        color: white;
        font-size: 18px;
    }
    """
);
    
st.button("Solve")
if st.button(): 
    if canvas_result.image_data is not None: 
        
        img = Image.fromarray(canvas_result.image_data.astype("uint8"), 'RGBA')
        img.save("canvas.png")
        
        user_prompt = "Selesaikan masalah matematika yang digambar. Berikan hanya jawaban akhir yang paling ringkas dan langkah-langkah perhitungan yang paling penting. Format output secara ketat menggunakan Markdown, dan gunakan LaTeX diapit oleh $$ untuk semua persamaan."

        st.header("Hasil")
        
        response = gemini_api.get_response("canvas.png", user_prompt)
        
        st.markdown(response)

















