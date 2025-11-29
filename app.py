# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
import json
import os
import base64
import requests
import tempfile
from datetime import datetime

# ML libs
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# webcam
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# maps
import folium
from streamlit.components.v1 import components
from geopy.geocoders import Nominatim

# PDF export
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Skin Disease Real-time Detector", layout="wide")

# ---------- Helper: load labels ----------
LABELS_PATH = "labels.json"
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        LABELS = json.load(f)
else:
    LABELS = ["benign", "melanoma", "eczema", "psoriasis"]  # fallback labels

# ---------- Helper: load model ----------
MODEL_PATH = "model.h5"
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.info("Loaded model.h5")
    except Exception as e:
        st.warning(f"Could not load model.h5: {e}\nUsing fallback predictor.")
        model = None
else:
    st.info("No model.h5 found — using fallback/dummy predictor. Place your trained model at model.h5 to enable real predictions.")

# ---------- Image preprocessing & predict ----------
IMG_SIZE = (224, 224)

def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    return arr

def predict_image(pil_img):
    arr = preprocess_image(pil_img)
    if model is not None:
        preds = model.predict(arr)[0]
        # if model outputs logits, apply softmax
        if preds.sum() <= 1.001 and preds.sum() >= 0.999:
            probs = preds
        else:
            probs = tf.nn.softmax(preds).numpy()
        top_idx = int(np.argmax(probs))
        return {"label": LABELS[top_idx], "confidence": float(probs[top_idx]), "probs": probs.tolist()}
    else:
        # Dummy prediction: random but deterministic from image content
        seed = int(np.sum(np.array(arr) * 255) % 100000)
        rng = np.random.RandomState(seed)
        probs = rng.rand(len(LABELS))
        probs = probs / probs.sum()
        top_idx = int(np.argmax(probs))
        return {"label": LABELS[top_idx], "confidence": float(probs[top_idx]), "probs": probs.tolist()}

# ---------- Webcam transformer ----------
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # small overlay: nothing heavy, just return frame
        return img

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ---------- Chatbot ----------
def local_chatbot_response(user_text):
    # Very simple rule-based responses; expand as needed
    text = user_text.lower()
    if "hello" in text or "hi" in text:
        return "Hello! I can help you test images, explain results, or find nearby hospitals. Try uploading an image or asking 'find hospitals near me'."
    if "what is" in text and "melanoma" in text:
        return "Melanoma is a type of skin cancer. If the model suggests melanoma, please consult a dermatologist immediately."
    if "nearby" in text or "hospital" in text:
        return "Use the 'Nearby Hospitals' panel and enter your city or allow location. I can also use Google Places if you provide an API key."
    return "Sorry, I don't know that yet. Try asking about uploading images, predictions, or nearby hospitals."

# Optional OpenAI chatbot integration (requires OPENAI_API_KEY env var)
def openai_chatbot_response(prompt):
    try:
        import openai
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return "OpenAI API key not provided. Set OPENAI_API_KEY environment variable."
        openai.api_key = key
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or "gpt-4o" / "gpt-4o-mini"
            messages=[{"role":"user","content":prompt}],
            max_tokens=300
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI chat failed: {e}"

# ---------- Nearby hospitals: two modes ----------
def find_hospitals_google_maps(api_key, query, lat=None, lng=None, radius=5000):
    """Use Google Places Text Search (requires API key)."""
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {"key": api_key, "radius": radius, "type": "hospital"}
    if lat and lng:
        params["location"] = f"{lat},{lng}"
    else:
        params["keyword"] = query
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

def find_hospitals_overpass(city_name, radius_km=5):
    """Free alternative: use Nominatim to geocode then Overpass to find hospitals (no API key)."""
    geolocator = Nominatim(user_agent="skin-detect-app")
    loc = geolocator.geocode(city_name)
    if not loc:
        return {"error": "Could not geocode location."}
    lat, lon = loc.latitude, loc.longitude
    overpass_url = "http://overpass-api.de/api/interpreter"
    # query hospitals within radius (in meters)
    query = f"""
    [out:json];
    node["amenity"="hospital"](around:{int(radius_km*1000)},{lat},{lon});
    out center;
    """
    r = requests.post(overpass_url, data={"data": query}, timeout=30)
    if r.status_code != 200:
        return {"error": "Overpass query failed."}
    data = r.json()
    hospitals = []
    for el in data.get("elements", []):
        name = el.get("tags", {}).get("name", "Unnamed")
        hospitals.append({
            "name": name,
            "lat": el.get("lat"),
            "lon": el.get("lon"),
            "tags": el.get("tags", {})
        })
    return {"center": (lat, lon), "hospitals": hospitals}

# ---------- PDF report ----------
def generate_pdf_report(image_bytes, prediction, out_path="report.pdf"):
    c = canvas.Canvas(out_path, pagesize=letter)
    width, height = letter
    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 50, "Skin Disease Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(40, height - 80, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Prediction
    c.drawString(40, height - 110, f"Prediction: {prediction['label']} (confidence: {prediction['confidence']:.3f})")
    # Add image
    img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img_temp.write(image_bytes)
    img_temp.flush()
    c.drawImage(img_temp.name, 40, height - 420, width=250, height=250)
    c.showPage()
    c.save()
    return out_path

# ---------- Streamlit layout ----------
st.title("🩺 Skin Disease — Real-time Detection (Streamlit)")

# two-column layout
col1, col2 = st.columns([2,1])

with col1:
    st.header("Image input & Detection")
    input_mode = st.radio("Input mode", ["Upload image", "Use webcam (real-time)"])
    uploaded_image = None
    webcam_image = None

    if input_mode == "Upload image":
        uploaded_file = st.file_uploader("Upload a skin image (jpg/png)", type=["jpg","jpeg","png"])
        if uploaded_file:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded image", use_column_width=True)
            if st.button("Run prediction on uploaded image"):
                with st.spinner("Predicting..."):
                    pred = predict_image(uploaded_image)
                st.success(f"Prediction: **{pred['label']}** — confidence {pred['confidence']:.2f}")
                st.write("All class probabilities:")
                for label, p in zip(LABELS, pred["probs"]):
                    st.write(f"- {label}: {p:.3f}")
                # provide PDF
                buf = io.BytesIO()
                uploaded_image.save(buf, format="PNG")
                pdf_path = generate_pdf_report(buf.getvalue(), pred, out_path="report.pdf")
                with open(pdf_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">Download PDF report</a>'
                    st.markdown(href, unsafe_allow_html=True)

    else:
        st.markdown("Start webcam and click **Capture & Predict** on the frame you want to analyze.")
        webrtc_ctx = webrtc_streamer(key="skin-webrtc", rtc_configuration=RTC_CONFIGURATION, video_transformer_factory=VideoTransformer, media_stream_constraints={"video": True, "audio": False})
        if webrtc_ctx.state.playing:
            if st.button("Capture & Predict"):
                # grab current frame from transformer
                transformer = webrtc_ctx.video_transformer
                if transformer and transformer.frame is not None:
                    import cv2
                    frame = transformer.frame  # numpy array (bgr)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    st.image(pil_img, caption="Captured frame", use_column_width=True)
                    pred = predict_image(pil_img)
                    st.success(f"Prediction: **{pred['label']}** — confidence {pred['confidence']:.2f}")
                    st.write("All class probabilities:")
                    for label, p in zip(LABELS, pred["probs"]):
                        st.write(f"- {label}: {p:.3f}")
                    # PDF
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    pdf_path = generate_pdf_report(buf.getvalue(), pred, out_path="report.pdf")
                    with open(pdf_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                        href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">Download PDF report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                else:
                    st.warning("No frame available yet. Wait a moment for the camera to warm up.")

with col2:
    st.header("Chatbot")
    st.markdown("Ask about the app, skin conditions, or hospitals.")
    chat_mode = st.radio("Chat mode", ["Local (no key)", "OpenAI (requires API key)"])
    user_input = st.text_input("You:", "")
    if st.button("Send"):
        if user_input.strip() == "":
            st.info("Type a message first.")
        else:
            if chat_mode == "Local":
                reply = local_chatbot_response(user_input)
            else:
                reply = openai_chatbot_response(user_input)
            st.markdown(f"**Bot:** {reply}")

    st.markdown("---")
    st.header("Nearby Hospitals")
    hospital_mode = st.radio("Mode", ["Free (OpenStreetMap)", "Google Places (recommended)"])
    if hospital_mode == "Google Places (recommended)":
        gkey = st.text_input("Google Places API key (leave blank to skip)", type="password")
        loc_input = st.text_input("Enter city or location (optional; leave blank to use ip-based/geolocation):", "")
        if st.button("Find hospitals (Google)"):
            if not gkey:
                st.error("Provide Google API key or switch to OpenStreetMap mode.")
            else:
                # try geocode if location provided
                lat = lng = None
                if loc_input:
                    geolocator = Nominatim(user_agent="skin-detect-app")
                    loc = geolocator.geocode(loc_input)
                    if loc:
                        lat, lng = loc.latitude, loc.longitude
                try:
                    res = find_hospitals_google_maps(gkey, loc_input or "hospital", lat, lng)
                    if res.get("results"):
                        st.success(f"Found {len(res['results'])} results (first 10 shown).")
                        # build map
                        center = None
                        if lat and lng:
                            center = (lat, lng)
                        else:
                            r0 = res["results"][0]
                            center = (r0["geometry"]["location"]["lat"], r0["geometry"]["location"]["lng"])
                        m = folium.Map(location=center, zoom_start=13)
                        for h in res["results"][:15]:
                            la = h["geometry"]["location"]["lat"]
                            lo = h["geometry"]["location"]["lng"]
                            name = h.get("name")
                            folium.Marker([la, lo], popup=name).add_to(m)
                        components.html(m._repr_html_(), height=500)
                    else:
                        st.warning("No results from Google Places.")
                except Exception as e:
                    st.error(f"Google Places error: {e}")

    else:
        city = st.text_input("Enter city name for OSM search (e.g., 'Bengaluru, India'):")
        if st.button("Find hospitals (OSM)"):
            if not city:
                st.error("Enter a city name.")
            else:
                with st.spinner("Searching OpenStreetMap/Overpass..."):
                    res = find_hospitals_overpass(city)
                if "error" in res:
                    st.error(res["error"])
                else:
                    center = res.get("center")
                    hospitals = res.get("hospitals", [])
                    st.success(f"Found {len(hospitals)} hospitals (first 20 shown).")
                    m = folium.Map(location=center, zoom_start=13)
                    for h in hospitals[:20]:
                        folium.Marker([h["lat"], h["lon"]], popup=h["name"]).add_to(m)
                    components.html(m._repr_html_(), height=500)

st.markdown("---")
st.write("Tips: upload a clear close-up photo, avoid heavy filters, and consult a dermatologist for confirmation.")
