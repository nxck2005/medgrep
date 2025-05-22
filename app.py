import streamlit as st
from PIL import Image
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from sentence_transformers import SentenceTransformer

# Load TrOCR only once
@st.cache_resource
def load_trocr():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return processor, model

# Load embedding model once
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

trocr_processor, trocr_model = load_trocr()
embed_model = load_embedder()

st.title("📄 Medical Report Reader (Printed + Handwritten)")
ocr_mode = st.radio("Choose OCR type:", ["🖨️ Printed", "✍️ Handwritten"])

uploaded_file = st.file_uploader("Upload a medical report image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Report", use_column_width=True)

    st.subheader("🧠 Extracted Text")

    if ocr_mode == "🖨️ Printed":
        text = pytesseract.image_to_string(image)
    else:
        with st.spinner("Running TrOCR..."):
            pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values
            generated_ids = trocr_model.generate(pixel_values)
            text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    st.code(text.strip() or "No text detected.")

    if text.strip():
        st.subheader("📝 Summary (via sentence embeddings)")
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
        if sentences:
            embeddings = embed_model.encode(sentences)
            scores = [sum(e**2 for e in emb) for emb in embeddings]
            summary = sentences[scores.index(max(scores))]
            st.success(summary)
        else:
            st.info("Couldn't find meaningful sentences to summarize.")
