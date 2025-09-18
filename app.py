%%writefile app.py
import streamlit as st
import PyPDF2
from difflib import get_close_matches
import pandas as pd
from io import BytesIO
import pytesseract
from pdf2image import convert_from_bytes
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

st.title("AI-Powered Medical Report Simplifier (Colab + OCR)")

# -------------------------------
# Load Hugging Face model (public, no auth needed)
# -------------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/gpt4all-j")
    model = AutoModelForCausalLM.from_pretrained("nomic-ai/gpt4all-j")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, torch_dtype=torch.float16)
    return generator

generator = load_model()

# -------------------------------
# Reference ranges & aliases
# -------------------------------
REFERENCE_RANGES = {
    "Hemoglobin": (13.5, 17.5),
    "WBC": (4.0, 11.0),
    "Platelets": (150, 450),
    "RBC": (4.5, 5.9),
    "Cholesterol": (0, 200),
    "HDL": (40, 60),
    "LDL": (0, 130),
    "Triglycerides": (0, 150),
}

ALIAS = {
    "Hb": "Hemoglobin",
    "White Blood Cells": "WBC",
    "Platelet Count": "Platelets",
    "Red Blood Cells": "RBC",
    "Total Cholesterol": "Cholesterol",
    "Triglyceride": "Triglycerides",
}

# -------------------------------
# Helper functions
# -------------------------------
def extract_text_from_pdf(pdf_bytes):
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text and page_text.strip():
            text += page_text + "\n"
    if not text.strip():
        st.info("No selectable text found, using OCR...")
        text = ocr_pdf(pdf_bytes)
    return text

def ocr_pdf(pdf_bytes):
    images = convert_from_bytes(pdf_bytes, dpi=300)
    ocr_text = ""
    for img in images:
        ocr_text += pytesseract.image_to_string(img) + "\n"
    return ocr_text

def parse_report_lines(report_text):
    results = {}
    for line in report_text.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().split()[0]
            try:
                results[key] = float(value)
            except ValueError:
                continue
    return results

def normalize_test_name(test_name):
    if test_name in ALIAS:
        return ALIAS[test_name]
    matches = get_close_matches(test_name, REFERENCE_RANGES.keys(), n=1, cutoff=0.7)
    if matches:
        return matches[0]
    return test_name

def detect_abnormal_results(results):
    abnormalities = {}
    normalized_results = {}
    for test, value in results.items():
        norm_test = normalize_test_name(test)
        normalized_results[norm_test] = value
        if norm_test in REFERENCE_RANGES:
            low, high = REFERENCE_RANGES[norm_test]
            if value < low:
                abnormalities[norm_test] = f"{value} (Low)"
            elif value > high:
                abnormalities[norm_test] = f"{value} (High)"
    return normalized_results, abnormalities

def detect_report_type(normalized_results):
    if any(x in normalized_results for x in ["Hemoglobin", "WBC", "Platelets", "RBC"]):
        return "CBC"
    elif any(x in normalized_results for x in ["Cholesterol", "HDL", "LDL", "Triglycerides"]):
        return "Lipid Profile"
    else:
        return "General Lab Report"

def display_results_table(normalized_results, abnormalities):
    data = []
    for test, value in normalized_results.items():
        if test in abnormalities:
            status = abnormalities[test].split(' ')[-1]
            data.append((test, value, status))
        else:
            data.append((test, value, "Normal"))
    df = pd.DataFrame(data, columns=["Test", "Value", "Status"])
    def highlight_abnormal(val):
        if val == "Low" or val == "High":
            return 'color: red'
        else:
            return ''
    st.dataframe(df.style.applymap(highlight_abnormal, subset=['Status']))

# -------------------------------
# Streamlit UI
# -------------------------------
uploaded_file = st.file_uploader("Upload your medical report PDF", type="pdf")

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    full_text = extract_text_from_pdf(pdf_bytes)
    st.subheader("Original Report Preview (first 500 chars)")
    st.text(full_text[:500])

    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if not page_text or not page_text.strip():
            page_text = ocr_pdf(pdf_bytes)

        st.markdown(f"### Page {i+1}")
        parsed_results = parse_report_lines(page_text)
        if not parsed_results:
            st.text("No numeric results detected on this page.")
            continue

        normalized_results, abnormalities = detect_abnormal_results(parsed_results)
        report_type = detect_report_type(normalized_results)
        st.subheader(f"Detected Report Type: {report_type}")
        display_results_table(normalized_results, abnormalities)

        prompt = f"""
        You are a medical assistant. Simplify this {report_type} for a patient in plain language.
        Highlight abnormal results and explain them clearly. Example: "Your hemoglobin is low => possible anemia."

        Medical report (page {i+1}):
        {page_text}

        Abnormal results (if any):
        {abnormalities}

        Simplified explanation:
        """

        with st.spinner(f"Simplifying report on page {i+1}..."):
            simplified = generator(prompt, max_new_tokens=500, do_sample=True, temperature=0.7)[0]['generated_text']
            st.subheader("Patient-Friendly Explanation")
            st.text(simplified)
