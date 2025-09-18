import streamlit as st
import PyPDF2
from difflib import get_close_matches
import pandas as pd
from io import BytesIO
import pytesseract
from pdf2image import convert_from_bytes
from transformers import pipeline
import torch

st.title("AI-Powered Medical Report Simplifier 🩺")

# -------------------------------
# Load Hugging Face model
# -------------------------------
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-large")

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
            value = value.strip()
            if not value:
                continue
            parts = value.split()
            if not parts:
                continue
            try:
                results[key] = float(parts[0])
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
            status = f"⚠️ {abnormalities[test].split(' ')[-1]}"
        else:
            status = "✅ Normal"
        data.append((test, value, status))

    df = pd.DataFrame(data, columns=["Test", "Value", "Status"])

    def highlight_abnormal(val):
        if "⚠️" in str(val):
            return 'color: red; font-weight: bold'
        else:
            return 'color: green'
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

    all_results = {}
    all_abnormalities = {}

    # ---- Collect results from all pages ----
    for page in reader.pages:
        page_text = page.extract_text()
        if not page_text or not page_text.strip():
            page_text = ocr_pdf(pdf_bytes)

        parsed_results = parse_report_lines(page_text)
        if not parsed_results:
            continue

        normalized_results, abnormalities = detect_abnormal_results(parsed_results)

        # Merge into one big dict
        all_results.update(normalized_results)
        all_abnormalities.update(abnormalities)

    # ---- Show results + single summary ----
    if all_results:
        report_type = detect_report_type(all_results)
        st.subheader(f"Detected Report Type: {report_type}")

        # Show results table once
        display_results_table(all_results, all_abnormalities)

        # ---- Prompt for unified summary ----
        prompt = f"""
        Summarize this {report_type} medical report in ONE non-repetitive explanation for the patient.

        Follow this exact format:

        ✅ Normal Results:
        - TestName: short note

        ⚠️ Abnormal Results:
        - TestName: short explanation of why high/low matters

        🩺 Overall Health Impression:
        - 2–3 simple sentences of advice

        Do not repeat the same test more than once.
        Do not rephrase abnormalities multiple times.
        Use plain, friendly language.

        Test results:
        {all_results}

        Abnormal results:
        {all_abnormalities}
        """

        with st.spinner("Generating final summary..."):
            simplified = generator(
                prompt,
                max_new_tokens=350,
                do_sample=False,   # no randomness
                temperature=0.0    # deterministic & concise
            )[0]['generated_text']

        st.subheader("Final Patient-Friendly Summary")
        st.markdown(simplified)

    else:
        st.warning("No test results detected in the uploaded PDF.")
