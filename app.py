import streamlit as st
import pdfplumber
import torch
import base64
import re
import docx
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

# Load T5 model
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)
summarizer = pipeline('summarization', model=base_model, tokenizer=tokenizer)

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    if not text:
        return "Empty PDF uploaded."
    return text

# Extract text from Word document
def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text if text else "No text extracted. Check if the document contains readable text."

# Extract key lease terms
def extract_key_lease_details(text):
    lease_details = {
        "Lessee(s)": re.search(r"(?i)(?:Lessee.*?resident\(s\):\s*)([^\n]+)", text),
        "Lessor (Landlord)": re.search(r"(?i)(?:Lessor[s]*|Landlord[s]*):?\s*([^\n]+)", text),
        "Property Location": re.search(r"(?i)(?:Premises|Property Address|Rental Location|Property Located at):?\s*([^\n]+)", text),
        "Lease Duration": re.search(r"(?i)(?:Lease Duration|Term of Lease|Rental Period|Start Date|End date):?\s*([^\n]+)", text),
        "Monthly Rent": re.search(r"(?i)(?:Monthly Rent|Base Rent|Rent Amount):?\s*\$?([\d,]+\.?\d*)", text),
        "Security Deposit": re.search(r"(?i)(?:Security Deposit|Deposit Amount):?\s*\$?([\d,]+\.?\d*)", text),
        "Total Lease Amount": re.search(r"(?i)(?:Total Rent|Lease Term|Total Amount Due):?\s*\$?([\d,]+\.?\d*)", text),
        "Late Payment Fee": re.search(r"(?i)(?:Late Fee|Late Payment Charges|Late Payment Policy|Late Rent Fee):?\s*\$?([\d,]+\.?\d*)", text),
        "Rent Payment Method": re.search(r"(?i)(?:Rent Payment Method|Payment Instructions|Payment Portal):?\s*(.*?)(?=\n\d+\.\s|\n[A-Z])", text, re.DOTALL),
        "Maintenance Request": re.search(r"(?i)(?:Maintenance Requests|Repair Responsibility|Upkeep Duties|Lessee to Maintain):?\s*([^\n]+)", text),
        "Termination Clause": re.search(r"(?i)(?:Termination Clause|End of Lease|Termination Policy|SURRENDER OF POSSESSION):?\s*([^\n]+)", text)
    }

    for key, match in lease_details.items():
        if key in ["Monthly Rent", "Security Deposit", "Total Lease Amount", "Late Payment Fee"] and match:
            lease_details[key] = f'<span style="color:green;"><b>${match.group(1).strip()}</b></span>'
        else:
            lease_details[key] = match.group(1).strip() if match else '<span style="color:red;">Not Found</span>'

    return lease_details

# Summarize lease contract
def extract_summary(text, lease_details):
    document_type = lease_details.get("Document Type", "Lease Contract")
    lessee = lease_details.get("Lessee(s)", "Not specified")
    lessor = lease_details.get("Lessor (Landlord)", "Not specified")
    property_location = lease_details.get("Property Location", "Not specified")

    concise_input = (
        f"The {document_type} is an agreement between {lessee} and {lessor}. "
        f"The property is located at {property_location}. "
        f"It outlines the terms and financial policies of the lease."
    )

    summary = summarizer(concise_input, max_length=150, min_length=100, truncation=True, do_sample=False)[0]['summary_text'].strip()
    return summary

# Summarize additional lease terms
def summarize_additional_terms(details):
    selected_keys = ["Rent Payment Method", "Maintenance Request", "Termination Clause"]
    term_summaries = {}

    for key in selected_keys:
        value = details.get(key, "").strip()

        if value and value.lower() != "not found":
            if key == "Rent Payment Method":
                summary = f"Pay rent via: {value.split('at: ')[-1].strip()}"
            elif key == "Maintenance Request":
                summary = summarizer(
                    f"Summarize this maintenance request policy briefly: {value}",
                    max_length=18, min_length=20, truncation=True, do_sample=False
                )[0]['summary_text'].strip()
            else:
                prompt = f"Summarize this briefly in 10 words or less: {value}"
                try:
                    summary = summarizer(
                        prompt, max_length=20, min_length=10, truncation=True, do_sample=False
                    )[0]['summary_text'].strip()
                except Exception:
                    summary = "Error summarizing this section."
        else:
            summary = "Not specified in the lease."

        term_summaries[key] = summary

    return term_summaries

# Extract utilities provider list
import re

def extract_utility_providers(text):
    # Step 1: Add space between name and number if the number sticks to the name
    text = re.sub(r'([a-zA-Z)])(?=\(?\d{3}[-.\s]?\d{3}[-.\s]?\d{4})', r'\1 ', text)

    # Step 2: Adjust the regex pattern to allow more flexible extraction from full sentences
    phone_pattern = re.compile(
        r'(?P<name>(?:(?!\b(?:contact information|please|you can|reach out)\b)[A-Za-z0-9&().,\- ]{3,100}))\s*[-–:]?\s*'
        r'(?P<number>(?:\+?1[-.\s]*)?(?:\(?\d{3}\)?[-.\s]*)\d{3}[-.\s]*\d{4})',
        re.IGNORECASE
    )

    results = []
    seen = set()

    for match in phone_pattern.finditer(text):
        name = match.group("name").strip(" .,-:\n")
        number = match.group("number").strip()

        # Step 3: Filter out irrelevant phrases like "contact information" and "reach out"
        if any(phrase in name.lower() for phrase in ["contact information", "please", "you can", "reach out"]):
            continue

        # Step 4: Only keep valid entries that match known utility providers (you can add more providers as needed)
        known_utility_keywords = ["United Illuminating", "Southern Connecticut Gas", "Eversource", "XFinity", "AT&T", "Frontier", "T-mobile"]
        if any(keyword in name for keyword in known_utility_keywords):
            # Step 5: Skip overly short names or invalid entries
            if len(name) < 3 or len(name.split()) < 1:
                continue

            # Step 6: Capture entries without duplicating
            entry = f"{name} {number}"
            if entry not in seen:
                seen.add(entry)
                results.append(entry)

    return results

# Display PDF in Streamlit (without caching)
def displayPDF(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit App
st.set_page_config(layout='wide', page_title="Lease Contract Summarization App")

def main():
    st.title('📄 Lease Contract Summarization App')
    st.markdown("Upload a lease contract (PDF or DOCX) to extract key terms and get a summary.")

    with st.sidebar:
        st.header("Upload Lease Contract")
        uploaded_file = st.file_uploader("Upload your Lease Contract (PDF or DOCX)", type=['pdf', 'docx'])

    if uploaded_file is not None:
        with st.spinner("Processing your document..."):
            if uploaded_file.name.endswith(".pdf"):
                text = extract_text_from_pdf(uploaded_file)
                if text == "Empty PDF uploaded.":
                    st.error("Empty PDF uploaded. Please upload a PDF with content.")
                    return
            elif uploaded_file.name.endswith(".docx"):
                text = extract_text_from_docx(uploaded_file)
            else:
                text = "Unsupported file format."

            structured_details = extract_key_lease_details(text)
            lease_summary = extract_summary(text, structured_details)
        tab1, tab2, tab3 = st.tabs(["📜 Lease Summary", "📂 View Document", "📑 Additional Terms"])

        with tab1:
            st.subheader("📑 Extracted Lease Contract Summary")
            st.success(lease_summary)

            with st.expander("🔍 View Key Lease Terms", expanded=True):
                def format_bullet_points(details, keys):
                    return "\n".join([f"- **{key}**: {details[key]}" for key in keys])

                st.markdown("### 🏠 General Lease Information")
                st.markdown(format_bullet_points(structured_details, ["Lessee(s)", "Lessor (Landlord)", "Property Location", "Lease Duration"]), unsafe_allow_html=True)

                st.markdown("### 💰 Financial Terms")
                st.markdown(format_bullet_points(structured_details, ["Monthly Rent", "Security Deposit", "Total Lease Amount", "Late Payment Fee"]), unsafe_allow_html=True)

        with tab2:
            st.subheader("📄 Uploaded Lease Contract")
            if uploaded_file.name.endswith(".pdf"):
                displayPDF(uploaded_file)
            else:
                st.text_area("Extracted Text from Word Document", text, height=400)

        with tab3:
            st.subheader("📑 Additional Lease Terms Summary")
            additional_term_summaries = summarize_additional_terms(structured_details)
            for term, summary in additional_term_summaries.items():
                st.markdown(f"**{term}:** {summary}", unsafe_allow_html=True)

            st.markdown("### 🔌 Utilities Providers")
            utility_providers = extract_utility_providers(text)
            for provider in utility_providers:
                st.markdown(f"- {provider}")
            
if __name__ == "__main__":
    main()
