# Lease-Contract-Summarization App
A local Streamlit-based app that extracts and summarizes lease contracts using a LaMini-Flan-T5 model. No internet access required — runs entirely on your local system.


## Features

- Extracts and displays:
  - Lessee and Lessor names
  - Lease period, rent amount, security deposit
  - Payment methods, maintenance, and termination terms
  - Utility provider names and phone numbers
- Generates concise summaries for legal and financial terms
- Supports PDF and DOCX lease files

---

## File Upload Formats

- PDF (`.pdf`)
- Word Document (`.docx`)

---

## Model Used

This app uses the [LaMini-Flan-T5-248M](https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M) model from Hugging Face to generate summaries.


---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/lease-summary-app.git
   cd lease-summary-app

---

## Steps

- python -m venv venv 
  source venv/bin/activate   # On Windows: venv\Scripts\activate
- pip install -r requirements.txt
- streamlit run app.py
  


   
