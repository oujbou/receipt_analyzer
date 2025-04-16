# Receipt Analyzer

An intelligent receipt analysis system that uses OCR and LLMs to extract, categorize, and validate receipt information.

## Features

- Upload receipt images 
- Extract vendor information, date, items, and totals
- Categorize expenses automatically
- Validate calculations with self-correction
- Interactive UI for reviewing and editing results

## Technology Stack

- Python 3.11+
- Mistral OCR API for image processing
- Mistral LLM for intelligent analysis
- LlamaIndex for agentic framework
- Streamlit for user interface

## Setup

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your API keys
6. Run the application: `streamlit run ui/app.py`

## Project Structure

- `app/` - Core application logic
- `ui/` - Streamlit user interface
- `tests/` - Test suite

## Development

- Run tests: `pytest`
- Format code: `black .`
- Check linting: `pylint app/ ui/ tests/`