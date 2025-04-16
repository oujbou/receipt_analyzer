from setuptools import  setup, find_packages

setup(
    name="receipt_analyser",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-dotenv",
        "pydantic",
        "llama-index",
        "llama-index-agent-openai",
        "mistralai",
        "requests",
        "pillow",
        "streamlit",
    ],
    python_requires=">=3.11",
    author="Oujlakh Tarik",
    author_email="toujlakh@gmail.com",
    description="An intelligent receipt analyzer using OCR and LLMs",
)