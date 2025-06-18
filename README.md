# Order of execution
1. First for the OCR of the given three image the jupyter notebook imgprocess.ipynb does that and are saved in ocr_text.
2. And for the extraction of text from the pdf run the pdf_process.py which extracts the text and stores it in the_pdf.txt.
3. And after that run the lang.py for the main LangChain app (Also I hardcoded API pls don't use it). It reads the text from the_pdf.txt and uses GROQ API llama3-8b-8192 as the llm.