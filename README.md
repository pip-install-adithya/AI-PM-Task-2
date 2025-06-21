# Order of execution
1. First for the OCR of the given three image the jupyter notebook imgprocess.ipynb does that and are saved in ocr_text.
2. And for the extraction of text from the pdf run the pdf_process.py which extracts the text and stores it in the_pdf.txt.
3. And after that run the lang.py for the main LangChain app (Also I hardcoded API pls don't use it). It reads the text from the_pdf.txt and uses GROQ API llama3-8b-8192 as the llm.

# Update - main.py
main.py is the main code that does the dynamic answering. (also since it is just combining two files that are already there i'm not putting that much comments). Also short points on what I did in it:
1. I added parallel OCR which does OCR for each page in the pdf in parallel using the ParallelPoolExecutor (didn't check the time improvement yet).
2. For the main langchain part I did the summary and key informations also because I thought it will improve the quality of the responses in the dynamic qa-ing.
3. I also added chat_history and giving it as input in the dynmic qa chain for better context.
4. Also stored the apikey in .env this time...