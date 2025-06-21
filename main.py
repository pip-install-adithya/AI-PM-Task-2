import fitz
from PIL import Image
import pytesseract
import io
from concurrent.futures import ProcessPoolExecutor
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# _____________________________________________________________________
# The OCR Part


# same function as before except getting path, pno. and zoom as args
# and returning page no. also
def process_page(args):
    pdf_path, page_number, zoom = args
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img)
    doc.close()
    return page_number, text


# main function which does the parallel ocr
def ocr(pdf_path, zoom=2.0):
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()

    with ProcessPoolExecutor() as executor:
        # first args for each page
        args = [(pdf_path, i, zoom) for i in range(num_pages)]

        # map is used to basically maintain the order of the pages and perform
        # the process_page function for all the elements in args
        results = list(executor.map(process_page, args))

    all_text = ""
    for page_number, text in results:
        # i only put like this cos it looked structured not gpt
        all_text += f"\n\n--- Page {page_number + 1} ---\n\n{text}"

    return all_text.strip()


# _____________________________________________________________________
# The LangChain Part

def summary_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are an expert summarizer that leaves no information out in the summary."),
        ("human", "Summarize the following:\n\n{text}")
    ])


def entity_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are an expert entity extractor"),
        ("human", "Extract names, dates, organizations, and locations:\n\n{text}")
    ])


# function to combine outputs from parallel chain
def combine_outputs(data):
    return f"Summary:\n{data["branches"]['summary']}\n\nEntities:\n{data["branches"]['entities']}"


# _______________________________________________________________________
# main function
# had to wrap it in main for the parallel ocr
# (some windows )
if __name__ == "__main__":
    load_dotenv()

    text = ocr("the_pdf.pdf")  # whatever pdf name
    # print(text)

    # automatically gets api key from .env
    llm = ChatGroq(model="llama3-8b-8192")
    # print(llm)

    summary_chain = summary_prompt() | llm | StrOutputParser()
    entity_chain = entity_prompt() | llm | StrOutputParser()

    parallel_chain = RunnableParallel(branches={
        "summary": summary_chain,
        "entities": entity_chain
    })

    final_chain = parallel_chain | RunnableLambda(combine_outputs)

    result = final_chain.invoke({"text": text})
    print(result)

    chat_history = []

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions using only the provided context."),
        MessagesPlaceholder(variable_name="chat_history"),  # used to store the chat history for context
        ("human", """Answer the following question using the given content.

    Question: {question}
    Context: {document}
    Summary and Key Informations: {result}""")
    ])

    qa_chain = qa_prompt | llm | StrOutputParser()

    while True:
        question = input("Qn: ")
        if question.strip().lower() == "exit":
            print("Exiting")
            break

        qa_input = {
            "chat_history": chat_history,  # pass the chat history
            "question": question,
            "document": text,
            "result": result
        }

        answer = qa_chain.invoke(qa_input)
        print("AI: ", answer)
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
        print(f"Chat history length rn: {len(chat_history)}")

    print(chat_history)
