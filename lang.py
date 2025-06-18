from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq.chat_models import ChatGroq


api_key = YOUR_GROQ_API_KEY


# first load the text from the given pdf
def load_txt(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    return text


text = load_txt('the_pdf.txt')
print(text)

llm = ChatGroq(api_key=api_key, model="llama3-8b-8192")
# print(llm)

summary_prompt = PromptTemplate(input_variables=["text"], template="Summarize the following:\n\n{text}")

entity_prompt = PromptTemplate(input_variables=["text"], template="Extract names, dates, organizations, and locations:\n\n{text}")


summary_chain = summary_prompt | llm | StrOutputParser()
entity_chain = entity_prompt | llm | StrOutputParser()


parallel_chain = RunnableParallel(branches={"summary": summary_chain, "entities": entity_chain})


def combine_outputs(data):
    return f"Summary:\n{data['summary']}\n\nEntities:\n{data['entities']}"


final_chain = parallel_chain | RunnableLambda(combine_outputs)


result = final_chain.invoke({"text": text})
print(result)

qa_prompt = PromptTemplate(
    input_variables=["question", "document"],
    template="""Answer the following question using the given content.

    Question: {question}
    Context: {document}"""
)


qa_chain = qa_prompt | llm | StrOutputParser()


while True:
    question = input("Qn: ")
    if question.strip().lower() == "exit":
        print("Exiting")
        break

    qa_input = {"question": question, "document": text}

    answer = qa_chain.invoke(qa_input)
    print("AI: ", answer)
