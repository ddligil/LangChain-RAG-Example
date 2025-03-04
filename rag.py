from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://maneaddicts.com/blogs/hair-tip/dyson-airstrait")

docs = loader.load()

#web base loader belirtilen URL'den web sayfası içeriğini alır ve bir document nesnesine dönüştürür.

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter()

documents = text_splitter.split_documents(docs)

vector = FAISS.from_documents(documents,embeddings)
# belgeleri embedding vektörlerine çevirerek FAISS vektör veritabanına kaydeder.

from langchain_openai import ChatOpenAI
model = ChatOpenAI()



from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}
    """
)

#<> Modele hangi kısmın bağlam (context) olduğunu belirtmek için kullanılır.

from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(model, prompt)

#dokumanları modele gecirmek için zincir

from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()

retrieval_chain = create_retrieval_chain(retriever, document_chain)
#vektor databasedeki dokumanlarla olusturdugum chaini birleştirme


response = retrieval_chain.invoke(
    {
        "input":"How to use Dyson Airstrait more efficiently",
    }
)
print(response["answer"])

#cikti:
# To use the Dyson Airstrait more efficiently, you should consider the following tips provided by Dyson:
#1. Make small, slow passes over each section of hair to get the best results.
#2. Ensure to dry the roots of your hair first before continuing with styling passes.
#3. Start with wet hair in the wet-to-dry mode for initial styling, then switch to dry mode for a smooth finish.
#4. Experiment with the different heat and airflow settings to find the best combination for your hair type.
#5. Use any styling products such as creams, gels, or oils, as the tool is self-cleaning and the diffusers can be removed for a deeper clean.
