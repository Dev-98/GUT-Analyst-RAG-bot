from langchain.embeddings import HuggingFaceEmbeddings
import random, string, PyPDF2, os
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 3000
HF_EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
pine = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pine.Index(os.environ.get("PINECONE_INDEX"))

def upsert_data(company,text):
    ''' Funtiom to upsert data into Pinecone with company name and text as metadata :
        @param company = name of the company text belongs to
        @param text = text extracted from the company pdf files
        
        @ returns None
    '''

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=50)
    text_chunks = text_splitter.split_text(text)

    for chunk in text_chunks:
        
        metadata = {"company": company, "text":chunk}
        # Text to be embedded
        vector = HF_EMBEDDINGS.embed_query(chunk)

        # Ids generation for vectors
        _id = ''.join(random.choices(string.ascii_letters + string.digits, k=10,))

        # Upserting vector into pinecone database
        index.upsert(vectors=[{"id":_id, "values": vector, "metadata": metadata}])

        print("Vector upserted successfully")


def process_and_upsert_pdf(pdf_path,company_name,chunk_size=CHUNK_SIZE,k=5):
    """ Process a pdf file and check if its content is in pincone DB, if not then upsert it:
        Inputs :- 
         @pfd_path = path to pdf file
         @company_name = name of the company pdf belongs to
         @chunk_size = then number of bytes of chunk to write into DB
         @k = it is the number of top results like : top 5, or top 3

        @returns a message indicating the status of the precess happened
    """
    with open(pdf_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text()

    if text == "":
        return "PDF is unreadable and cannot be extracted"

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    text_chunks = text_splitter.split_text(text)
    chunk = text_chunks[0]
    vector = HF_EMBEDDINGS.embed_query(chunk)

    # Query the DB with pdf text encoding
    pinecone_resp = index.query(
        vector=vector, 
        top_k=k, 
        include_metadata=True,
        namespace=company_name,
        filter={"company": company_name}
    )

    # If there os no matching data, then upsert it into the pinecone
    if not pinecone_resp['matches']:
        upsert_data(company_name,text)
        print("No matches found")
        return "PDF upserted successfully into Pinecone database, under {} company".format(company_name)

    score = pinecone_resp['matches'][0]["score"] 
    print("Score of last index :",score) 
    
    # If there is a match then check for score. and if it is lower than 85% then upsert it 
    if score <= 0.85 :
        upsert_data(company_name, text)
        return "PDF upserted successfully into Pinecone database, under {} company".format(company_name)  


if __name__ == '__main__':
    pdf_path = 'CNN_paper.pdf'

    # a = process_and_upsert_pdf(company_name = "TESLA",year = 2023,quarter= "Q4",pdf_path=pdf_path,vertical="research",category="stocks")
    # print(a)
    upsert_data("test",""" Google Workspace and Duet AI in Google Workspace provide easy-to-use, secure communication and collaboration tools, including apps like Gmail, Docs, Drive, Calendar, Meet, and more. These tools enable secure hybrid and remote work, boosting productivity and collaboration. AI has been used in Google Workspace for years to improve grammar, efficiency, security, and more with features like Smart Reply, Smart Compose, and malware and phishing protection in Gmail. Duet AI in Google Workspace helps users write, organize, visualize, accelerate workflows, and have richer meetings. 
•AI Platform and Duet AI for Google Cloud:  Our Vertex AI platform gives developers the ability to train, tune, augment, and deploy applications using generative AI models and services such as Enterprise Search and Conversations. Duet AI for Google Cloud provides pre-packaged AI agents that assist developers to write, test, document, and operate software. Other Bets Across Alphabet, we are also using technology to try to solve big problems that affect a wide variety of industries from improving transportation and health technology to exploring solutions to address climate change. Alphabet’s investment in the portfolio of Other Bets includes businesses that are at various stages of development, ranging from those in the R&D phase to those that are in the beginning stages of commercialization. Our goal is for them to become thriving, successful businesses.""")
