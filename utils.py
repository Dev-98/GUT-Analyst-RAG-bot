import os
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

HF_EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
genai.configure(api_key=os.environ.get("GENAI_API_KEY"))


def get_context_new(input_query,company_name,k=5):
    
    pine = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pine.Index(os.environ.get("PINECONE_INDEX"))

    input_embed = HF_EMBEDDINGS.embed_query(input_query)

    pinecone_resp = index.query(vector=input_embed, top_k=k, include_metadata=True,
                                filter={"company": company_name,
                                })
    
    if not pinecone_resp['matches']:
        # print(pinecone_resp)
        return "No context Found"

    context = ""
    for i in range(len(pinecone_resp['matches'])):

        score = pinecone_resp['matches'][i]["score"] 
        if score >= 0.53:
            context += "".join(pinecone_resp['matches'][i]['metadata']['text'])
        
    if context == "":
        context = f"No context Found, answer it yourself "
    
    return context


def get_gemini_response(context:str,query:str) -> str :
    input = f"""Use the given context as a refrence to answer the query related to company sales, profit, and business. If there is no context then answer query as an expert on your own trained data.
          context:{context}
        
          This is the query you have to answer
          query:{query}

        """
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content(input)
    return response.text


if __name__ == "__main__":
    # contex = "And we have outstanding capabilities in all three arenas. And really, just don't know how anyone could do what we're doing, even if they had our software and had our computer, if they did not have the the training data. So, speaking of which, our Dojo training computer is designed to significantly reduce the cost of neural net training. It is designed to -- it's somewhat optimized for the kind of training that we need, which is a video training. So, you know, we just see that the need for neural net training, talking about quasi-infinite things is -- is just enormous. So, I think having -- having -- we expect to use both Nvidia and Dojo to be clear. But this -- there's -- we just see a demand for really vast training resources. And we think we may reach in-house neural net -- neural net training capability of 100 exo FLOPs by the end of next year. So, to date, over 300 million miles have been driven using FSD Beta. That 300-million-mile number is going to seem very small, very quickly. Will very -- it'll soon be billions of miles and tens of billions of miles. And FSD will go from being -- from being as good as a human to then being vastly better than a human. We see a clear path to full self-driving being 10 times safer than the average human driver. So -- and between Autopilot, Dojo computer, our inference hardware in the car, which we call sort of hardware 3, 4, you know, but it's really dedicated. It's a -- it's a high-efficiency inference computer that's in the car. And our Optimus robot, Tesla is clearly at the cutting edge of AI with that robot. With regardsubstitute for massive amount of data. And obviously, Tesla has more vehicles on the road. Then, collecting this data, then all of the companies combined. I think maybe even an order of magnitude. So, I think we might -- we might have 90% of all or a very big number. So, you know, the success in AI endeavors is a function of talent, you -- sort of unique data and computing resources. And we have outstanding capabilities in all three arenas. And really, just don't know how anyone could do what we're doing, even if they had our software and had our computer, if they did not have the the training data. So, speaking of which, our Dojo training computer is designed to significantly reduce the cost of neural net training. It is designed to -- it's somewhat optimized for the kind of training that we need, which is a video training. So, you know, we just see that the need for neural net training, talking about quasi-infinite things is -- is just enormous. So, I think having -- having -- we expect to use both Nvidia and Dojo to be clear. But this -- there's -- we just see a demand for really vast training resources. And we think we may reach in-house neural net -- neural net training capability of 100 exo FLOPs by the end of next year. So, to date, over 300 million miles have been driven using FSD Beta. That 300-million-mile number is going to seem very small, very quickly. Will very -- it'll soon be billions of miles and tens of billions of miles. And FSD will go from being -- from being as good as a human to then being vastly better than a human. We see a clear path to full self-driving being 10 times safer than the average human driver. So -- and between Autopilot, Dojo computer, our inference hardware in the car, which we call sort of hardware 3, 4, you know, but it's really dedicated. It's a -- it's a high-efficiency inference computer that's in the car. And our Optimus robot, Tesla is clearly at the cutting edge of AI with that robot. With regardsubstitute for massive amount of data. And obviously, Tesla has more vehicles on the road. Then, collecting this data, then all of the companies combined. I think maybe even an order of magnitude. So, I think we might -- we might have 90% of all or a very big number. So, you know, the success in AI endeavors is a function of talent, you -- sort of unique data and computing resources. And we have outstanding capabilities in all three arenas. And really, just don't know how anyone could do what we're doing, even if they had our software and had our computer, if they did not have the the training data. So, speaking of which, our Dojo training computer is designed to significantly reduce the cost of neural net training. It is designed to -- it's somewhat optimized for the kind of training that we need, which is a video training. So, you know, we just see that the need for neural net training, talking about quasi-infinite things is -- is just enormous. So, I think having -- having -- we expect to use both Nvidia and Dojo to be clear. But this -- there's -- we just see a demand for really vast training resources. And we think we may reach in-house neural net -- neural net training capability of 100 exo FLOPs by the end of next year. So, to date, over 300 million miles have been driven using FSD Beta. That 300-million-mile number is going to seem very small, very quickly. Will very -- it'll soon be billions of miles and tens of billions of miles. And FSD will go from being -- from being as good as a human to then being vastly better than a human. We see a clear path to full self-driving being 10 times safer than the average human driver. So -- and between Autopilot, Dojo computer, our inference hardware in the car, which we call sort of hardware 3, 4, you know, but it's really dedicated. It's a -- it's a high-efficiency inference computer that's in the car. And our Optimus robot, Tesla is clearly at the cutting edge of AI with that robot. "
    context = ""
    prompt = "What is the total revenue for Google Search?"
    if "google" in prompt.lower():
        context += get_context_new(prompt, 'google')
    if "uber" in prompt.lower():
        context += get_context_new(prompt, 'uber')
    if "tesla" in prompt.lower():
        context += get_context_new(prompt, 'tesla')
    
    print(context)
