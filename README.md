## GUT Analyst: A Research-Oriented Chatbot

**Project Overview** : https://gut-analyst-rag-bot-fpdp6xmtyfawgx2qbqjbw9.streamlit.app/

The GUT Analyst is a chatbot designed to analyze and answer your questions about three industry titans: Google, Tesla, and Uber. It leverages the power of 2023 data, including annual reports and future plans, to provide you with insights, identify trends, and assist with research.

**Usage**

This chatbot is ideal for:

* **Students and Researchers:** Gain quick access to key information about Google, Tesla, and Uber for project work and analysis.
* **Investors:** Stay informed about the latest trends and future plans of these companies to make informed investment decisions.
* **Business Professionals:** Get a comparative perspective on the strategies and performance of these leading corporations.

## How to Setup 

1. **Install Dependencies**: Ensure all dependencies are installed using the provided `requirements.txt`.
2. **Run the Summarization Script**: Extract and summarize text from the desired PDF.
3. **Generate Embeddings**: Convert the summarized text into embeddings.
4. **Upload to Pinecone**: Store the embeddings in a Pinecone index, and all this by running `PDF_process.py`.
5. **Set Up Retrieval System**: Configure Pinecone to enable text retrieval based on queries.
6. **Deploy Chatbot Interface**: Run `streamlit run main.py` to deploy the chatbot interface for user interaction.

## Installation

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Step-by-Step Guide

### 1. Summarizing the PDF

We extract text from the PDF using `PyPDF2` and summarize it using a combination of `sumy`'s LexRank algorithm and `transformers`' T5 model. This involves extracting text from specified pages of the PDF, creating an initial summary with LexRank, and further compressing it using T5 to produce a more concise and meaningful summary.

### 2. Generating Embeddings

Using `sentence-transformers`, we generate vector embeddings for the summarized text. These embeddings represent the text in a numerical format suitable for similarity comparison and retrieval tasks. The `sentence-transformers` library provides pre-trained models for generating high-quality text embeddings.

### 3. Uploading to Pinecone

The generated embeddings are then uploaded to Pinecone, a vector database service. Pinecone enables efficient storage and retrieval of high-dimensional vectors, making it ideal for applications like semantic search and text retrieval. We use Pinecone's client library to create an index, upload the embeddings, and set up the retrieval system.

### 4. Setting Up the Retrieval System

We configure Pinecone to serve as the backend for our text retrieval system. This involves creating an index in Pinecone, uploading the embeddings, and setting up search functionality to retrieve relevant text segments based on query embeddings. The `langchain` library is used to facilitate the integration with Pinecone and manage the retrieval process.

### 5. Creating the Chatbot Interface

Finally, we use Streamlit to build a chatbot interface that interacts with the retrieval system. Streamlit is a powerful library for creating interactive web applications with minimal code. The chatbot interface allows users to input queries, which are processed to retrieve relevant text segments from Pinecone and display them in an intuitive manner.


**Conclusions**

The GUT Analyst demonstrates the potential of chatbots for research and information retrieval. By leveraging data analysis and natural language processing, this project offers a convenient and informative tool for anyone interested in gaining insights into the dynamic world of Google, Tesla, and Uber.

**Future Directions**

Future iterations of the GUT Analyst could include:

* Expanding the data source to include news articles and market data.
* Implementing sentiment analysis to gauge public perception of the companies.
* Integrating with financial analysis tools for more in-depth investment research.

This project serves as a springboard for further development in the realm of data-driven chatbots for business and financial research.

**Try out the chatbot** 
