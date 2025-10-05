import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

faqs_path=Path(__file__).parent/ "resources/faq_data.csv"
chroma_client=chromadb.Client()
ef=embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)
collection_name_faq='faq'
groq_client=Groq()


def ingest_faq_data(path):
    if collection_name_faq not in [c.name for c in chroma_client.list_collections()]:
        print(f'FAQ Data is ingesting into Chroma collection:{collection_name_faq}')
        collection=chroma_client.get_or_create_collection(
        name=collection_name_faq,
        embedding_function=ef
        )
        df=pd.read_csv(path)
        docs=df['question'].tolist()
        metadata= [{'answer': ans} for ans in df['answer'].tolist()]
        ids = [f'id_{i}' for i in range(len(docs))]

        collection.add(
        documents=docs,
        metadatas=metadata,
        ids=ids
        )
        print(f'FAQ Data successfully ingested into Chroma collection:{collection_name_faq}')
    else:
        print(f"Collection {collection_name_faq} already exists")


def get_relevant_qa(query):
    collection=chroma_client.get_collection(name=collection_name_faq)
    result=collection.query(
        query_texts=[query],
        n_results=2
    )
    return result

def faq_chain(query):
    result=get_relevant_qa(query)
    context=''.join([r.get('answer') for r in result['metadatas'][0]])
    answer=generate_answer(query,context)
    return answer

def generate_answer(query,context):
    prompt=f'''Given the question and context below, generate the answer based on the ccontext only.
    If you don't find the answer inside the context then say "I don't know".
    Do not make things up.
    QUESTION:{query}
    CONTEXT:{context}
    '''

    completion = groq_client.chat.completions.create(
        model=os.environ['GROQ_MODEL'],
        messages=[{"role": "user","content":prompt}]
    )
    return completion.choices[0].message.content





if __name__=="__main__":
    ingest_faq_data(faqs_path)
    query="what's your policy on defective products?"
    answer=faq_chain(query)
    print(answer)