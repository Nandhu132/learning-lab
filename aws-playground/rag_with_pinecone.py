import os
from dotenv import load_dotenv
import boto3
import uuid
from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
)

embeddings = BedrockEmbeddings(
    client=bedrock,
    model_id="amazon.titan-embed-text-v2:0"
)

with open("sample_document.txt", "r") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)


vectors = []
for chunk in chunks:
    emb = embeddings.embed_query(chunk)  
    vectors.append({
        "id": str(uuid.uuid4()),
        "values": emb,
        "metadata": {"text": chunk}
    })


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "bedrock-simple-index"

index = pc.Index(index_name)
index.upsert(vectors=vectors)
# index.delete(delete_all=True)

query = "What is the main idea of the document?"
query_emb = embeddings.embed_query(query)

results = index.query(
    vector=query_emb,
    top_k=1,
    include_metadata=True
)

best_match = results["matches"][0]["metadata"]["text"]
print(best_match)

