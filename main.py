import os
import uuid
import random
import torch

from PIL import Image
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer(
    'jinaai/jina-clip-v2',
    trust_remote_code=True,
    truncate_dim=1024,
    device = device
)

if not os.path.exists('image-store'):
    client = QdrantClient(path = 'image_store')
    images = [os.path.join('images',f) for f in os.listdir('images')]
    print(f"Processing {len(images)} images for embeddings...")
    embeddings = [model.encode(img, normalize_embeddings=True, device=device) for img in images]
    
    client.recreate_collection(
        collection_name='images',
        vectors_config=VectorParams(size=len(embeddings[0])),
        Distance=Distance.COSINE
    )
    
    client.upsert(
        collection_name = 'images',
        points=[
            PointStruct(id=uuid.uuid4(),vector=embeddings[i],payload={'path':images[i]})
            for i in range(len(images))
        ]
    )

else:
    client = QdrantClient(path = 'image_store')


print('Done!')

search_query = input('Enter query:')
query_embedding = model.encode(search_query, normalize_embeddings=True, device=device)
results = client.search(collection_name='images', query_vector=query_embedding, limit=5).points


print('Results:')
for result in results:
    print(f"Path: {result.payload['path']}, Score: {result.score}")
    img = Image.open(result.payload['path'])
    img.show()
    
# print([r.payload['path'] for r in results])  
#are both the same thing?
