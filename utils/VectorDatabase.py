import faiss
import json, os
import numpy as np

embedding_path = "../data/issue/issue_embedding_large/"
normalize = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
dl_list = ['pytorch', 'tensorflow', 'keras']
mq_list = ['kafka', 'rocketmq', 'pulsar']

def create_vector_database(project_list, index_name=None):
    records = []
    for project in project_list:
        embedding_file = os.path.join(embedding_path, f"{project}_embedding.json")
        with open(embedding_file, 'r', encoding='utf-8') as f:
            records.extend(json.load(f))
    embeddings = np.array([item["embedding"] for item in records], dtype='float32')
    embeddings = normalize(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    if index_name:
        faiss.write_index(index, f"../saved/vector/{index_name}.index")
        with open(f"../saved/vector/{index_name}_meta.json", 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=4)
        
def get_near_records(query, index_name, top_k=10, meta_data=None, index=None):
    if meta_data is None:
        with open(f"../saved/vector/{index_name}_meta.json", 'r', encoding='utf-8') as f:
            records = json.load(f)
    records = meta_data if meta_data else records
    if index is None:
        index = faiss.read_index(f"../saved/vector/{index_name}.index")
    query = normalize(np.array([query], dtype='float32').reshape(1, -1).astype('float32'))
    D, I = index.search(query, top_k+1)
    return D[0][1:], [records[i] for i in I[0]][1:]

if __name__ == "__main__":
    create_vector_database(dl_list, index_name="dl")
    create_vector_database(mq_list, index_name="mq")
    query_vector = np.random.rand(1, 3072).astype('float32')
    D, near_issues = get_near_records(query_vector, index_name="dl", top_k=5)
    print(D)
    print(near_issues[0])