from py2neo import Graph
from torch.utils.data import Dataset
import pandas as pd
import re, os

uri = "bolt://localhost:7687"
user = "neo4j"
password = "xxx"
SAVE_DIR = "../saved/data"
MODEL_DIR = "../saved/models"

class Triple:
    # 按照某项目中的类名获取关系
    @staticmethod
    def get_relation_by_class(project_name, entity, log=True):
        graph = Graph(uri, auth=(user, password))
        query = f"MATCH (n:{project_name}Class {{name: '{entity}'}})-[r]-(m) RETURN n, r, m"
        if log:
            print("Executing: " + query + "; ...")
        result = graph.run(query)
        ls = []
        for item in result:
            ls.append([item["n"]["name"], parse_relation(str(item["r"])), item["m"]["name"]])
        return pd.DataFrame(ls, columns=["head", "relation", "tail"])

    # 获取某项目的全部类实体
    @staticmethod
    def get_all_nodes(project_name):
        graph = Graph(uri, auth=(user, password))
        query = f"MATCH (n:{project_name}Class) RETURN n"
        result = graph.run(query)
        ls = []
        for item in result:
            ls.append(item["n"]["name"])
        return ls

    # 获取所有关系
    @staticmethod
    def get_all_relations():
        if os.path.exists(f"{SAVE_DIR}/relations.txt"):
            with open(f"{SAVE_DIR}/relations.txt", "r") as f:
                return f.read().split(",")
        graph = Graph(uri, auth=(user, password))
        query = f"MATCH ()-[r]-() RETURN r"
        result = graph.run(query)
        ls = set()
        for item in result:
            ls.add(parse_relation(str(item["r"])))
        ls = list(ls)
        save_str = ",".join(ls)
        with open(f"{SAVE_DIR}/relations.txt", "w") as f:
            f.write(save_str)
        return ls

    # 获取某项目的全部三元组
    @staticmethod
    def get_all_triplets(project_name, entity_name=['Class', 'Method'], limited=None):
        if os.path.exists(f"{SAVE_DIR}/{project_name}_triplets.csv"):
            return pd.read_csv(f"{SAVE_DIR}/{project_name}_triplets.csv")
        graph = Graph(uri, auth=(user, password))
        sub_str = ""
        for i in range(len(entity_name)):
            sub_str += f"(n:{project_name}{entity_name[i]}) "
            if i != len(entity_name) - 1:
                sub_str += 'OR '
        query = f"MATCH (n) WHERE {sub_str} WITH n MATCH (n)-[r]-(m) RETURN n, r, m"
        if limited is not None:
            query += f" LIMIT {limited}"
        print("Executing: " + query + "; ...")
        result = graph.run(query)
        ls = []
        for item in result:
            ls.append([item["n"]["name"], parse_relation(str(item["r"])), item["m"]["name"], 1])
        df = pd.DataFrame(ls, columns=["head", "relation", "tail", "label"])
        df.to_csv(f"{SAVE_DIR}/{project_name}_triplets.csv", index=False)
        return df

class TripletDataset(Dataset):
    def __init__(self, triplets, entity_to_id, relation_to_id):
        self.triplets = triplets
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        head, relation, tail, label = self.triplets[idx]
        return (self.entity_to_id[head], self.relation_to_id[relation], self.entity_to_id[tail], label)


def parse_relation(relation):
    match_result = re.search(r'\[(.*?)\]', relation)
    if match_result:
        content_inside_brackets = match_result.group(1)
        cleaned_content = re.sub(r'[^\w]', '', content_inside_brackets)
        return cleaned_content
    return None

def get_project_triplets(project_type='dl'):
    if project_type == "dl":
        pytorch_triples = Triple.get_all_triplets("pytorch", entity_name=['Class'])
        tensorflow_triples = Triple.get_all_triplets("tensorflow", entity_name=['Class'])
        keras_triples = Triple.get_all_triplets("keras", entity_name=['Class'])
        triplets = pd.concat([pytorch_triples, tensorflow_triples, keras_triples])
        return triplets.drop_duplicates()
    elif project_type == "mq":
        kafka_triples = Triple.get_all_triplets("kafka", entity_name=['Class'])
        rocketmq_triples = Triple.get_all_triplets("rocketmq", entity_name=['Class'])
        pulsar_triples = Triple.get_all_triplets("pulsar", entity_name=['Class'])
        triplets = pd.concat([kafka_triples, rocketmq_triples, pulsar_triples])
        return triplets.drop_duplicates()
    return None

if __name__ == '__main__':
    # triple = Triple.get_relation_by_class("oodt", "DirListNonRecursiveHandler")
    # print(triple)
    df = Triple.get_all_triplets("pytorch", entity_name=['Class'])
    print(df.head())