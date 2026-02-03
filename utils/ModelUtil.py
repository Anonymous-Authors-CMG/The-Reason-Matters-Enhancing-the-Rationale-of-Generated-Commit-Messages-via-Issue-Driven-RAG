import os, re, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModel, RobertaConfig
from transformers import RobertaTokenizer, RobertaModel
from model.TransR import *
from model.MLP import *
from model.lstm.Bi_LSTM import *
from model.BertModel.model import BTModel
from VectorDatabase import get_near_records
from model.encoder.CodeIssueEncoder import CodeIssueEncoder
from multiprocessing import Pool, cpu_count
import faiss
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelUtil:
    def __init__(self, project_name):
        self.text_model_path = config.text_model_path
        self.code_model_path = config.code_model_path
        self.model_base_path = config.model_base_path
        self.project_name = project_name
        self.project_type = config.project_type_mapping[project_name]
        
        self._init_tokenizers()
        self._init_encoders()
        self._init_vector_database()
        self._init_coarse_filter()
        
        self.transR_util = TransRUtils()
        self.link_model = self.load_link_model(project_name)
        self.mlp_model = self.load_mlp_model(project_name)
        self.commit_classifier_model = self.load_lstm_model(config.commit_classifier_model_path)
        
        
    def _init_tokenizers(self):
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_path, local_files_only=True)
        self.code_tokenizer = AutoTokenizer.from_pretrained(self.code_model_path, local_files_only=True)
        self.commit_classifier_tokenizer = BertTokenizer.from_pretrained(config.commit_classifier_tokenizer)
    
    def _init_encoders(self):
        config = RobertaConfig.from_pretrained(self.text_model_path, local_files_only=True)
        config.num_labels = 2
        self.textEncoder = AutoModel.from_pretrained(
            self.text_model_path,
            config=config,
            local_files_only=True
        )
        
        config4Code = RobertaConfig.from_pretrained(self.code_model_path, local_files_only=True)
        config4Code.num_labels = 2
        self.codeEncoder = AutoModel.from_pretrained(
            self.code_model_path,
            config=config4Code,
            local_files_only=True
        )
        self.text_hidden_size = config.hidden_size
        self.code_hidden_size = config4Code.hidden_size
        
    def _init_vector_database(self):
        with open(f"{config.vector_database_path}dl_meta.json", 'r', encoding='utf-8') as f:
            self.dl_records = json.load(f)
        with open(f"{config.vector_database_path}mq_meta.json", 'r', encoding='utf-8') as f:
            self.mq_records = json.load(f)
        self.dl_index = faiss.read_index(f"{config.vector_database_path}dl.index")
        self.mq_index = faiss.read_index(f"{config.vector_database_path}mq.index")
    
    def _init_coarse_filter(self):
        self.coarse_tokenizer = RobertaTokenizer.from_pretrained(config.code_model_path)
        self.coarse_model = RobertaModel.from_pretrained(config.code_model_path)
        self.fine_tune_encoder = CodeIssueEncoder(base_model=self.code_model_path)

    def _encode_inputs(self, issue_text, commit_code):
        text_tokens = self.text_tokenizer(issue_text, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        code_tokens = self.code_tokenizer(commit_code, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        return text_tokens["input_ids"].to(device), code_tokens["input_ids"].to(device)

    def load_link_model(self, project_name):
        model_path = os.path.join(self.model_base_path, f'{project_name.lower()}_checkpoint-best', "model.pth")
        # 加载模型参数
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_k] = v
        link_model = BTModel(self.textEncoder, self.codeEncoder,
                     self.text_hidden_size, self.code_hidden_size, num_class=2).to(device)
        link_model.load_state_dict(new_state_dict)
        link_model.to(device)
        link_model.eval()

        print(f"loaded {project_name.lower()} model from {model_path}...")
        return link_model
    
    @staticmethod
    def load_mlp_model(project_name):
        model_path = os.path.join(config.model_base_path, 'mlps', f'{project_name.lower()}_mlp.pth')
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = checkpoint["model"].to(device)
        model.eval()
        print(f"loaded {project_name.lower()} MLP model from {model_path}...")
        return model
    
    @staticmethod
    def load_lstm_model(checkpoint_path):
        print(f"loaded commit classifier model from {checkpoint_path}...")
        return CommitClassifierLightning.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=CONFIG
        )
    
    def extract_link_feature(self, issue_text, commit_code, cpu=False):
        text_input_ids, code_input_ids = self._encode_inputs(issue_text, commit_code)
        model = self.link_model
        with torch.no_grad():
            text_output = model.textEncoder(text_input_ids, attention_mask=text_input_ids.ne(1))[1]
            code_output = model.codeEncoder(code_input_ids, attention_mask=code_input_ids.ne(1))[1]
            feature_vector = torch.cat([text_output, code_output], dim=-1)
        return feature_vector.cpu().numpy() if cpu else feature_vector
    
    def extract_ckg_feature(self, code_diff, project_name, pooling_method='mean', cpu=False):
        if self.project_type == 'dl':
            files = re.findall(r'\w+\.py', str(code_diff) if code_diff else "")
            files = list(set(f.replace(".py", "") for f in files))
        else:
            files = re.findall(r'\w+\.java', str(code_diff) if code_diff else "")
            files = list(set(f.replace(".java", "") for f in files))
        files = [f for f in files if f]
        if not files:
            fv = ",".join(map(str, self.transR_util.embedding_dim * [0]))
            return fv if cpu else torch.zeros((1, self.transR_util.embedding_dim))
        fv = [self.transR_util.get_class_embedding(project_name, file, pooling_method=pooling_method) for file in files]
        return self.transR_util.polling_embeddings(torch.stack(fv), pooling_method, cpu)
    
    def compute_link_score(self, issue_text, commit_code):
        fv1 = self.extract_link_feature(issue_text, commit_code).to(device)
        fv2 = self.extract_ckg_feature(commit_code, self.project_name,'attention').to(device)
        fv1 = fv1.unsqueeze(0) if fv1.dim() == 1 else fv1
        fv2 = fv2.unsqueeze(0) if fv2.dim() == 1 else fv2
        with torch.no_grad():
            logits = self.mlp_model(fv1, fv2)
            prob = torch.sigmoid(logits).item()
        return prob
    
    def classify_commit(self, commit_message):
        encoded = self.commit_classifier_tokenizer.encode_plus(
            commit_message,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            logits = self.commit_classifier_model(
                encoded['input_ids'],
                encoded['attention_mask'],
                encoded['token_type_ids']
            )
            return torch.argmax(logits, dim=1).item()
        
    def find_high_quality_sim_records(self, query_vector, index_name):
        params = config.high_quality_params
        records = self.dl_records if index_name == 'dl' else self.mq_records
        index = self.dl_index if index_name == 'dl' else self.mq_index
        D, I = get_near_records(query_vector, index_name, top_k=params["top_k"], meta_data=records, index=index)
        # 过滤相似度和低质量数据
        filtered_pairs = [(d, i) for d, i in zip(D, I) if d >= params["similarity_threshold"]]
        filtered_pairs = [(d, i) for d, i in zip(D, I) if self.classify_commit(i['message']) >= params["label_level"]]
        if filtered_pairs:
            D, I = zip(*filtered_pairs)
            D, I = list(D), list(I)
            D = [float(x) for x in D]
            messages = [i['message'] for i in I]
        else:
            D, I, messages = [], [], []
        return D, messages
    
    def get_issue_set(self):
        issue_set_path = os.path.join(config.issue_path, 'issue_set', f'{self.project_name}_issues.json')
        with open(issue_set_path, 'r', encoding='utf-8') as f:
            issues = json.load(f)
        return issues

    def _encode_coarse_inputs(self, text, fine_tune=False):
        if fine_tune:
            return self.fine_tune_encoder.encode([text], convert_to_tensor=True)
        inputs = self.coarse_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.coarse_model(**inputs)
        return outputs.last_hidden_state[:,0,:]  # CLS向量
    
    def find_possible_issues_coarse(self, code_diff, result_num=None, fine_tune=False):
        issue_set_path = os.path.join(config.issue_path, 'issue_set', f'{self.project_name}_issues.json')
        params = config.possible_issue_params
        result_num = result_num if result_num is not None else params['max_candidates']
        with open(issue_set_path, 'r', encoding='utf-8') as f:
            issues = json.load(f)
            code_vec = self._encode_coarse_inputs(code_diff, fine_tune=fine_tune).to(config.device)
            key = 'encoded' if not fine_tune else 'encoded_v2'
            candidates = []
            for issue in issues:
                issue_vec = torch.tensor(issue[key]).squeeze(0).to(config.device)
                title = issue.get('title') or ''
                body = issue.get('body') or ''
                if len(title) <= 3 and len(body) <= 3:
                    continue
                issue_text = (title + '\n' + body).strip()
                similarity = torch.nn.functional.cosine_similarity(issue_vec, code_vec)
                prob = torch.sigmoid(similarity)
                if prob.item() > params['coarse_threshold']:
                    candidates.append(
                        {
                            'id': issue['id'],
                            'issue': issue_text,
                            'similarity': prob.item()
                        }
                    )
                candidates = sorted(candidates, key=lambda x: x['similarity'], reverse=True)
            return candidates[:result_num]
    
    def find_possible_issues(self, code_diff, result_num=None, fine_tune=False):
        result_num = result_num if result_num is not None else config.possible_issue_params['max_result']
        candidates = self.find_possible_issues_coarse(code_diff, fine_tune=fine_tune)
        for c in candidates:
            c['link_score'] = self.compute_link_score(c['issue'], code_diff)
        candidates = sorted(candidates, key=lambda x: x['link_score'], reverse=True)
        return candidates[:result_num]
