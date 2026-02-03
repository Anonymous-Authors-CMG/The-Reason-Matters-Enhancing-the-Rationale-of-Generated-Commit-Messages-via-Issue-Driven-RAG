import time, os, re, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.TransR import *
from joblib import Parallel, delayed
from tqdm import tqdm

feature_path = "../saved/features"
input_path = "../data/input"
model_base_path = "../saved/models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_issue_diff_feature(project_name, suffix="_TRAIN"):
    # 先加载对应的 BTLink 模型
    model_path = os.path.join(model_base_path, f'{project_name.lower()}_checkpoint-best', "model.bin")
    model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    model.eval()
    print(f"loaded {project_name.lower()} model from {model_path}...")
    data = pd.read_csv(os.path.join(input_path, f"{project_name + suffix}.csv"),
                       usecols=["label", "Issue_KEY", "Issue_Text", "Commit_Code"])
    data["Issue_Text"] = data["Issue_Text"].fillna("")
    data["Commit_Code"] = data["Commit_Code"].fillna("")

    df = pd.read_csv(os.path.join(feature_path, suffix[1:].lower(), f"{project_name.lower()}.csv"))
    for i, row in data.iterrows():
        fv1 = extract_feature(model, row["Issue_Text"], row["Commit_Code"], cpu=True)
        df.loc[i]['fv1'] = ",".join([str(x) for x in fv1[0]])
    df.to_csv(os.path.join(feature_path, suffix[1:].lower(), f"{project_name.lower()}.csv"), index=False)
    print(f"Already convert features for project {project_name}...")


def convert_ckg_feature(project_name, suffix="_TRAIN", n_jobs=4, project_type="mq"):
    print(f"Converting data for project {project_name}...")
    time1 = time.time()

    data = pd.read_csv(os.path.join(input_path, f"{project_name + suffix}.csv"),
                       usecols=["label", "Issue_KEY", "Issue_Text", "Commit_Code"],
                       dtype={"label": int, "Issue_KEY": str, "Issue_Text": str, "Commit_Code": str})
    df = pd.read_csv(os.path.join(feature_path, suffix[1:].lower(), f"{project_name.lower()}.csv"))
    print(f"Read data from {os.path.join(feature_path, suffix[1:].lower(), project_name.lower())}.csv")


    def process_row(i, row):
        transr_utils = TransRUtils()
        if project_type == 'dl':
            files = re.findall(r'\w+\.py', str(row["Commit_Code"]) if row["Commit_Code"] else "")
            files = list(set(f.replace(".py", "") for f in files))
        else:
            files = re.findall(r'\w+\.java', str(row["Commit_Code"]) if row["Commit_Code"] else "")
            files = list(set(f.replace(".java", "") for f in files))

        files = [f for f in files if f]
        if not files:
            return i, ",".join(map(str, np.zeros(2 * transr_embedding_dim + 30).tolist()))

        fv2 = []
        for file in files:
            res = transr_utils.get_class_embedding(project_name, file)
            fv2.append(res)

        fv2 = TransRUtils.polling_embeddings(torch.stack(fv2), 'attention', True)
        return i, ",".join(map(str, fv2.tolist()))

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(i, row)
        for i, row in tqdm(data.iterrows(), total=len(data), desc="Processing")
    )

    fv2_values = {i: fv2 for i, fv2 in results}
    df["fv2"] = df.index.map(fv2_values)
    df.to_csv(os.path.join(feature_path, suffix[1:].lower(), f"{project_name.lower()}.csv"), index=False)

    print(f"Already convert features for project {project_name}...")
    print(f"Saved data at {os.path.join(feature_path, suffix[1:].lower(), project_name.lower())}.csv")
    print(f"Time cost: {time.time() - time1:.2f}s")

if __name__ == '__main__':
    suffixs = ["_TRAIN", "_DEV", "_TEST"]
    projects = ['pytorch', 'tensorflow', 'keras']
    for project in projects:
        for suffix in suffixs: 
            convert_ckg_feature(project, suffix, n_jobs=32, project_type='dl')