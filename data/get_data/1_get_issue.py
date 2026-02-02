import json
import requests
from tqdm import tqdm
import random

# 从配置文件导入
from config import GITHUB_TOKENS, OWNER, REPO, ISSUES_FILE

# 随机获取一个 Token
def get_random_token():
    return random.choice(GITHUB_TOKENS)

# 获取所有 issues 并保存到本地
def fetch_and_save_all_issues(owner, repo, output_file):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    params = {"state": "all", "per_page": 100}
    issues = []
    page = 1

    print("Fetching issues from GitHub API...")
    with tqdm(desc="Fetching issues", unit="page") as pbar:
        while True:
            params["page"] = page
            token = get_random_token()  # 随机选择一个 Token
            headers = {"Authorization": f"token {token}"}
            try:
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 401:  # Token 无效时切换
                    print(f"Invalid token: {token}, switching to another token...")
                    continue
                elif response.status_code != 200:
                    print(f"Failed to fetch issues: {response.status_code}, {response.text}")
                    break
                data = response.json()
                if not data:  # 没有更多数据
                    break
                issues.extend(data)
                page += 1
                pbar.update(1)
            except requests.exceptions.RequestException as e:
                print(f"Error occurred: {e}, retrying with another token...")
                continue

    # 保存到 JSON 文件
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(issues, file, indent=4, ensure_ascii=False)
    print(f"Issues data saved to {output_file}")

# 主函数
if __name__ == "__main__":
    fetch_and_save_all_issues(OWNER, REPO, ISSUES_FILE)