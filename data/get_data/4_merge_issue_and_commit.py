import json
from tqdm import tqdm

# 从配置文件导入
from config import ISSUES_FILE, COMMIT_ISSUE_MAP_FILE, FILTERED_DIFF_FILE, FINAL_OUTPUT_FILE

# 加载数据
with open(ISSUES_FILE, 'r') as f:
    issues_data = json.load(f)

with open(COMMIT_ISSUE_MAP_FILE, 'r') as f:
    commit_issue_mapping = json.load(f)

with open(FILTERED_DIFF_FILE, 'r') as f:
    commit_details = json.load(f)

# 创建issue快速查找字典（key: issue number）
issue_dict = {issue["number"]: issue for issue in issues_data}

# 处理数据
result = []

for commit in tqdm(commit_details, desc="Processing commits"):
    commit_id = commit["commit id"]
    author_name = commit["author_name"]
    author_email = commit["author_email"]


    # 获取关联的issue IDs
    issue_ids = commit_issue_mapping.get(commit_id, [])

    # 收集issue详细信息
    related_issues = []
    for issue_id in issue_ids:
        issue = issue_dict.get(issue_id)
        if issue:
            related_issues.append({
                "id": issue_id,
                "title": issue.get("title", ""),
                "body": issue.get("body", "")
            })
        else:
            print(f"Warning: Issue {issue_id} not found in issues data")

    # 构建最终结构
    result.append({
        "commit_id": commit_id,
        "author_name": author_name,
        "author_email": author_email,
        "message": commit["message"],
        "issues": related_issues,
        "diff": commit["diff"]
    })

# 保存结果
with open(FINAL_OUTPUT_FILE, 'w') as f:
    json.dump(result, f, indent=2)

print(f"处理完成！共处理 {len(result)} 个提交")