"""
数据获取主脚本
===============
使用方法：
1. 先修改 config.py 中的配置参数
2. 运行此脚本: python run_all.py

此脚本会按顺序执行以下步骤：
1. 获取 Issues 数据
2. 获取 Issue-Commit 映射关系  
3.0 获取 Commit 详细信息
3.1 移除 Bot 提交
3.2 过滤超过 4096 token 的 diff
4. 合并 Issue 和 Commit 数据
"""

import subprocess
import sys
import os

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 定义要按顺序执行的脚本
SCRIPTS = [
    ("1_get_issue.py", "步骤 1: 获取 Issues 数据"),
    ("2_get_issue-commit_link.py", "步骤 2: 获取 Issue-Commit 映射关系"),
    ("3.0_get_commit_detail.py", "步骤 3.0: 获取 Commit 详细信息"),
    ("3.1_remove_bot.py", "步骤 3.1: 移除 Bot 提交"),
    ("3.2_remove_4096_token_diff.py", "步骤 3.2: 过滤超长 diff"),
    ("4_merge_issue_and_commit.py", "步骤 4: 合并 Issue 和 Commit 数据"),
]


def run_script(script_name, description):
    """运行单个脚本"""
    print("\n" + "=" * 60)
    print(f"🚀 {description}")
    print(f"   执行脚本: {script_name}")
    print("=" * 60 + "\n")
    
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    if result.returncode != 0:
        print(f"\n❌ 脚本 {script_name} 执行失败，返回码: {result.returncode}")
        return False
    
    print(f"\n✅ {description} 完成")
    return True


def main():
    print("\n" + "=" * 60)
    print("       数据获取流水线")
    print("=" * 60)
    print("\n请确保已在 config.py 中正确配置以下参数：")
    print("  - GITHUB_TOKENS: GitHub API tokens")
    print("  - OWNER: 仓库所有者")
    print("  - REPO: 仓库名")
    print("  - GIT_REPO_PATH: 本地 Git 仓库路径")
    print("\n" + "-" * 60)
    
    # 导入配置以显示当前设置
    try:
        import config
        print(f"\n当前配置:")
        print(f"  OWNER: {config.OWNER}")
        print(f"  REPO: {config.REPO}")
        print(f"  GIT_REPO_PATH: {config.GIT_REPO_PATH}")
        print(f"  GITHUB_TOKENS: {len(config.GITHUB_TOKENS)} 个 token")
    except Exception as e:
        print(f"❌ 错误: 无法加载配置文件: {e}")
        return
    
    print("\n" + "-" * 60)
    
    # 按顺序执行所有脚本
    for script_name, description in SCRIPTS:
        if not os.path.exists(script_name):
            print(f"\n⚠️ 警告: 脚本 {script_name} 不存在，跳过...")
            continue
        
        success = run_script(script_name, description)
        if not success:
            return
    
    print("\n" + "=" * 60)
    print("🎉 所有步骤执行完成！")
    print("=" * 60)
    print(f"\n最终输出文件: {config.FINAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()

