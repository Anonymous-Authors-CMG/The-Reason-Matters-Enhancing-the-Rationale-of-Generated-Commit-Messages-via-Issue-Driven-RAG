import json
from collections import Counter

# 从配置文件导入
from config import COMMIT_DETAIL_FILE, REMOVE_BOT_FILE, BOT_EXCLUSION_LIST

input_file_path = COMMIT_DETAIL_FILE
output_file_path = REMOVE_BOT_FILE

# Load the data from the JSON file
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Step 1: Count occurrences of each author_name containing 'bot'
bot_author_names = [item['author_name'] for item in data if 'bot' in item['author_name'].lower()]

# Count occurrences of each author_name
author_name_counts = Counter(bot_author_names)

# Step 2: Print the count for each author_name
print("Count of each author_name containing 'bot':")
for author_name, count in author_name_counts.items():
    print(f"{author_name}: {count}")

# Step 3: Filter out items with author_name in the exclusion list
data = [item for item in data if item['author_name'] not in BOT_EXCLUSION_LIST]

# Step 4: Filter out all items where author_name contains 'bot'
filtered_data = [item for item in data if 'bot' not in item['author_name'].lower()]

# Step 5: Save the filtered data to a new file
with open(output_file_path, 'w', encoding='utf-8', errors='ignore') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

print(f"Filtered data has been saved to {output_file_path}")