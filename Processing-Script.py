import json
from collections import defaultdict

dataset_name = 'Beauty1'
input_path = '/kaggle/input/amazondataset/All_Beauty.jsonl'
output_path = f'{dataset_name}.txt'

countU = defaultdict(int)
countP = defaultdict(int)
User = defaultdict(list)
usermap = {}
itemmap = {}
usernum = 0
itemnum = 0

with open(input_path, 'r') as f:
    for line in f:
        l = json.loads(line.strip())
        countU[l['user_id']] += 1
        countP[l['parent_asin']] += 1

with open(input_path, 'r') as f:
    for line in f:
        l = json.loads(line.strip())
        rev, asin, time = l['user_id'], l['parent_asin'], l['timestamp']
        
        if countU[rev] < 5 or countP[asin] < 5:
            continue
        
        if rev not in usermap:
            usernum += 1
            usermap[rev] = usernum
        
        if asin not in itemmap:
            itemnum += 1
            itemmap[asin] = itemnum
        
        User[usermap[rev]].append((time, itemmap[asin]))

for userid in User:
    User[userid].sort()

with open(output_path, 'w') as f:
    for userid, interactions in User.items():
        for _, itemid in interactions:
            f.write(f'{userid} {itemid}\n')

print(f"Processed {usernum} users and {itemnum} items. Data saved to {output_path}.")
