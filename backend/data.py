import json

def load_known_cats():
    # 加载已注册猫咪数据（可以是手动更新的JSON文件）
    with open('database/cats_db.json', 'r') as f:
        cats_data = json.load(f)

    cats = []
    for cat in cats_data:
        cats.append(Cat(cat['name'], cat['id']))
    return cats

class Cat:
    def __init__(self, name, cat_id):
        self.name = name
        self.cat_id = cat_id
