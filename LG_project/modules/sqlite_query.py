

import sqlite3

db_path = "/content/drive/MyDrive/LG_project/lg_info.db"  
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

def query_sqlite_db(user_question: str) -> str:
    lowered = user_question.lower()

    if "price" in lowered or "specification" in lowered or "model" in lowered:
        cursor.execute("SELECT name, specifications, price FROM products")
        for name, spec, price in cursor.fetchall():
            if name.lower() in lowered:
                return f"Product: {name}\nSpecs: {spec}\nPrice: ₹{price}"

    return None
