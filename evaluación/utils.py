import json
import pandas as pd

def get_final_word(sent: str) -> (str):
    words = sent.split()
    last = words[-1]
    if last[-1] == '.':
        return (" " + (last[:-1]))
    else:
        return (" " + (last))

def get_prefix(sent: str) -> str:
    words = sent.split()[:-1]
    return " ".join(words)

def create_dataset():
    with open("QuUANTO_unified_final.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # df = pd.read_json()