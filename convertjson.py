import pandas as pd
import json

# Load Excel
df = pd.read_excel("data/Harshita tours and travels.xlsx")

flows = {}

for _, row in df.iterrows():
    flow_id = row["Flow_ID"]

    step = {
        "stage": row["Stage"],
        "current_intent": row["CurrentIntent"],
        "bot_question": row["BotQuestion"],
        "user_utterance": row["UserUtterance"],
        "entity": row["Entity"],
        "next_intent": row["NextIntent"]
    }

    if flow_id not in flows:
        flows[flow_id] = {
            "flow_id": flow_id,
            "parent_intent": row["ParentIntent"],
            "steps": []
        }

    flows[flow_id]["steps"].append(step)

# Convert dict → list
output = list(flows.values())

# Save JSON
with open("data/dataset.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("JSON file created: data/dataset.json")
