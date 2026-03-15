import json
import math
import os
import re
import sys
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    embedding_model = None

DATASET_FILE = 'data/dataset.json'
STATE_FILE = 'session_state.json'

def _load_env_from_dotenv():
    base = os.path.dirname(__file__)
    dotenv_path = os.path.join(base, ".env")
    if os.path.exists(dotenv_path):
        try:
            with open(dotenv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and not os.getenv(k):
                        os.environ[k] = v
        except Exception:
            pass

_load_env_from_dotenv()


try:
    from groq import Groq
    GROQ_KEY = os.getenv("GROQ_API_KEY")
    groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
except ImportError:
    groq_client = None
except Exception as e:
    groq_client = None
    print("Groq init error:", e, file=sys.stderr)


# ---------- Data Loading ----------
def load_flows():
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_steps(flows):
    steps = []
    for flow in flows:
        for s in flow.get('steps', []):
            steps.append(s)
    return steps


def build_state_index(steps):
    # Group steps by current_intent to discover bot messages for states
    states = {}
    for s in steps:
        cur = s['current_intent']
        states.setdefault(cur, []).append(s)
    return states


# ---------- State Persistence ----------
def default_context():
    return {
        "destination": None,
        "travellers": None,
        "duration": None,
        "hotel_type": None,
        "travel_mode": None,
        "caller_location": None,
        "interest_type": None,
        "preference": None
    }


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            st = json.load(f)
            if 'context' not in st:
                st['context'] = default_context()
            else:
                # ensure all expected keys exist
                for k, v in default_context().items():
                    st['context'].setdefault(k, v)
            return st
    return {"current_state": "Greeting", "context": default_context()}


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


# ---------- Embeddings & Similarity ----------
def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def bow_embedding(text):
    # Simple bag-of-words frequency as fallback embedding
    tokens = tokenize(text)
    return Counter(tokens)


def cosine_similarity(vec1, vec2):
    if isinstance(vec1, Counter) and isinstance(vec2, Counter):
        # BoW cosine
        keys = set(vec1.keys()) | set(vec2.keys())
        dot = sum(vec1[k] * vec2[k] for k in keys)
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    # Numeric vectors
    dot = sum(a * b for a, b in zip(vec1, vec2))
    n1 = math.sqrt(sum(a * a for a in vec1))
    n2 = math.sqrt(sum(b * b for b in vec2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def embed_text(text):
    if embedding_model is not None:
        try:
            return embedding_model.encode(text).tolist()
        except Exception:
            return bow_embedding(text)
    return bow_embedding(text)


# ---------- Entity Extraction ----------
def extract_entity(user_text, entity_label):
    text = user_text.strip()
    if not entity_label:
        return None
    
    label = entity_label.lower()
    
    # Normalization for travellers
    if label in ("members", "travelers", "no_of_members", "number_of_travellers", "number_of_travelers"):
        label = "travellers"
    
    # Try LLM extraction first if Groq is available
    if groq_client:
        try:
            prompt = f"""Extract the requested information from the user's message.
User message: "{text}"
Information to extract: "{entity_label}"

Instructions:
1. Return ONLY the extracted value and absolutely nothing else.
2. If the entity is a number of people like 'two people', 'a couple', 'family of four', return just the digit (e.g., '2', '4').
3. If the entity is a destination like 'a trip to Assam', return just the location (e.g., 'Assam').
4. If you cannot extract a reasonable value, return the exact word "None".
"""
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise data extraction assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.0
            )
            val = chat_completion.choices[0].message.content.strip()
            
            # Clean up the response
            if val == "None" or not val:
                return None
                
            # Formatting rules based on entity label
            if label in ("travellers", "travelers", "members", "no_of_members", "number_of_travellers", "number_of_travelers"):
                m = re.search(r"\b(\d+)\b", val)
                if m:
                    return int(m.group(1))
            
            # If the label is destination or place, try regex for better extraction
            if label in ("destination", "place"):
                # Regex for "to ", "go to ", "visit ", "travel to "
                patterns = [
                    r"(?i)\b(?:go to|travel to|visit|to)\s+([a-zA-Z\s]+)",
                ]
                for p in patterns:
                    match = re.search(p, val)
                    if match:
                        return match.group(1).strip()
            
            return val
        except Exception as e:
            print("Groq extraction error:", e, file=sys.stderr)

    # Fallback to regex if LLM fails or is not available
    if label in ("travellers", "travelers", "members", "no_of_members", "number_of_travellers", "number_of_travelers"):
        m = re.search(r"\b(\d+)\b", text)
        return int(m.group(1)) if m else None
    if label in ("duration", "days"):
        m = re.search(r"\b(\d+)\s*(day|days|night|nights)?\b", text, re.IGNORECASE)
        return f"{m.group(1)} {m.group(2) or 'days'}" if m else None
    if label in ("caller_location", "location", "source"):
        return text
    if label in ("destination", "place"):
        # Regex for "to ", "go to ", "visit ", "travel to "
        patterns = [
            r"(?i)\b(?:go to|travel to|visit|to)\s+([a-zA-Z\s]+)",
        ]
        for p in patterns:
            match = re.search(p, text)
            if match:
                return match.group(1).strip()
        return text
    if label in ("interest_type", "preference", "hotel_type", "travel_mode"):
        # For simple labels, return text if it's short, otherwise None
        # This is a bit arbitrary, but better than returning long sentences
        if len(text.split()) <= 3:
            return text
        return None
    return None


# ---------- LLM Fallback ----------
def llm_fallback(current_state, current_bot_message, user_message):

    if groq_client:
        try:
            prompt = f"""You are a travel booking assistant.

User message: {user_message}

Answer briefly and then guide the user back to the booking flow.

Next question in flow:
{current_bot_message}"""

            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful travel assistant.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",
            )

            answer = chat_completion.choices[0].message.content.strip()

            if current_bot_message:
                return answer + "\n\n" + current_bot_message

            return answer

        except Exception as e:
            print("Groq error:", e, file=sys.stderr)

    # safe fallback
    if current_bot_message:
        return f"I didn't fully understand that. {current_bot_message}"

    return "Hello! How can I help you plan your travel?"

# ---------- Main Logic ----------
SLOT_ORDER = [
    "destination",
    "travellers",
    "duration",
    "hotel_type",
    "travel_mode",
    "caller_location",
    "interest_type",
    "preference"
]

def valid_bot_message(msg):
    return isinstance(msg, str) and msg.strip() != ""


def first_bot_message_for_state(states_index, state_name, context, available_entities):
    # Find the next missing slot in the preferred order that is actually in the dataset
    next_missing_slot = None
    for slot in SLOT_ORDER:
        if slot not in available_entities:
            continue
        if context.get(slot) in (None, "", []):
            next_missing_slot = slot
            break
    
    # Try to find a question in the current state that matches the next missing slot
    for s in states_index.get(state_name, []):
        entity_label = s.get("entity")
        if entity_label:
            # Normalize key
            key = "travellers" if entity_label in ("members", "travelers", "travellers", "no_of_members", "number_of_travellers", "number_of_travelers") else entity_label
            
            # If this step is for the next missing slot, return its question
            if key == next_missing_slot:
                bm = s.get("bot_question")
                if valid_bot_message(bm):
                    return bm
            
            # Skip if it's already filled or not the next one we want
            continue
        
        # If there's no entity label, it's a general question for this state
        # We only return it if we haven't found a specific slot question yet
        bm = s.get("bot_question")
        if valid_bot_message(bm):
            return bm
            
    return None


def main(user_message):
    flows = load_flows()
    steps = flatten_steps(flows)
    states_index = build_state_index(steps)
    state = load_state()
    current_state = state["current_state"]
    
    # Get set of all entities available in the dataset for validation
    available_entities = set()
    for s in steps:
        ent = s.get("entity")
        if ent:
            available_entities.add(ent)
            # Add common normalizations
            if ent in ("members", "travelers", "travellers", "no_of_members", "number_of_travellers", "number_of_travelers"):
                available_entities.add("travellers")

    # Identify current state's bot message (from JSON)
    current_bot_message = first_bot_message_for_state(states_index, current_state, state["context"], available_entities) or ""

    # --- Pre-similarity Entity Extraction ---
    # Check if the current state expects an entity
    for s in states_index.get(current_state, []):
        entity_label = s.get("entity")
        if entity_label:
            extracted = extract_entity(user_message, entity_label)
            if extracted is not None:
                # Normalize entity keys
                key_to_store = "travellers" if entity_label in ("members", "travelers", "travellers", "no_of_members", "number_of_travellers", "number_of_travelers") else entity_label
                state["context"][key_to_store] = extracted
                
                # Debug logs
                print("Current state:", current_state)
                print("Extracted entity:", extracted)
                print("Context after update:", state["context"])
                
                # Determine the next_intent
                next_intent = s.get("next_intent")
                if next_intent:
                    # Fetch the next state's bot question
                    next_reply = first_bot_message_for_state(states_index, next_intent, state["context"], available_entities)
                    
                    # If the next state's questions are all filled, recursively find the next state
                    while next_reply is None:
                        # Find the next intent from the last step of the current next_intent
                        steps_for_intent = states_index.get(next_intent, [])
                        if not steps_for_intent:
                            break
                        last_step = steps_for_intent[-1]
                        new_next = last_step.get("next_intent")
                        if not new_next or new_next == next_intent:
                            break
                        next_intent = new_next
                        next_reply = first_bot_message_for_state(states_index, next_intent, state["context"], available_entities)

                    if next_reply:
                        state["current_state"] = next_intent
                        save_state(state)
                        print("Next state:", next_intent)
                        response = {"reply": next_reply, "next_state": next_intent, "context": state["context"]}
                        print(json.dumps(response, indent=2))
                        return

    # Build candidate examples (user_utterance) for current_state
    candidates = []
    for s in states_index.get(current_state, []):
        if s.get("user_utterance"):
            candidates.append(s)

    # If no candidates for current_state, fallback
    if not candidates:
        reply = llm_fallback(current_state, current_bot_message or "", user_message)
        response = {"reply": reply, "next_state": current_state, "context": state["context"]}
        print(json.dumps(response, indent=2))
        return

    # Embedding for user message
    user_emb = embed_text(user_message)

    # Compute best match by cosine similarity
    best = None
    best_score = -1.0
    for s in candidates:
        ref_emb = embed_text(s["user_utterance"])
        score = cosine_similarity(user_emb, ref_emb)
        if score > best_score:
            best_score = score
            best = s

    THRESHOLD = 0.35
    if best and best_score >= THRESHOLD:
        # Attempt entity extraction if required (for similarity matches)
        entity_label = best.get("entity")
        extracted = None
        if entity_label:
            context_key = "travellers" if entity_label in ("members", "travelers", "travellers", "no_of_members", "number_of_travellers", "number_of_travelers") else entity_label
            current_val = state["context"].get(context_key)
            if current_val in (None, "", []):
                extracted = extract_entity(user_message, entity_label)
                if extracted is not None:
                    state["context"][context_key] = extracted
                    # Debug logs
                    print("Current state:", current_state)
                    print("Extracted entity:", extracted)
                    print("Context after update:", state["context"])

        # If required entity still missing, re-ask current state's question
        required_label = best.get("entity")
        missing = False
        if required_label:
            check_key = "travellers" if required_label in ("members", "travelers", "travellers", "no_of_members", "number_of_travellers", "number_of_travelers") else required_label
            if state["context"].get(check_key) in (None, "", []):
                missing = True

        if missing:
            reply = current_bot_message or "Please provide the requested information."
            response = {"reply": reply, "next_state": current_state, "context": state["context"]}
            print(json.dumps(response, indent=2))
            return

        # Advance to next state
        next_state = best.get("next_intent")
        # Determine next bot message from JSON for the next state
        next_reply = first_bot_message_for_state(states_index, next_state, state["context"], available_entities) if next_state else None
        
        # If the next state's questions are all filled, recursively find the next state
        while next_reply is None and next_state:
            steps_for_intent = states_index.get(next_state, [])
            if not steps_for_intent:
                break
            last_step = steps_for_intent[-1]
            new_next = last_step.get("next_intent")
            if not new_next or new_next == next_state:
                break
            next_state = new_next
            next_reply = first_bot_message_for_state(states_index, next_state, state["context"], available_entities)

        if not next_reply:
            # If next state not found, fallback safely
            next_reply = llm_fallback(current_state, current_bot_message or "", user_message)
            next_state = current_state

        state["current_state"] = next_state
        save_state(state)
        print("Next state:", next_state)
        response = {"reply": next_reply, "next_state": next_state, "context": state["context"]}
        print(json.dumps(response, indent=2))
    else:
        # LLM fallback without changing state
        reply = llm_fallback(current_state, current_bot_message or "", user_message)
        response = {"reply": reply, "next_state": current_state, "context": state["context"]}
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a message")
