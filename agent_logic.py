import json
import math
import os
import re
import sys
from collections import Counter

DATASET_FILE = 'data/dataset.json'
STATE_FILE = 'session_state.json'

from google import genai
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None
if GEMINI_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_KEY)
    except Exception as e:
        print("Gemini init error:", e)


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
    return bow_embedding(text)


# ---------- Entity Extraction ----------
def extract_entity(user_text, entity_label):
    text = user_text.strip()
    if not entity_label:
        return None
    label = entity_label.lower()
    if label in ("travellers", "travelers", "members", "no_of_members", "number_of_travellers", "number_of_travelers"):
        m = re.search(r"\b(\d+)\b", text)
        return int(m.group(1)) if m else None
    if label in ("duration", "days"):
        m = re.search(r"\b(\d+)\s*(day|days|night|nights)?\b", text, re.IGNORECASE)
        return f"{m.group(1)} {m.group(2) or 'days'}" if m else None
    if label in ("caller_location", "location", "source"):
        return text
    if label in ("destination", "place"):
        return text
    if label in ("interest_type", "preference", "hotel_type", "travel_mode"):
        return text
    return text


# ---------- LLM Fallback ----------
def llm_fallback(current_state, current_bot_message, user_message):

    if gemini_client:
        try:
            prompt = f"""
                You are a travel booking assistant.

                User message: {user_message}

                Answer briefly and then guide the user back to the booking flow.

                Next question in flow:
                {current_bot_message}
                """

            response = gemini_client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=prompt
            )

            answer = response.text.strip()

            if current_bot_message:
                return answer + "\n\n" + current_bot_message

            return answer

        except Exception as e:
            print("Gemini error:", e)

    # safe fallback
    if current_bot_message:
        return f"I didn't fully understand that. {current_bot_message}"

    return "Hello! How can I help you plan your travel?"

# ---------- Main Logic ----------
def valid_bot_message(msg):
    return isinstance(msg, str) and msg.strip() != ""


def first_bot_message_for_state(states_index, state_name):
    for s in states_index.get(state_name, []):
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

    # Identify current state's bot message (from JSON)
    current_bot_message = first_bot_message_for_state(states_index, current_state) or ""

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

    THRESHOLD = 0.55
    if best and best_score >= THRESHOLD:
        # Attempt entity extraction if required
        entity_label = best.get("entity")
        if entity_label:
            key_for_check = "travellers" if entity_label in ("members", "travelers") else entity_label
            current_val = state["context"].get(key_for_check)
            if current_val is None:
                extracted = extract_entity(user_message, entity_label)
                if entity_label in ("members", "travelers"):
                    # Normalize to 'travellers'
                    if extracted is not None:
                        state["context"]["travellers"] = extracted
                else:
                    if extracted is not None:
                        state["context"][entity_label] = extracted

        # If required entity still missing, re-ask current state's question
        required_label = best.get("entity")
        missing = False
        if required_label:
            check_key = "travellers" if required_label == "members" else required_label
            if state["context"].get(check_key) in (None, ""):
                missing = True

        if missing:
            reply = current_bot_message or "Please provide the requested information."
            response = {"reply": reply, "next_state": current_state, "context": state["context"]}
            print(json.dumps(response, indent=2))
            return

        # Advance to next state
        next_state = best.get("next_intent")
        # Determine next bot message from JSON for the next state
        next_reply = first_bot_message_for_state(states_index, next_state) if next_state else None
        if not next_reply:
            # If next state not found, fallback safely
            next_reply = llm_fallback(current_state, current_bot_message or "")
            next_state = current_state

        state["current_state"] = next_state
        save_state(state)
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
