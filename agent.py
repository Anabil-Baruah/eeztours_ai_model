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
    "travel_mode",
    "hotel_type",
    "caller_location",
    "interest_type",
    "preference"
]

SLOT_TO_STATE = {
    "destination": "Ask_Destination",
    "travellers": "Ask_Members",
    "duration": "Ask_Duration",
    "travel_mode": "Ask_Travel_Mode",
    "hotel_type": "Ask_Hotel_Type",
    "caller_location": "Ask_Location",
    "interest_type": "Ask_Interest",
    "preference": "Ask_Preference"
}

def valid_bot_message(msg):
    return isinstance(msg, str) and msg.strip() != ""


def first_bot_message_for_state(states_index, state_name):
    # Simply return the first valid bot message for this state
    for s in states_index.get(state_name, []):
        bm = s.get("bot_question")
        if bm and isinstance(bm, str) and bm.strip() != "":
            return bm
    return None


def is_question(text):
    text = text.lower().strip()
    question_patterns = [
        r"\bwhat\b", r"\bwhy\b", r"\bhow\b", r"\bwhen\b", r"\bwhere\b", r"\bwhich\b",
        r"\bcan you\b", r"\bcould you\b", r"\bdo you\b", r"\btell me\b", r"\bexplain\b",
        r"\brates\b", r"\bprice\b", r"\bcost\b", r"\bbest hotel\b", r"\brecommend\b"
    ]
    if text.endswith("?"):
        return True
    for pattern in question_patterns:
        if re.search(pattern, text):
            return True
    return False


def is_slot_relevant(text, slot):
    text = text.lower().strip()
    if not text:
        return False
    
    if slot == "destination":
        # Destination should look like a place name (handled by regex/LLM extraction, here just a basic check)
        # We allow it if it's reasonably short or contains "to"
        return len(text.split()) <= 5 or "to" in text or "visit" in text
    elif slot == "travellers":
        # Travellers should contain a number
        return any(char.isdigit() for char in text) or any(word in text for word in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "couple", "family"])
    elif slot == "duration":
        # Duration should contain time words
        time_words = ["day", "week", "month", "night", "days", "weeks", "months", "nights"]
        return any(word in text for word in time_words)
    elif slot == "travel_mode":
        # travel_mode -> train, flight, bus, car
        modes = ["train", "flight", "bus", "car", "plane", "taxi"]
        return any(mode in text for mode in modes)
    elif slot == "hotel_type":
        # hotel_type -> budget, luxury, 3 star, 4 star, resort
        types = ["budget", "luxury", "star", "resort", "hotel", "homestay", "hostel"]
        return any(t in text for t in types)
    elif slot == "caller_location":
        # caller_location -> city name
        return len(text.split()) <= 3
    elif slot == "interest_type":
        # interest_type -> adventure, cultural, wildlife, relaxation
        interests = ["adventure", "cultural", "wildlife", "relaxation", "trekking", "sightseeing", "beach", "history"]
        return any(interest in text for interest in interests)
    elif slot == "preference":
        # preference -> view, luxury, budget, near beach, etc.
        prefs = ["view", "luxury", "budget", "beach", "pool", "garden", "wifi", "ac"]
        return any(pref in text for pref in prefs)
    
    return True


def main(user_message):
    flows = load_flows()
    steps = flatten_steps(flows)
    states_index = build_state_index(steps)
    state = load_state()
    current_state = state["current_state"]
    
    # Identify current state's bot message (from JSON)
    current_bot_message = first_bot_message_for_state(states_index, current_state) or ""

    # --- 1. Determine the expected slot ---
    current_slot = None
    for slot in SLOT_ORDER:
        if state["context"].get(slot) is None:
            current_slot = slot
            break
    
    # --- 2. Create a slot relevance check & Detect general questions ---
    is_rel = is_slot_relevant(user_message, current_slot) if current_slot else False
    is_ques = is_question(user_message)
    
    # Trigger LLM fallback if it's a question OR not relevant to current slot
    # (Exception: if it's a greeting, we might not want fallback, but let's follow instructions)
    fallback_triggered = is_ques or not is_rel
    
    # Special case: if it's a simple greeting like "hi", don't necessarily trigger fallback as "not relevant"
    if user_message.lower().strip() in ["hi", "hello", "hey", "greetings"]:
        fallback_triggered = False

    # --- 3. Entity Extraction & Context Update (Only if not fallback) ---
    extracted_slot = None
    if not fallback_triggered and current_slot:
        # Try to extract the entity for the current expected slot
        entity_label = None
        for s in steps:
            ent = s.get("entity")
            if ent:
                key = "travellers" if ent in ("members", "travelers", "travellers", "no_of_members", "number_of_travellers", "number_of_travelers") else ent
                if key == current_slot:
                    entity_label = ent
                    break
        
        if not entity_label:
            entity_label = current_slot
            
        extracted = extract_entity(user_message, entity_label)
        if extracted is not None:
            state["context"][current_slot] = extracted
            extracted_slot = current_slot

        # If still not extracted, try similarity matching
        if extracted_slot is None:
            candidates = [s for s in states_index.get(current_state, []) if s.get("user_utterance")]
            if candidates:
                user_emb = embed_text(user_message)
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
                    entity_label = best.get("entity")
                    if entity_label:
                        extracted = extract_entity(user_message, entity_label)
                        if extracted is not None:
                            key_to_store = "travellers" if entity_label in ("members", "travelers", "travellers", "no_of_members", "number_of_travellers", "number_of_travelers") else entity_label
                            state["context"][key_to_store] = extracted
                            extracted_slot = key_to_store

    # --- 4. Determine Next State based on Missing Slots ---
    next_slot = None
    for slot in SLOT_ORDER:
        if state["context"].get(slot) is None:
            next_slot = slot
            break
    
    if next_slot:
        next_state = SLOT_TO_STATE[next_slot]
        next_reply = first_bot_message_for_state(states_index, next_state)
        if not next_reply:
            next_reply = f"Could you please tell me about your {next_slot.replace('_', ' ')}?"
    else:
        next_state = "Booking_Complete"
        next_reply = "Thank you! I have collected all the details for your trip. We will get back to you soon."

    # --- 5. Debugging logs ---
    print("User message:", user_message)
    print("Current slot:", current_slot)
    print("Slot relevance:", is_rel)
    print("Question detected:", is_ques)
    print("Fallback triggered:", fallback_triggered)
    print("Context now:", state["context"])
    print("Next slot:", next_slot)
    print("Next state:", next_state)

    # --- 6. Save and Return ---
    state["current_state"] = next_state
    save_state(state)
    
    if fallback_triggered:
        # LLM answers the question AND guides back to the flow
        # We use next_reply which is the question for the current (still missing) slot
        reply = llm_fallback(next_state, next_reply, user_message)
    else:
        reply = next_reply

    response = {"reply": reply, "next_state": next_state, "context": state["context"]}
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a message")
