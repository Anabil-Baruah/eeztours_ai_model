import json
import difflib
import os
import sys

DATASET_FILE = 'data/dataset.json'
STATE_FILE = 'session_state.json'

def load_data():
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        flows = json.load(f)
    # Flatten steps
    all_steps = []
    for flow in flows:
        all_steps.extend(flow['steps'])
    return all_steps

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"current_state": "Greeting", "context": {}, "last_step_index": -1}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def get_similarity(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def extract_entity(user_text, entity_label):
    # Simple extraction logic (placeholder for more complex NLP)
    # If entity_label is 'destination', look for capitalized words not at start?
    # For now, we return the user_text as a fallback or a dummy logic
    # In a real tool, this would use NER.
    # We will try to be smart: if the user utterance in JSON has the entity, 
    # we might map the structure. 
    # E.g. JSON: "I want a trip to Rajasthan (Jaipur)" -> Entity: Rajasthan
    # User: "I want a trip to Kerala" -> Entity: Kerala
    # This requires structural matching.
    return user_text # Simplified

def main(user_message):
    data = load_data()
    state = load_state()
    
    current_state = state['current_state']
    last_idx = state['last_step_index']
    
    # 1. Filter candidates
    # Candidates are steps where current_intent == current_state
    candidates = []
    for idx, step in enumerate(data):
        if step['current_intent'] == current_state:
            candidates.append((idx, step))
            
    # 2. Match Intent
    best_match = None
    best_score = 0
    matched_idx = -1
    
    for idx, step in candidates:
        score = get_similarity(user_message, step['user_utterance'])
        if score > best_score:
            best_score = score
            best_match = step
            matched_idx = idx
            
    THRESHOLD = 0.4 # Lowered for demo robustness
    
    if best_score >= THRESHOLD:
        # Match found
        next_state = best_match['next_intent']
        
        # Entity Extraction
        if best_match.get('entity'):
            # Very basic extraction: 
            # If the user utterance is "I want a trip to Rajasthan (Jaipur)"
            # And user says "I want a trip to Kerala"
            # We assume the diff is the entity.
            # For now, just store the full message or a heuristic
            entity_val = user_message 
            state['context'][best_match['entity']] = entity_val
            
        # Determine Bot Reply
        # Logic: Look at next step in sequence (matched_idx + 1)
        # If it belongs to next_state, use its bot_question.
        # Else, find first step of next_state.
        
        reply = "I'm not sure what to say."
        next_step_idx = matched_idx + 1
        
        found_next = False
        if next_step_idx < len(data):
            next_step = data[next_step_idx]
            # Check if this next step logically follows in the conversation flow
            # The 'current_intent' of the next step should match the 'next_state' we just transitioned to
            if next_step['current_intent'] == next_state:
                reply = next_step['bot_question']
                found_next = True
        
        if not found_next:
            # Fallback: Find first step of next_state
            for step in data:
                if step['current_intent'] == next_state:
                    reply = step['bot_question']
                    break
        
        # Update State
        state['current_state'] = next_state
        state['last_step_index'] = matched_idx
        save_state(state)
        
        response = {
            "reply": reply,
            "next_state": next_state,
            "context": state['context']
        }
        print(json.dumps(response, indent=2))
        
    else:
        # LLM Fallback
        response = {
            "reply": "I apologize, but I can only assist with travel bookings. Could you please clarify your request regarding your trip?",
            "next_state": current_state, # Don't advance
            "context": state['context']
        }
        print(json.dumps(response, indent=2))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a message")
