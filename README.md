# Travel Booking Conversation Intelligence Agent

This project implements a Conversation Intelligence Agent for a travel booking system. It operates as a strict controller over pre-defined conversation flows defined in an Excel sheet.

## Project Structure

```
.
├── agent_logic.py          # Main agent logic (intent matching, state management, flow control)
├── convertjson.py          # Utility script to convert Excel flow definition to JSON
├── data/                   # Data directory
│   ├── Harshita tours and travels.xlsx  # Source of truth for conversation flows
│   └── dataset.json        # Processed JSON conversation flows (generated)
├── session_state.json      # Temporary file storing the current conversation state
└── README.md               # Project documentation
```

## Setup & Requirements

- Python 3.x
- pandas
- openpyxl

Install dependencies:

```bash
pip install pandas openpyxl
```

## Usage

### 1. Update Conversation Flows (Optional)

If you modify `data/Harshita tours and travels.xlsx`, regenerate the JSON dataset:

```bash
python3 convertjson.py
```

### 2. Run the Agent

Interact with the agent via the command line. The agent maintains state between calls using `session_state.json`.

**Example Interaction:**

```bash
# Start the conversation
python3 agent_logic.py "I want a trip to Rajasthan"

# Respond to the agent
python3 agent_logic.py "Udaipur"
```

### 3. Resetting State

To start a fresh conversation, delete the `session_state.json` file:

```bash
rm session_state.json
```

## Features

- **Strict Flow Adherence:** Follows the defined steps in the Excel/JSON.
- **Intent Matching:** Uses string similarity to match user inputs to defined user utterances.
- **Entity Extraction:** Extracts entities (e.g., destination, interest type) based on the flow definition.
- **State Management:** Persists conversation state to handle multi-turn interactions.
- **Fallback Mechanism:** Provides generic responses when the user intent is unclear or off-topic.

## Configuration

- **Threshold:** The similarity threshold for intent matching is set in `agent_logic.py` (default: 0.4).
- **Paths:** File paths for data and state are defined at the top of `agent_logic.py`.
