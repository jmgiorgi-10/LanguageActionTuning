import os
import json
import time
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the intent types we want to scale out
SEED_BACKBONES = [
    {"intent": "send_message"},
    {"intent": "start_workout"},
    {"intent": "check_surf"},
    {"intent": "open_app"}
]

def get_localized_variants(intent, model_name="gpt-oss:20b-cloud"):
    url = "https://ollama.com/api/chat"
    api_key = os.getenv("OLLAMA_API_KEY")
    
    # Prompt explicitly commanding high parametric entropy and locale-specific grounding
    prompt = f"""
    You are an expert NLU data engineer. Generate exactly 500 highly realistic, diverse conversational user voice prompts for the intent: '{intent}'
    
    Distribute the 100 variations cleanly across these target locales:
    1. es_ES (Spain Spanish)
    2. es_AR (Argentine Spanish)
    3. en_US (American English)

    # THE DIVERSITY MANDATE (EXTEND THE MATRIX)
    You must dynamically invent and rotate slots (names, locations, activities, apps) across all 500 variations. 
    Do NOT limit yourself to a few repetitions. Use your internal knowledge to maximize vocabulary entropy.

    Here are reference seeds, but you MUST extend them with hundreds of your own variations:
    - RECIPIENTS: [Juan, Mateo, Maca, John] -> Extend this to include dozens of authentic local names matching the specific locale (e.g., utilize common Anglo names for en_US, and distinct local names for es_ES and es_AR).
    - SURF SPOTS: [Barceloneta, Ericeira] -> Extend this to dozens of real-world surf breaks globally (e.g., Huntington Beach, Mundaka, Bells Beach, Jeffreys Bay, etc.).
    - WORKOUTS: [running, surfing, fútbol] -> Extend this to any realistic sport, training session, or physical activity.
    - APPS: [Playtomic, Smou, Spotify] -> Extend this to widely used mobile applications across utility, sports booking, mobility, and messaging spaces.

    # RULES FOR DYNAMIC GENERATION:
    1. HIGH ENTROPY: A specific name, spot, or app should ideally never appear more than twice in the entire dataset. Continually introduce new entities.
    2. LOCAL REALISM: For es_AR, use realistic Argentine syntax (voseo: "fijate", "vení", "che", "cancha"). For es_ES, use Iberian terms ("pistas", "entrenamiento", "móvil"). For en_US, use natural American vocal phrasing.
    3. Return a clean JSON array of objects containing 'locale', 'text', and 'slots'. Ensure the 'slots' object explicitly captures the exact unique values you generated for that specific string.
    """
    
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,  
        "format": "json"  
    }
    
    response = requests.post(
        url, 
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, 
        json=data
    )
    
    if response.status_code != 200:
        print(f"\nSERVER ERROR ({response.status_code}): {response.text}")
        raise Exception(f"Ollama Cloud responded with status code {response.status_code}")

    raw_text = response.json()["message"]["content"].strip()
    
    if not raw_text:
        raise Exception("Server returned a success status but the response string was completely empty.")
        
    # Clean out markdown block wrappers if the model injected them
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]  
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]  
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3] 
        
    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as je:
        print(f"\nJSON PARSE ERROR: Failed to decode this cleaned string:\n{raw_text}")
        raise je


# --- THE EXECUTION LOOP ---
output_dataset = []
output_file = "synthetic_dataset.jsonl"

print("Starting generation pipeline...")

for backbone in SEED_BACKBONES:
    current_intent = backbone["intent"]
    print(f"Processing intent: {current_intent}...")
    
    try:
        # Request the 500 expanded, dynamic variants from the LLM
        variants = get_localized_variants(current_intent)
        
        intent_count = 0
        for variant in variants:
            training_row = {
                "intent": current_intent,
                "locale": variant["locale"],
                "user_query": variant["text"],
                "slots": variant.get("slots", {})
            }
            output_dataset.append(training_row)
            intent_count += 1
            
        print(f"-> Successfully extracted {intent_count} rows for '{current_intent}'")
        
        # Cooldown to respect rate limits and keep connection healthy
        time.sleep(4.0)
        
    except Exception as e:
        print(f"⚠️ Skipped backbone '{current_intent}' due to error: {e}")

# Save the structured rows in line-by-line JSONL format
print(f"\nWriting a total of {len(output_dataset)} rows to {output_file}...")
with open(output_file, "a", encoding="utf-8") as f:
    for row in output_dataset:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Pipeline complete. You are ready to split and run your tokenizer now!")