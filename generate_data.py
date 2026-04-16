import json
from model import predict_text

data = []

examples = [
    # Toxic
    ("Why are you like this?", "You are useless"),
    ("Explain this", "This is stupid"),

    # Polite
    ("Can you help me?", "Sure, I'd be happy to help!"),
    ("Explain ML", "Of course! Machine learning allows systems to learn from data."),

    # Neutral
    ("Did you finish?", "Yes"),
    ("What time is it?", "It is 5 PM"),

    # Irrelevant
    ("What is AI?", "I like pizza"),
    ("Explain ML", "Football is fun"),

    # Relevant
    ("What is AI?", "AI is the simulation of human intelligence"),
    ("Explain ML", "Machine learning is a subset of AI"),

    # Emotional
    ("I feel sad", "I understand how you feel"),
    ("I am stressed", "Stay strong, things will improve"),

    # Short
    ("Did you do it?", "ok"),
    ("Are you ready?", "yes"),

    # Detailed
    ("Explain ML", "Machine learning involves training models on data to make predictions."),
    ("What is AI?", "AI enables machines to mimic human intelligence and decision making.")
]

# helper function to convert numpy → python types
def convert_types(obj):
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(v) for v in obj]
    else:
        try:
            return float(obj) if isinstance(obj, (float,)) else int(obj)
        except:
            return obj


# Generate 50 samples
for i in range(50):
    user, assistant = examples[i % len(examples)]

    output = predict_text(assistant, user)

    clean_scores = convert_types(output["scores"])
    clean_conf = convert_types(output["confidence"])

    data.append({
        "conversation": [
            {"user": user},
            {"assistant": assistant}
        ],
        "scores": clean_scores,
        "confidence": clean_conf
    })

# Save file
with open("data.json", "w") as f:
    json.dump(data, f, indent=4)

print("data.json with scores + confidence created")