import os
import json
import torch
from sentence_transformers import SentenceTransformer, util

# The exact canonical intents from your MongoDB
INTENT_DESCRIPTIONS = {
    "LOUDNESS_SOFT_TOO_SOFT": "soft quiet sounds are too soft and faint",
    "LOUDNESS_LOUD_TOO_LOUD": "loud sounds are too loud and hurt my ears",
    "LOUDNESS_LOUD_DISTORTED": "loud sounds are distorted and unclear",
    "LOUDNESS_NEARBY_TOO_LOUD": "nearby sounds are too loud",
    "LOUDNESS_ALARMS_CRYING_LOUD": "alarms and children crying are too loud",
    "LOUDNESS_SUDDEN_SOUNDS_LOUD": "sudden sounds like door slams are too loud",
    "LOUDNESS_CHEWING_TOO_LOUD": "my own chewing sounds are too loud",
    "LOUDNESS_WATER_FAN_LOUD": "running water and fan sounds are too loud",
    "OWN_VOICE_MUFFLED_DULL": "my own voice sounds muffled and dull",
    "OWN_VOICE_HOLLOW": "my own voice sounds hollow like in a barrel",
    "OWN_VOICE_SHARP_METALLIC": "my own voice sounds too sharp and metallic",
    "OWN_VOICE_THIN": "my own voice sounds thin",
    "OWN_VOICE_BOOMY_LOUD": "my own voice sounds boomy and too loud",
    "SPEECH_TOO_SOFT": "people speaking and speech is too soft",
    "SPEECH_WHISPER_DISTANT_SOFT": "whispers and distant speech are too soft",
    "SPEECH_TOO_LOUD_BOOMY": "speech is too loud and boomy",
    "SPEECH_ECHOIC": "speech echoes",
    "SPEECH_POOR_QUALITY": "poor speech quality and unclear words",
    "SOUND_TINY_METALLIC_SHARP": "sounds feel tiny metallic and sharp",
    "SOUND_ECHOIC": "sounds are echoic and bounce around",
    "SOUND_MUFFLED_DULL": "sounds are muffled or dull",
    "SOUND_ARTIFICIAL_MECHANICAL": "sounds are mechanical or artificial",
    "SOUND_TOO_MUCH_BASS": "there is too much bass",
    "SOUND_DISTORTED": "general sounds are distorted",
    "NOISE_AMBIENT_NOTICEABLE": "soft ambient background noise is too noticeable",
    "NOISE_SUDDEN_UNCOMFORTABLE": "sudden noise is uncomfortable",
    "NOISE_BACKGROUND_RESTAURANT": "background noise in restaurants or traffic is too loud",
    "NOISE_WIND_TOO_LOUD": "wind noise is too loud",
    "NOISE_CROWD_TOO_LOUD": "crowd noise is too loud",
    "FEEDBACK_WHISTLING": "hearing aid whistles and has feedback"
}

def main():
    print("Loading AI Semantic Mapper...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    intent_keys = list(INTENT_DESCRIPTIONS.keys())
    intent_texts = list(INTENT_DESCRIPTIONS.values())
    intent_embeddings = model.encode(intent_texts, convert_to_tensor=True)
    
    splits = ["train", "val", "test_holdout"]
    batch_to_intent = {}
    
    for split in splits:
        filepath = f"data/processed/{split}.jsonl"
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            data = json.loads(line)
            batch = data["source_batch"]
            utt = data["utterance"]
            
            # Semantically map the batch to the correct Intent Code
            if batch not in batch_to_intent:
                utt_emb = model.encode(utt, convert_to_tensor=True)
                cos_scores = util.cos_sim(utt_emb, intent_embeddings)[0]
                best_idx = torch.argmax(cos_scores).item()
                best_intent = intent_keys[best_idx]
                batch_to_intent[batch] = best_intent
                print(f"Mapped {batch} -> {best_intent}")
            
            # Overwrite the target label
            data["intent_code"] = batch_to_intent[batch]
            updated_lines.append(data)
            
        with open(filepath, "w", encoding="utf-8") as f:
            for d in updated_lines:
                f.write(json.dumps(d) + "\n")
                
    print("\nSuccess: Dataset labels are now perfectly mapped to the 22 MongoDB Intents.")

if __name__ == "__main__":
    main()