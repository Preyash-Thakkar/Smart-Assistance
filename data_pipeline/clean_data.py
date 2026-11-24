import os
import re
import json
import glob
import random

QUESTION_TO_INTENT = {
    "all soft sound are too soft": "LOUDNESS_SOFT_TOO_SOFT",
    "all soft sounds too soft": "LOUDNESS_SOFT_TOO_SOFT",
    "all loud sounds are too loud": "LOUDNESS_LOUD_TOO_LOUD",
    "all loud sounds too loud": "LOUDNESS_LOUD_TOO_LOUD",
    "loud sounds are distorted": "LOUDNESS_LOUD_DISTORTED",
    "loud sounds distorted": "LOUDNESS_LOUD_DISTORTED",
    "nearby sounds are loud": "LOUDNESS_NEARBY_TOO_LOUD",
    "nearby sounds too loud": "LOUDNESS_NEARBY_TOO_LOUD",
    "alarm/childrencrying are loud": "LOUDNESS_ALARMS_CRYING_LOUD",
    "alarms/children crying too loud": "LOUDNESS_ALARMS_CRYING_LOUD",
    "alarms/childr en crying too loud": "LOUDNESS_ALARMS_CRYING_LOUD",
    "sudden sounds/door slamming sounds are loud": "LOUDNESS_SUDDEN_SOUNDS_LOUD",
    "sudden sounds (e.g., door slams) too loud": "LOUDNESS_SUDDEN_SOUNDS_LOUD",
    "own chewing sounds are loud": "LOUDNESS_CHEWING_TOO_LOUD",
    "own chewing sounds too loud": "LOUDNESS_CHEWING_TOO_LOUD",
    "running water sound feels loud": "LOUDNESS_WATER_FAN_LOUD",
    "running fan sounds are loud": "LOUDNESS_WATER_FAN_LOUD",
    "running water/fan sounds too loud": "LOUDNESS_WATER_FAN_LOUD",
    "own voice muffled/dull": "OWN_VOICE_MUFFLED_DULL",
    "own voice hollow": "OWN_VOICE_HOLLOW",
    "own voice too sharp/metallic": "OWN_VOICE_SHARP_METALLIC",
    "own voice thin": "OWN_VOICE_THIN",
    "own voice boomy/loud": "OWN_VOICE_BOOMY_LOUD",
    "speech too soft": "SPEECH_TOO_SOFT",
    "whisper/distant speech too soft": "SPEECH_WHISPER_DISTANT_SOFT",
    "whisper/dista nt speech too soft": "SPEECH_WHISPER_DISTANT_SOFT",
    "speech too loud/boomy": "SPEECH_TOO_LOUD_BOOMY",
    "speech echoic": "SPEECH_ECHOIC",
    "poor speech quality": "SPEECH_POOR_QUALITY",
    "sounds feel tiny & metallic & sharp": "SOUND_TINY_METALLIC_SHARP",
    "sounds tiny/metallic/sharp": "SOUND_TINY_METALLIC_SHARP",
    "sounds tiny/metallic/s harp": "SOUND_TINY_METALLIC_SHARP",
    "sounds tiny/metallic/ sharp": "SOUND_TINY_METALLIC_SHARP",
    "sounds are echoic": "SOUND_ECHOIC",
    "sounds echoic": "SOUND_ECHOIC",
    "sounds are muffled or dull": "SOUND_MUFFLED_DULL",
    "sounds muffled/dull": "SOUND_MUFFLED_DULL",
    "sounds are mechanical or artificial": "SOUND_ARTIFICIAL_MECHANICAL",
    "sounds artificial/mechanical": "SOUND_ARTIFICIAL_MECHANICAL",
    "sounds artificial/mech anical": "SOUND_ARTIFICIAL_MECHANICAL",
    "too much bass": "SOUND_TOO_MUCH_BASS",
    "sounds are distorted": "SOUND_DISTORTED",
    "sounds distorted": "SOUND_DISTORTED",
    "soft ambience noise is too noticeable": "NOISE_AMBIENT_NOTICEABLE",
    "ambient noise too noticeable": "NOISE_AMBIENT_NOTICEABLE",
    "sound in noise is uncomfortable": "NOISE_SOUND_UNCOMFORTABLE",
    "sudden noise is uncomfortable": "NOISE_SUDDEN_UNCOMFORTABLE",
    "sudden noise uncomfortable": "NOISE_SUDDEN_UNCOMFORTABLE",
    "background noise is too loud": "NOISE_BACKGROUND_RESTAURANT",
    "restaurant noise is loud": "NOISE_BACKGROUND_RESTAURANT",
    "street&#x2f;traffic noise is too loud": "NOISE_BACKGROUND_RESTAURANT",
    "background noise (e.g., restaurants) too loud": "NOISE_BACKGROUND_RESTAURANT",
    "wind noise is loud": "NOISE_WIND_TOO_LOUD",
    "wind noise too loud": "NOISE_WIND_TOO_LOUD",
    "crowd noise is loud": "NOISE_CROWD_TOO_LOUD",
    "crowd noise too loud": "NOISE_CROWD_TOO_LOUD",
    "hearing aid whistles": "FEEDBACK_WHISTLING",
    "hearing aid whistles (feedback)": "FEEDBACK_WHISTLING",
}

GARBAGE_QUESTIONS = {"sdc", "afse", "sdfgsd", "sdfghj"}

def normalize_text(text: str) -> str:
    if not text: return ""
    return re.sub(r"\s+", " ", text.strip().rstrip(".").lower())

def extract_dicts(data):
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        for item in data:
            yield from extract_dicts(item)

def safe_parse_json(content: str):
    dicts = []
    try:
        parsed = json.loads(content)
        dicts.extend(list(extract_dicts(parsed)))
        if dicts: return dicts
    except json.JSONDecodeError:
        pass
        
    for line in content.splitlines():
        line = line.strip()
        if not line: continue
        try:
            parsed = json.loads(line)
            dicts.extend(list(extract_dicts(parsed)))
        except json.JSONDecodeError:
            continue
            
    return dicts

def main():
    os.makedirs("data/processed", exist_ok=True)
    
    # 1. Clean Database
    db_path = "Real Data/test.complains.json"
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            raw_db = json.load(f)
            
        cleaned_db = []
        for doc in raw_db:
            raw_q = doc.get("question", "")
            q_norm = normalize_text(raw_q)
            if q_norm in GARBAGE_QUESTIONS: continue
            doc["intent_code"] = QUESTION_TO_INTENT.get(q_norm, "UNMAPPED")
            cleaned_db.append(doc)
            
        with open("data/processed/complains_cleaned.json", "w", encoding="utf-8") as f:
            json.dump(cleaned_db, f, indent=2)
        print(f"[Phase 1] Cleaned database saved. Total valid records: {len(cleaned_db)}")

    # 2. Process Training Batches
    batch_files = sorted(glob.glob("Config Assist Multiples/**/batch*_complaint_english.json*", recursive=True))
    print(f"[Phase 2] Found {len(batch_files)} batch utterance files.")
    
    master_dataset = []
    
    for filepath in batch_files:
        batch_name = os.path.basename(os.path.dirname(filepath))
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        records = safe_parse_json(content)
        
        source_mapping_path = filepath.replace("_complaint_english.jsonl", "_source_mapping.json").replace("_complaint_english.json", "_source_mapping.json")
        canonical_intent = "UNKNOWN"
        
        if os.path.exists(source_mapping_path):
            with open(source_mapping_path, "r", encoding="utf-8") as sm_f:
                sm_content = sm_f.read().strip()
                sm_records = safe_parse_json(sm_content)
                if sm_records:
                    src_q = sm_records[0].get("source_complaint_record", {}).get("question", "")
                    canonical_intent = QUESTION_TO_INTENT.get(normalize_text(src_q), "UNKNOWN")

        for r in records:
            if r.get("is_hard_negative", False) or r.get("safety_flag", False):
                continue
            utterance_text = r.get("text") or r.get("text_romanized")
            if not utterance_text:
                continue
            
            # Safe extraction for malformed slots
            slots = r.get("slots")
            persona = slots.get("situation", "unspecified") if isinstance(slots, dict) else "unspecified"
                
            master_dataset.append({
                "utterance": utterance_text,
                "intent_code": canonical_intent,
                "topic_id": r.get("topic_id"),
                "persona": persona,
                "source_batch": batch_name
            })

    print(f"[Phase 2] Consolidated {len(master_dataset)} valid training utterances.")
    if not master_dataset: return

    # 3. Stratified Split
    random.seed(42)
    random.shuffle(master_dataset)
    
    train_idx = int(0.80 * len(master_dataset))
    val_idx = int(0.90 * len(master_dataset))
    
    train_data = master_dataset[:train_idx]
    val_data = master_dataset[train_idx:val_idx]
    test_data = master_dataset[val_idx:]
    
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test_holdout", test_data)]:
        out_path = f"data/processed/{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        print(f"[Phase 3] Saved {len(split_data)} records to {out_path}")

if __name__ == "__main__":
    main()