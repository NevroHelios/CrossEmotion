from pathlib import Path
from collections import defaultdict, Counter
data = "data/ifeed/IFEED_Base"

files = list(Path(data).glob("*/*/*"))
print(f"Found {len(files)} files in {data}")
print("First 10 files:")
for file in files:
    if file.name.split("_")[-1].split(".")[0].lower() == "fea":
        print(file)
    if file.name.split("_")[-1].split(".")[0].lower() == "fear":
        print(file)

emotions = [file.name.split("_")[-1].split(".")[0].lower() for file in files]
print(f"Unique emotions found: {set(emotions)}")
print(f"Total unique emotions: {len(set(emotions))}")

emotion_counts = Counter(emotions)
print("Emotion counts:")
for emotion, count in emotion_counts.items():
    print(f"{emotion}: {count}")