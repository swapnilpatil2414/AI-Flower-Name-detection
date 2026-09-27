import os

print("=== Scanning E:/ for any flower folders or videos ===")
for root, dirs, files in os.walk("e:/"):
    # skip .git, $RECYCLE.BIN
    if any(p in ['$RECYCLE.BIN', 'System Volume Information', '.git', 'node_modules'] for p in root.split(os.sep)):
        continue
    for d in dirs:
        if "flower" in d.lower():
            p = os.path.join(root, d)
            print("Found dir in E:/ :", p)
            try:
                print("   Files in dir:", os.listdir(p))
            except Exception as e:
                print("   Error:", e)
    for f in files:
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            if "flower" in f.lower() or "test" in f.lower():
                print("Found video in E:/ :", os.path.join(root, f))

print("\n=== Scanning C:/Users/saibaba for flower folders or videos ===")
user_base = "C:/Users/saibaba"
for sub in ["Desktop", "Downloads", "Documents", "Videos"]:
    sub_path = os.path.join(user_base, sub)
    if os.path.exists(sub_path):
        for root, dirs, files in os.walk(sub_path):
            for d in dirs:
                if "flower" in d.lower():
                    p = os.path.join(root, d)
                    print(f"Found dir in {sub} :", p)
                    try:
                        print("   Files in dir:", os.listdir(p))
                    except Exception as e:
                        print("   Error:", e)
            for f in files:
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    print(f"Found video in {sub} :", os.path.join(root, f))

