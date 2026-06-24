from pathlib import Path
import pickle

VIDEO_DIR = Path("downloaded_data/Video")
LABEL_DIR = Path("downloaded_data/Label")

print("=== 영상 폴더 확인 ===")

if not VIDEO_DIR.exists():
    print("downloaded_data/Video 폴더가 없습니다.")
else:
    video_files = [p for p in VIDEO_DIR.rglob("*") if p.is_file()]
    print("영상 파일 개수:", len(video_files))

    print("영상 파일 예시:")
    for video in video_files[:20]:
        print(video)

print()
print("=== 라벨 폴더 확인 ===")

if not LABEL_DIR.exists():
    print("downloaded_data/Label 폴더가 없습니다.")
    exit()

label_files = [p for p in LABEL_DIR.rglob("*.p") if p.is_file()]

if not label_files:
    print(".p 라벨 파일을 찾지 못했습니다.")
    exit()

for label_file in label_files:
    print()
    print("=" * 80)
    print("라벨 파일:", label_file)
    print("=" * 80)

    try:
        with open(label_file, "rb") as f:
            data = pickle.load(f)
    except Exception:
        with open(label_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")

    print("라벨 데이터 자료형:", type(data))

    if isinstance(data, dict):
        print("딕셔너리 형태입니다.")
        print("전체 key 개수:", len(data))

        keys = list(data.keys())

        print()
        print("앞부분 20개 출력:")
        for key in keys[:20]:
            print("KEY:", key)
            print("VALUE:", data[key])
            print("-" * 40)

    elif isinstance(data, list):
        print("리스트 형태입니다.")
        print("전체 길이:", len(data))

        print()
        print("앞부분 20개 출력:")
        for item in data[:20]:
            print(item)

    elif isinstance(data, tuple):
        print("튜플 형태입니다.")
        print("전체 길이:", len(data))

        print()
        print("앞부분 출력:")
        for i, item in enumerate(data[:10]):
            print("INDEX:", i)
            print("TYPE:", type(item))
            print("VALUE:", str(item)[:1000])
            print("-" * 40)

    else:
        print("기타 형태입니다.")
        print(str(data)[:3000])
