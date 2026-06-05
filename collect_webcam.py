# 웹캠으로 직접 수어 영상을 촬영하여 데이터셋을 만드는 파일
# 촬영한 영상은 dataset/raw/라벨명/ 폴더 안에 저장

import argparse    
import time
from pathlib import Path

import cv2


def main():    # 명령어 옵션을 받기 위한 parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--label", required=True)    # 저장할 수어 단어 이름
    parser.add_argument("--save_dir", default="dataset/raw")     # 데이터가 저장될 기본 폴더.
    parser.add_argument("--seconds", type=float, default=3.0)    # 영상 하나당 촬영 시간
    parser.add_argument("--count", type=int, default=20)    # 촬영할 영상 개수
    parser.add_argument("--camera", type=int, default=0)     # 사용할 웹캠 번호

    args = parser.parse_args()

    label_dir = Path(args.save_dir) / args.label    # 라벨 이름으로 폴더를 만
    label_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)     # 웹캠을 

    if not cap.isOpened():        # 웹캠이 열리지 않으면 오류를 발생
        raise RuntimeError("웹캠을 열 수 없습니다.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640        # 웹캠 영상의 가로, 세로 크기를 가져.
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = 30      # 저장할 영상의 FPS

    saved_count = 0     # 현재까지 저장한 영상 개수

    print("스페이스바를 누르면 녹화가 시작됩니다.")
    print("q를 누르면 종료됩니다.")

    while saved_count < args.count:
        success, frame = cap.read()    # 웹캠에서 프레임을 읽

        if not success:        # 프레임을 읽지 못하면 반복을 종료
            break

        frame = cv2.flip(frame, 1)    # 사용자가 보기 편하도록 좌우 반전

        cv2.putText(     # 화면에 현재 라벨과 저장 개수를 표시
            frame,
            f"Label: {args.label} / Saved: {saved_count}/{args.count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(      # 화면에 조작 방법을 표시
            frame,
            "Press SPACE to record, q to quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Collect KSL Data", frame)       # 웹캠 화면을 보여줌

        key = cv2.waitKey(1) & 0xFF     # 키 입력을 확인함

        if key == ord("q"):     # q를 누르면 종료
            break

        if key == 32:      # 스페이스바를 누르면 녹화를 시작
            file_name = f"{args.label}_{int(time.time())}_{saved_count + 1:03d}.mp4"     # 저장할 파일 이름을 만
            save_path = label_dir / file_name

            writer = cv2.VideoWriter(    # 영상 저장 객체를 만
                str(save_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            start_time = time.time()    # 녹화 시작 시간을 저장

            while time.time() - start_time < args.seconds:      # 지정한 시간 동안 녹화
                success, frame = cap.read()

                if not success:
                    break

                frame = cv2.flip(frame, 1)
                writer.write(frame)     # 현재 프레임을 영상 파일에 저장

                remaining_time = args.seconds - (time.time() - start_time)     # 남은 시간을 계산.

                cv2.putText(    # 화면에 녹화 중임을 표시
                    frame,
                    f"Recording... {remaining_time:.1f}s",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("Collect KSL Data", frame)
                cv2.waitKey(1)

            writer.release()            # 영상 저장을 종료

            saved_count += 1      # 저장 개수를 1 증가

            print("저장 완료:", save_path)

    cap.release()     # 웹캠과 창을 닫음
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
