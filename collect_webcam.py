import argparse
import time
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--label", required=True)
    parser.add_argument("--save_dir", default="dataset/raw")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--camera", type=int, default=0)

    args = parser.parse_args()

    label_dir = Path(args.save_dir) / args.label
    label_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = 30

    saved_count = 0

    print("스페이스바를 누르면 녹화가 시작됩니다.")
    print("q를 누르면 종료됩니다.")

    while saved_count < args.count:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            f"Label: {args.label} / Saved: {saved_count}/{args.count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press SPACE to record, q to quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Collect KSL Data", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 32:
            file_name = f"{args.label}_{int(time.time())}_{saved_count + 1:03d}.mp4"
            save_path = label_dir / file_name

            writer = cv2.VideoWriter(
                str(save_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            start_time = time.time()

            while time.time() - start_time < args.seconds:
                success, frame = cap.read()

                if not success:
                    break

                frame = cv2.flip(frame, 1)
                writer.write(frame)

                remaining_time = args.seconds - (time.time() - start_time)

                cv2.putText(
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

            writer.release()

            saved_count += 1

            print("저장 완료:", save_path)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
