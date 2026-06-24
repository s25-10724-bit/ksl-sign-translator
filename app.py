import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from config import (
    DEFAULT_MAX_FRAMES,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    CONFIDENCE_THRESHOLD,
    STABLE_COUNT,
)

from extract_landmarks import extract_frame_landmarks, resample_or_pad
from label_map import to_display_label


def load_labels(path):
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            label = line.strip()

            if label:
                labels.append(label)

    return labels


def load_scaler(path):
    data = np.load(path)

    mean = data["mean"].astype(np.float32)
    scale = data["scale"].astype(np.float32)

    scale[scale == 0] = 1.0

    return mean, scale


def preprocess_sequence(sequence, max_frames, mean, scale):
    sequence = np.asarray(sequence, dtype=np.float32)

    sequence = resample_or_pad(
        sequence,
        max_frames=max_frames
    )

    x = sequence.reshape(1, -1)

    x = (x - mean) / scale

    return x.astype(np.float32)


class TFLiteClassifier:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, x):
        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            x
        )

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        return output[0]


def load_korean_font(size=36):
    font_candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]

    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)

    return ImageFont.load_default()


def draw_korean_text(frame, text, position=(20, 35), font_size=36):
    font = load_korean_font(font_size)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    draw.text(
        position,
        text,
        font=font,
        fill=(255, 255, 255)
    )

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def init_tts():
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        return engine
    except Exception:
        return None


def speak(engine, text):
    if engine is None:
        return

    try:
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="models/ksl_model.tflite"
    )

    parser.add_argument(
        "--labels",
        default="models/labels.txt"
    )

    parser.add_argument(
        "--scaler",
        default="models/scaler.npz"
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=DEFAULT_MAX_FRAMES
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    labels_path = Path(args.labels)
    scaler_path = Path(args.scaler)

    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

    if not labels_path.exists():
        raise FileNotFoundError(f"라벨 파일이 없습니다: {labels_path}")

    if not scaler_path.exists():
        raise FileNotFoundError(f"스케일러 파일이 없습니다: {scaler_path}")

    labels = load_labels(labels_path)
    mean, scale = load_scaler(scaler_path)
    classifier = TFLiteClassifier(model_path)

    tts_engine = init_tts()

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    sequence = []

    last_label = None
    stable_counter = 0

    confirmed_label = "인식 대기 중"
    spoken_label = None
    last_speak_time = 0.0

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    print("KSL 수어 번역 프로그램 실행 중")
    print("종료하려면 q 키를 누르세요.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            draw_result = hands.process(rgb)

            if draw_result.multi_hand_landmarks:
                for hand_landmarks in draw_result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            feature = extract_frame_landmarks(frame, hands)

            sequence.append(feature)

            if len(sequence) > args.max_frames:
                sequence = sequence[-args.max_frames:]

            current_label = "인식 대기 중"
            current_confidence = 0.0

            if len(sequence) >= max(5, args.max_frames // 3):
                x = preprocess_sequence(
                    sequence,
                    args.max_frames,
                    mean,
                    scale
                )

                probabilities = classifier.predict(x)

                predicted_index = int(np.argmax(probabilities))
                current_confidence = float(probabilities[predicted_index])
                raw_label = labels[predicted_index]
                current_label = to_display_label(raw_label)

                if current_confidence >= CONFIDENCE_THRESHOLD:
                    if raw_label == last_label:
                        stable_counter += 1
                    else:
                        stable_counter = 1
                        last_label = raw_label

                    if stable_counter >= STABLE_COUNT:
                        confirmed_label = current_label

                        now = time.time()

                        if (
                            confirmed_label != spoken_label
                            and now - last_speak_time > 2.0
                        ):
                            spoken_label = confirmed_label
                            last_speak_time = now
                            speak(tts_engine, confirmed_label)

                else:
                    stable_counter = 0
                    last_label = None
                    current_label = "확신 부족"

            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 90),
                (0, 0, 0),
                -1
            )

            display_text = f"현재: {current_label} / 확정: {confirmed_label} / 신뢰도: {current_confidence:.2f}"

            frame = draw_korean_text(
                frame,
                display_text,
                position=(20, 25),
                font_size=32
            )

            cv2.imshow(
                "KSL Sign Translator",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
