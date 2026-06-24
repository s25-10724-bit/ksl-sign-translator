import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tensorflow as tf

from config import (
    DEFAULT_MAX_FRAMES,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    CONFIDENCE_THRESHOLD,
    STABLE_COUNT,
)

from label_map import to_display_label


# MediaPipe 버전 호환 처리
try:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
except AttributeError:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_draw


MODEL_PATH = Path("models/ksl_model.tflite")
LABEL_PATH = Path("models/labels.txt")
SCALER_PATH = Path("models/scaler.npz")


def empty_hand():
    return np.zeros((21, 3), dtype=np.float32)


def landmark_to_array(hand_landmarks):
    points = []

    for landmark in hand_landmarks.landmark:
        points.append([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return np.array(points, dtype=np.float32)


def normalize_hand(hand_array):
    wrist = hand_array[0].copy()
    centered = hand_array - wrist

    scale = np.linalg.norm(centered[9])

    if scale < 1e-6:
        scale = 1.0

    normalized = centered / scale

    return normalized.astype(np.float32)


def extract_features_from_result(result):
    hands = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks[:MAX_NUM_HANDS]:
            hand_array = landmark_to_array(hand_landmarks)
            hand_array = normalize_hand(hand_array)
            hands.append(hand_array)

    while len(hands) < MAX_NUM_HANDS:
        hands.append(empty_hand())

    hands = hands[:MAX_NUM_HANDS]

    feature = np.concatenate(hands, axis=0).reshape(-1)

    return feature.astype(np.float32)


def resample_or_pad(sequence, max_frames):
    sequence = np.asarray(sequence, dtype=np.float32)

    if len(sequence) == 0:
        return np.zeros((max_frames, MAX_NUM_HANDS * 21 * 3), dtype=np.float32)

    if len(sequence) == max_frames:
        return sequence

    if len(sequence) > max_frames:
        indices = np.linspace(0, len(sequence) - 1, max_frames).astype(int)
        return sequence[indices]

    pad_count = max_frames - len(sequence)
    padding = np.zeros((pad_count, sequence.shape[1]), dtype=np.float32)

    return np.concatenate([sequence, padding], axis=0)


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
    sequence = resample_or_pad(sequence, max_frames=max_frames)

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
        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        return output[0]


def check_model_files():
    missing_files = []

    if not MODEL_PATH.exists():
        missing_files.append(str(MODEL_PATH))

    if not LABEL_PATH.exists():
        missing_files.append(str(LABEL_PATH))

    if not SCALER_PATH.exists():
        missing_files.append(str(SCALER_PATH))

    return missing_files


def apply_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 45%, #ecfeff 100%);
        }

        .main-title {
            font-size: 44px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 8px;
        }

        .sub-title {
            font-size: 18px;
            color: #475569;
            margin-bottom: 28px;
        }

        .card {
            background: rgba(255, 255, 255, 0.9);
            padding: 24px;
            border-radius: 22px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(226, 232, 240, 0.9);
            margin-bottom: 18px;
        }

        .result-card {
            background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%);
            padding: 30px;
            border-radius: 24px;
            color: white;
            text-align: center;
            box-shadow: 0 12px 36px rgba(37, 99, 235, 0.25);
            margin-bottom: 18px;
        }

        .result-label {
            font-size: 18px;
            opacity: 0.9;
            margin-bottom: 8px;
        }

        .result-text {
            font-size: 48px;
            font-weight: 900;
            line-height: 1.2;
        }

        .small-muted {
            color: #64748b;
            font-size: 15px;
            line-height: 1.6;
        }

        .word-chip {
            display: inline-block;
            background: #e0f2fe;
            color: #075985;
            padding: 8px 14px;
            border-radius: 999px;
            margin: 4px;
            font-weight: 700;
            font-size: 15px;
        }

        .status-good {
            color: #059669;
            font-weight: 800;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_header():
    st.markdown(
        """
        <div class="main-title">KSL 수어 번역 대시보드</div>
        <div class="sub-title">
            웹캠으로 입력된 한국 수어 동작을 인식하고 한국어 텍스트로 변환합니다.
        </div>
        """,
        unsafe_allow_html=True
    )


def render_word_chips(labels):
    chips = ""

    for label in labels:
        chips += f'<span class="word-chip">{to_display_label(label)}</span>'

    st.markdown(
        f"""
        <div class="card">
            <h3>인식 가능한 단어</h3>
            <div>{chips}</div>
            <p class="small-muted">
                현재 모델은 3개 단어를 대상으로 학습된 프로토타입입니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_result_card(label):
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">확정 인식 결과</div>
            <div class="result-text">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(
        page_title="KSL 수어 번역 대시보드",
        page_icon="🖐️",
        layout="wide"
    )

    apply_style()
    render_header()

    missing_files = check_model_files()

    if missing_files:
        st.error("모델 파일이 없습니다.")
        st.write("아래 파일들이 models 폴더 안에 있어야 합니다.")

        for file in missing_files:
            st.write("-", file)

        return

    labels = load_labels(LABEL_PATH)
    mean, scale = load_scaler(SCALER_PATH)
    classifier = TFLiteClassifier(MODEL_PATH)

    left_col, right_col = st.columns([1.6, 1])

    with right_col:
        st.markdown(
            """
            <div class="card">
                <h3>모델 상태</h3>
                <p class="status-good">모델 파일 정상 로드 완료</p>
                <p class="small-muted">
                    TensorFlow Lite 모델, 라벨 파일, 스케일러 파일을 정상적으로 불러왔습니다.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        render_word_chips(labels)

        st.markdown(
            """
            <div class="card">
                <h3>사용 방법</h3>
                <p class="small-muted">
                1. 실시간 인식 시작 버튼을 누릅니다.<br>
                2. 웹캠 앞에서 손이 잘 보이도록 합니다.<br>
                3. 안녕하세요, 감사합니다, 미안합니다 동작을 천천히 수행합니다.<br>
                4. 종료하려면 실행 창에서 Ctrl + C를 누릅니다.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with left_col:
        st.markdown(
            """
            <div class="card">
                <h3>실시간 웹캠 화면</h3>
                <p class="small-muted">
                    MediaPipe가 손의 랜드마크를 추출하고, 학습된 모델이 수어 단어를 예측합니다.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        start_button = st.button("실시간 인식 시작", use_container_width=True)

        video_area = st.empty()
        waiting_area = st.empty()

    if not start_button:
        with left_col:
            waiting_area.info("버튼을 누르면 웹캠 인식이 시작됩니다.")
        with right_col:
            render_result_card("대기 중")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다. 카메라 연결 또는 권한을 확인하세요.")
        return

    result_area = st.empty()
    confidence_area = st.empty()
    history_area = st.empty()

    sequence = []

    last_label = None
    stable_counter = 0

    confirmed_label = "대기 중"
    confirmed_confidence = 0.0
    history = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:

        while True:
            ok, frame = cap.read()

            if not ok:
                st.error("웹캠 화면을 읽을 수 없습니다.")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            feature = extract_features_from_result(result)
            sequence.append(feature)

            if len(sequence) > DEFAULT_MAX_FRAMES:
                sequence = sequence[-DEFAULT_MAX_FRAMES:]

            current_label = "분석 중"
            current_confidence = 0.0

            if len(sequence) >= max(5, DEFAULT_MAX_FRAMES // 3):
                x = preprocess_sequence(
                    sequence,
                    DEFAULT_MAX_FRAMES,
                    mean,
                    scale
                )

                probabilities = classifier.predict(x)

                predicted_index = int(np.argmax(probabilities))
                current_confidence = float(probabilities[predicted_index])

                raw_label = labels[predicted_index]
                display_label = to_display_label(raw_label)

                if current_confidence >= CONFIDENCE_THRESHOLD:
                    current_label = display_label

                    if raw_label == last_label:
                        stable_counter += 1
                    else:
                        stable_counter = 1
                        last_label = raw_label

                    if stable_counter >= STABLE_COUNT:
                        confirmed_label = display_label
                        confirmed_confidence = current_confidence

                        if not history or history[-1] != confirmed_label:
                            history.append(confirmed_label)

                        if len(history) > 8:
                            history = history[-8:]

                else:
                    current_label = "확신 부족"
                    stable_counter = 0
                    last_label = None

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with left_col:
                video_area.image(
                    frame_rgb,
                    channels="RGB",
                    use_container_width=True
                )

            with right_col:
                with result_area.container():
                    render_result_card(confirmed_label)

                confidence_area.markdown(
                    f"""
                    <div class="card">
                        <h3>신뢰도</h3>
                        <p class="small-muted">
                            현재 예측: <b>{current_label}</b><br>
                            현재 신뢰도: <b>{current_confidence:.2f}</b><br>
                            확정 신뢰도: <b>{confirmed_confidence:.2f}</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if history:
                    history_text = " → ".join(history)
                else:
                    history_text = "아직 인식 기록 없음"

                history_area.markdown(
                    f"""
                    <div class="card">
                        <h3>최근 인식 기록</h3>
                        <p class="small-muted">{history_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            time.sleep(0.03)

    cap.release()


if __name__ == "__main__":
    main()
