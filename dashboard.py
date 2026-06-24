# Streamlit 기반 KSL 수어 번역 대시보드
# 웹캠 영상, 현재 인식 결과, 예측 확률, 최근 인식 기록 표시

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import mediapipe as mp

from config import (
    DEFAULT_MAX_FRAMES,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    CONFIDENCE_THRESHOLD,
    STABLE_COUNT,
)

from extract_landmarks import extract_frame_landmarks, resample_or_pad


# 학습된 모델 파일 경로
MODEL_PATH = Path("models/ksl_model.tflite")

# 라벨 파일 경로
LABEL_PATH = Path("models/labels.txt")

# 표준화 정보 파일 경로
SCALER_PATH = Path("models/scaler.npz")


def load_labels(label_path):
    # 라벨 목록 불러오기
    labels = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    return labels


def load_scaler(scaler_path):
    # 학습 때 저장한 평균과 표준편차 불러오기
    data = np.load(scaler_path)

    mean = data["mean"].astype(np.float32)
    scale = data["scale"].astype(np.float32)

    # 0으로 나누는 오류 방지
    scale[scale == 0] = 1.0

    return mean, scale


def preprocess_sequence(sequence, max_frames, mean, scale):
    # 프레임 시퀀스의 모델 입력 형태 변환
    sequence = resample_or_pad(
        np.asarray(sequence, dtype=np.float32),
        max_frames=max_frames
    )

    # 1차원 입력 벡터 변환
    x = sequence.reshape(1, -1)

    # 학습 기준 표준화 적용
    x = (x - mean) / scale

    return x.astype(np.float32)


class TFLiteClassifier:
    # TensorFlow Lite 모델 실행 클래스

    def __init__(self, model_path):
        # TFLite 모델 불러오기
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))

        # 텐서 할당
        self.interpreter.allocate_tensors()

        # 입력 정보 저장
        self.input_details = self.interpreter.get_input_details()

        # 출력 정보 저장
        self.output_details = self.interpreter.get_output_details()

    def predict(self, x):
        # 입력 데이터 설정
        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            x
        )

        # 모델 실행
        self.interpreter.invoke()

        # 출력값 반환
        output = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        return output[0]


def check_model_files():
    # 모델 파일 존재 여부 확인
    if not MODEL_PATH.exists():
        st.error("models/ksl_model.tflite 파일이 없습니다. 먼저 train.py로 학습하세요.")
        st.stop()

    if not LABEL_PATH.exists():
        st.error("models/labels.txt 파일이 없습니다. 먼저 train.py로 학습하세요.")
        st.stop()

    if not SCALER_PATH.exists():
        st.error("models/scaler.npz 파일이 없습니다. 먼저 train.py로 학습하세요.")
        st.stop()


def main():
    # 대시보드 기본 설정
    st.set_page_config(
        page_title="KSL 수어 번역 대시보드",
        page_icon="🤟",
        layout="wide"
    )

    # 제목 영역
    st.title("KSL 수어 번역 대시보드")
    st.write("웹캠으로 입력된 한국 수어를 인식하여 텍스트로 출력하는 대시보드")

    # 모델 파일 확인
    check_model_files()

    # 라벨, 표준화 정보, 모델 불러오기
    labels = load_labels(LABEL_PATH)
    mean, scale = load_scaler(SCALER_PATH)
    classifier = TFLiteClassifier(MODEL_PATH)

    # 세션 상태 초기화
    if "history" not in st.session_state:
        st.session_state.history = []

    # 화면 영역 분할
    left_col, right_col = st.columns([2, 1])

    with right_col:
        st.subheader("모델 정보")
        st.write(f"인식 가능 단어 수: {len(labels)}개")
        st.write(f"사용 프레임 수: {DEFAULT_MAX_FRAMES}")
        st.write(f"예측 확률 기준값: {CONFIDENCE_THRESHOLD}")

        st.subheader("인식 가능 단어")
        st.write(", ".join(labels))

        st.warning("대시보드 실행 중지는 터미널에서 Ctrl + C")

    with left_col:
        st.subheader("웹캠 화면")
        video_area = st.empty()

    result_area = right_col.empty()
    confidence_area = right_col.empty()
    history_area = right_col.empty()

    # 웹캠 열기
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("웹캠을 열 수 없습니다.")
        return

    # 최근 프레임 랜드마크 저장 리스트
    sequence = []

    # 이전 예측 라벨
    last_label = None

    # 같은 라벨 연속 인식 횟수
    stable_counter = 0

    # 최종 확정 라벨
    confirmed_label = "인식 대기 중"

    # 예측 확률 초기값
    confidence = 0.0

    # MediaPipe Hands와 그리기 도구
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # MediaPipe Hands 객체 생성
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:
        while True:
            # 웹캠 프레임 읽기
            success, frame = cap.read()

            # 프레임 읽기 실패 시 종료
            if not success:
                st.error("웹캠 프레임을 읽을 수 없습니다.")
                break

            # 사용자 확인용 좌우 반전
            frame = cv2.flip(frame, 1)

            # 손 관절 표시용 RGB 변환
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 손 감지
            draw_result = hands.process(rgb)

            # 감지된 손 관절 화면 표시
            if draw_result.multi_hand_landmarks:
                for hand_landmarks in draw_result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            # 현재 프레임 랜드마크 추출
            feature = extract_frame_landmarks(frame, hands)

            # 최근 프레임 리스트 추가
            sequence.append(feature)

            # 최대 프레임 수 유지
            if len(sequence) > DEFAULT_MAX_FRAMES:
                sequence = sequence[-DEFAULT_MAX_FRAMES:]

            # 일정 프레임 이상 모인 뒤 예측 시작
            if len(sequence) >= max(5, DEFAULT_MAX_FRAMES // 3):
                # 모델 입력 전처리
                x = preprocess_sequence(
                    sequence,
                    DEFAULT_MAX_FRAMES,
                    mean,
                    scale
                )

                # 모델 예측
                probabilities = classifier.predict(x)

                # 가장 높은 확률의 라벨 선택
                predicted_index = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_index])
                predicted_label = labels[predicted_index]

                # 기준 확률 이상인 경우
                if confidence >= CONFIDENCE_THRESHOLD:
                    # 같은 라벨 연속 인식 확인
                    if predicted_label == last_label:
                        stable_counter += 1
                    else:
                        stable_counter = 1
                        last_label = predicted_label

                    # 안정적으로 인식된 경우 최종 확정
                    if stable_counter >= STABLE_COUNT:
                        confirmed_label = predicted_label

                        # 최근 인식 기록 추가
                        if (
                            len(st.session_state.history) == 0
                            or st.session_state.history[-1] != confirmed_label
                        ):
                            st.session_state.history.append(confirmed_label)

                        # 최근 10개만 유지
                        if len(st.session_state.history) > 10:
                            st.session_state.history = st.session_state.history[-10:]

                else:
                    # 기준 확률 미만일 때의 보류 처리
                    confirmed_label = "확신 부족"
                    stable_counter = 0
                    last_label = None

            # 화면 상단 검은 배경
            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 80),
                (0, 0, 0),
                -1
            )

            # 영상 위 예측 결과 표시
            cv2.putText(
                frame,
                f"{confirmed_label} / {confidence:.2f}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Streamlit 출력용 RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 웹캠 화면 표시
            video_area.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )

            # 현재 인식 결과 표시
            result_area.metric(
                label="현재 인식 결과",
                value=confirmed_label
            )

            # 예측 확률 표시
            confidence_area.progress(
                min(max(confidence, 0.0), 1.0),
                text=f"예측 확률: {confidence:.2f}"
            )

            # 최근 인식 기록 표시
            history_area.write("최근 인식 기록")
            history_area.write(st.session_state.history[::-1])

            # 프레임 속도 조절
            time.sleep(0.03)

    # 웹캠 종료
    cap.release()


if __name__ == "__main__":
    main()
3. .gitignore 수정 또는 확인

.gitignore에 이 내용이 있으면 돼.

__pycache__/
*.pyc
.venv/
venv/

models/*.keras
models/*.tflite
models/*.npz
models/labels.txt

dataset/raw/**/*.mp4
dataset/raw/**/*.avi
dataset/raw/**/*.mov
dataset/raw/**/*.mkv
dataset/raw/**/*.webm
