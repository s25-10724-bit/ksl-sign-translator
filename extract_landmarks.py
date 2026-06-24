# 영상 또는 웹캠 프레임에서 손 랜드마크를 추출하는 파일
# MediaPipe Hands를 사용하여 손가락 관절 좌표를 뽑아

import cv2
import numpy as np

from config import (
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    FEATURES_PER_FRAME,
)
import mediapipe as mp

try:
    mp_hands = mp.solutions.hands
except AttributeError:
    from mediapipe.python.solutions import hands as mp_hands    # MediaPipe Hands 모듈


def empty_hand():    # 손이 감지되지 않았을 때 사용할 빈 좌표
    return np.zeros((21, 3), dtype=np.float32)    # 손 하나는 21개 관절, 각 관절은 x, y, z 좌표를 가짐


def landmark_to_array(hand_landmarks):    # MediaPipe가 반환한 손 관절 정보를 numpy 배열로 바
    coords = []

    for lm in hand_landmarks.landmark:    # 손의 21개 관절 좌표를 하나씩 저장
        coords.append([lm.x, lm.y, lm.z])

    return np.array(coords, dtype=np.float32)


def normalize_hand(hand):    # 손 위치와 크기가 달라도 비슷하게 인식되도록 정규화
    if np.allclose(hand, 0):    # 손이 없는 경우에는 그대로 반환
        return hand

    wrist = hand[0].copy()    # 0번 관절(손목)
    hand = hand - wrist    # 손목을 기준점으로 하여 모든 좌표를 이동

    scale = np.linalg.norm(hand[9])    # 손목에서 9번 관절(중지 아래쪽 관절)까지의 거리를 손 크기의 기준으로 사용

    if scale < 1e-6:    # scale이 너무 작으면 1로 처리합니다.
        scale = 1.0

    return hand / scale    # 손 크기 차이를 줄이기 위해 좌표를 scale로 나


def extract_frame_landmarks(image_bgr, hands):    # 웹캠 프레임 한 장에서 양손 랜드마크를 추출(반환값은 길이 126짜리 배열)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)    # MediaPipe로 손을 인식

    left = empty_hand()    # 왼손과 오른손 기본값을 빈 좌표로 설정
    right = empty_hand()

    if result.multi_hand_landmarks and result.multi_handedness:    # 손이 감지되었을 때만 좌표를 저장
        for hand_landmarks, handedness in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):
            label = handedness.classification[0].label    # 감지된 손이 왼손인지 오른손인지 확인.
            hand_array = landmark_to_array(hand_landmarks)    # 손 관절 좌표를 배열로 변환   
            hand_array = normalize_hand(hand_array)    # 좌표를 정규화

            if label == "Left":    # 왼손과 오른손을 구분하여 저장   
                left = hand_array
            else:
                right = hand_array

    frame_features = np.concatenate([    # 왼손 좌표와 오른손 좌표를 하나의 긴 배열로 합침
        left.reshape(-1),
        right.reshape(-1)
    ])

    return frame_features.astype(np.float32)


def resample_or_pad(sequence, max_frames):    # 영상마다 프레임 수가 다르기 때문에 길이를 동일하게 맞

    if len(sequence) == 0:    # 프레임이 하나도 없으면 전부 0으로 채운 배열을 반환
        return np.zeros((max_frames, FEATURES_PER_FRAME), dtype=np.float32)

    sequence = np.asarray(sequence, dtype=np.float32)

    if len(sequence) >= max_frames:    # 영상이 max_frames보다 길면 일정 간격으로 프레임을 골라.    
        indices = np.linspace(0, len(sequence) - 1, max_frames).astype(int)
        return sequence[indices]

    padded = np.zeros((max_frames, FEATURES_PER_FRAME), dtype=np.float32)    # 영상이 max_frames보다 짧으면 부족한 부분을 0으로 채움

    padded[:len(sequence)] = sequence

    return padded


def extract_video_landmarks(video_path, max_frames):    # 영상 파일 하나에서 손 랜드마크 시퀀스를 추출
    cap = cv2.VideoCapture(str(video_path))      # OpenCV로 영상 파일을 
    frames = []    # 각 프레임의 랜드마크를 저장할 리스트

    with mp_hands.Hands(     # MediaPipe Hands 객체를 생성
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:
        while True:
            success, frame = cap.read()     # 영상에서 프레임을 한 장씩 읽

            if not success:       # 더 이상 읽을 프레임이 없으면 종료
                break

            feature = extract_frame_landmarks(frame, hands)    # 현재 프레임의 손 랜드마크를 추출
            frames.append(feature)    # 추출한 특징을 리스트에 저장

    cap.release()     # 영상 파일을 닫음

    return resample_or_pad(np.array(frames, dtype=np.float32), max_frames)       # 프레임 수를 max_frames로 맞춘 뒤 반환
