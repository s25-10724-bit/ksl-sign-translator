# 수어 영상 데이터셋을 학습하여 KSL 단어 분류 모델을 만드는 파일
# 학습 후 Keras 모델과 TensorFlow Lite 모델 저장

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from extract_landmarks import extract_video_landmarks
from config import DEFAULT_MAX_FRAMES


# 학습에 사용할 수 있는 영상 확장자 목록
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_dataset(data_dir, max_frames):
    # dataset/raw/라벨명/*.mp4 구조 데이터 읽기

    data_dir = Path(data_dir)

    # data_dir 안의 폴더를 라벨 폴더로 인식
    label_dirs = sorted([
        path for path in data_dir.iterdir()
        if path.is_dir()
    ])

    # 라벨 폴더가 없는 경우의 예외 처리
    if not label_dirs:
        raise ValueError("dataset/raw 안에 라벨 폴더가 없습니다.")

    # 폴더 이름을 라벨 이름으로 사용
    labels = [path.name for path in label_dirs]

    # 라벨 이름을 숫자로 바꾸기 위한 딕셔너리
    label_to_id = {label: index for index, label in enumerate(labels)}

    # 입력 데이터와 정답 라벨 저장 리스트
    X = []
    y = []

    # 각 라벨 폴더 처리
    for label_dir in label_dirs:
        # 라벨 폴더 안의 영상 파일 검색
        video_files = sorted([
            path for path in label_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ])

        print(f"[{label_dir.name}] 영상 {len(video_files)}개 처리 중")

        # 각 영상의 랜드마크 추출
        for video_path in video_files:
            try:
                # 영상에서 손 랜드마크 시퀀스 추출
                sequence = extract_video_landmarks(video_path, max_frames)

                # 모델 입력용 1차원 배열 변환
                X.append(sequence.reshape(-1))

                # 정답 라벨 숫자 저장
                y.append(label_to_id[label_dir.name])

            except Exception as error:
                # 오류 영상 건너뛰기
                print(f"건너뜀: {video_path} / 이유: {error}")

    # 학습 데이터가 없는 경우의 예외 처리
    if len(X) == 0:
        raise ValueError("학습할 영상 데이터가 없습니다.")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y, labels


def build_model(input_dim, num_classes):
    # KSL 단어 분류 신경망 모델 생성

    model = tf.keras.Sequential([
        # 입력층
        tf.keras.layers.Input(shape=(input_dim,)),

        # 첫 번째 은닉층
        tf.keras.layers.Dense(512, activation="relu"),

        # 과적합 방지용 Dropout
        tf.keras.layers.Dropout(0.35),

        # 두 번째 은닉층
        tf.keras.layers.Dense(256, activation="relu"),

        # 과적합 방지용 Dropout
        tf.keras.layers.Dropout(0.25),

        # 클래스별 확률 출력층
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    # 모델 학습 방식 설정
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_labels(labels, output_path):
    # 라벨 이름 목록 txt 저장
    output_path.write_text("\n".join(labels), encoding="utf-8")


def export_tflite(model, output_path):
    # Keras 모델을 TensorFlow Lite 모델로 변환

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 모델 크기 감소를 위한 기본 최적화
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # TFLite 모델 변환
    tflite_model = converter.convert()

    # TFLite 모델 파일 저장
    output_path.write_bytes(tflite_model)


def main():
    # 명령어 옵션 처리 객체
    parser = argparse.ArgumentParser()

    # 데이터셋 경로
    parser.add_argument("--data_dir", default="dataset/raw")

    # 모델 저장 폴더
    parser.add_argument("--output_dir", default="models")

    # 영상당 사용할 프레임 수
    parser.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)

    # 학습 반복 횟수
    parser.add_argument("--epochs", type=int, default=40)

    # 한 번에 학습할 데이터 개수
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    # 출력 폴더 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터셋 불러오기
    X, y, labels = load_dataset(args.data_dir, args.max_frames)

    print("전체 샘플 수:", len(X))
    print("클래스 수:", len(labels))
    print("입력 차원:", X.shape[1])

    # 클래스별 데이터 개수 확인
    class_counts = np.bincount(y)

    # 클래스별 데이터가 2개 이상인 경우의 stratify 사용
    if len(class_counts) > 0 and min(class_counts) >= 2:
        stratify = y
    else:
        stratify = None

    # 학습 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    # 입력값 표준화 객체
    scaler = StandardScaler()

    # 학습 데이터 기준 평균과 표준편차 계산
    X_train = scaler.fit_transform(X_train).astype(np.float32)

    # 테스트 데이터 표준화
    X_test = scaler.transform(X_test).astype(np.float32)

    # 실시간 실행용 표준화 정보 저장
    np.savez(
        output_dir / "scaler.npz",
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32)
    )

    # 라벨 목록 저장
    save_labels(labels, output_dir / "labels.txt")

    # 모델 생성
    model = build_model(
        input_dim=X_train.shape[1],
        num_classes=len(labels)
    )

    # 검증 정확도 기준 조기 종료 설정
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True
    )

    # 모델 학습
    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    # 테스트 데이터 예측
    y_pred = model.predict(X_test).argmax(axis=1)

    # 정확도 계산
    accuracy = accuracy_score(y_test, y_pred)

    print()
    print("테스트 정확도:", round(accuracy * 100, 2), "%")
    print()
    print("분류 리포트")
    print(classification_report(
        y_test,
        y_pred,
        target_names=labels,
        zero_division=0
    ))
    print()
    print("혼동 행렬")
    print(confusion_matrix(y_test, y_pred))

    # Keras 모델 저장
    model.save(output_dir / "ksl_model.keras")

    # TensorFlow Lite 모델 저장
    export_tflite(model, output_dir / "ksl_model.tflite")

    print()
    print("모델 저장 완료")
    print(output_dir / "ksl_model.keras")
    print(output_dir / "ksl_model.tflite")
    print(output_dir / "labels.txt")
    print(output_dir / "scaler.npz")


if __name__ == "__main__":
    main()
