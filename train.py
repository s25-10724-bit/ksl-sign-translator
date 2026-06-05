import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from extract_landmarks import extract_video_landmarks
from config import DEFAULT_MAX_FRAMES


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_dataset(data_dir, max_frames):
    data_dir = Path(data_dir)

    label_dirs = sorted([
        path for path in data_dir.iterdir()
        if path.is_dir()
    ])

    if not label_dirs:
        raise ValueError("dataset/raw 안에 라벨 폴더가 없습니다.")

    labels = [path.name for path in label_dirs]
    label_to_id = {label: index for index, label in enumerate(labels)}

    X = []
    y = []

    for label_dir in label_dirs:
        video_files = sorted([
            path for path in label_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ])

        print(f"[{label_dir.name}] 영상 {len(video_files)}개 처리 중")

        for video_path in video_files:
            try:
                sequence = extract_video_landmarks(video_path, max_frames)
                X.append(sequence.reshape(-1))
                y.append(label_to_id[label_dir.name])
            except Exception as error:
                print(f"건너뜀: {video_path} / 이유: {error}")

    if len(X) == 0:
        raise ValueError("학습할 영상 데이터가 없습니다.")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y, labels


def build_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.35),

        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_labels(labels, output_path):
    output_path.write_text("\n".join(labels), encoding="utf-8")


def export_tflite(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="dataset/raw")
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, labels = load_dataset(args.data_dir, args.max_frames)

    print("전체 샘플 수:", len(X))
    print("클래스 수:", len(labels))
    print("입력 차원:", X.shape[1])

    class_counts = np.bincount(y)

    if len(class_counts) > 0 and min(class_counts) >= 2:
        stratify = y
    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    np.savez(
        output_dir / "scaler.npz",
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32)
    )

    save_labels(labels, output_dir / "labels.txt")

    model = build_model(
        input_dim=X_train.shape[1],
        num_classes=len(labels)
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred = model.predict(X_test).argmax(axis=1)

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

    model.save(output_dir / "ksl_model.keras")
    export_tflite(model, output_dir / "ksl_model.tflite")

    print()
    print("모델 저장 완료")
    print(output_dir / "ksl_model.keras")
    print(output_dir / "ksl_model.tflite")
    print(output_dir / "labels.txt")
    print(output_dir / "scaler.npz")


if __name__ == "__main__":
    main()
