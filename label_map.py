DISPLAY_LABELS = {
    "hi": "안녕하세요",
    "thank": "감사합니다",
    "sorry": "미안합니다",
}


def to_display_label(label):
    return DISPLAY_LABELS.get(label, label)
