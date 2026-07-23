# src/ner/lexicon.py

from __future__ import annotations

MEDICATIONS = {
    "aspirin",
    "amlodipine",
    "metoprolol",
    "metoprolol succinate",
    "guaifenesin",
    "nystatin",
    "acetaminophen",
    "pravastatin",
    "docusate sodium",
    "senna",
    "clonazepam",
    "doxycycline",
    "atenolol",
    "amoxicillin",
    "ibuprofen",
    "paracetamol",
    "warfarin",
    "heparin",
    "insulin",
    "atorvastatin",
    "simvastatin",
    "losartan",
    "lisinopril",
    "furosemide",
    "omeprazole",
    "pantoprazole",
}


SYMPTOMS = {
    "đánh trống ngực",
    "khó thở",
    "mệt mỏi",
    "đau ngực",
    "đau",
    "thắt chặt ngực",
    "cảm giác thắt chặt ngực",
    "buồn nôn",
    "nôn",
    "đổ mồ hôi",
    "sốt",
    "sốt cao",
    "đau nhức",
    "lo âu",
    "mất ngủ",
    "táo bón",
    "giảm dung nạp gắng sức",
    "ho",
    "vàng da",
    "vàng mắt",
    "thiếu máu",
    "tan huyết",
    "suy thận cấp",
    "tim đập nhanh",
}


DISEASES = {
    "viêm tuyến mồ hôi",
    "viêm gan cấp tính do virus B",
    "thiếu men G6PD",
    "thiếu men glucose-6-phosphate dehydrogenase",
}


def get_seed_lexicon() -> list[tuple[str, str]]:
    terms: list[tuple[str, str]] = []

    for term in MEDICATIONS:
        terms.append((term, "THUỐC"))

    for term in SYMPTOMS:
        terms.append((term, "TRIỆU_CHỨNG"))

    for term in DISEASES:
        terms.append((term, "BỆNH"))

    # Longest first.
    terms.sort(
        key=lambda item: len(item[0]),
        reverse=True,
    )

    return terms
