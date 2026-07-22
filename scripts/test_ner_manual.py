from src.ner.inference import NERModel


def main():

    text = """
    Thuốc trước khi nhập viện
    - metoprolol 25mg po bid
    - doxycycline cho viêm tuyến mồ hôi
    - atenolol (uống hôm nay)

    Bệnh nhân xuất hiện triệu chứng đánh trống ngực.
    Bệnh nhân khó thở nhẹ.
    Cảm giác thắt chặt ngực.
    Cảm thấy mệt mỏi.
    """

    model = NERModel()

    print("INPUT:")
    print(text)

    print()
    print("ENTITIES:")
    print("=" * 80)

    entities = model.predict(text)

    for entity in entities:

        print(
            f"{entity.type:15} "
            f"{entity.start:5} "
            f"{entity.end:5} "
            f"{entity.text!r}"
        )


if __name__ == "__main__":
    main()
