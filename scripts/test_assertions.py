from src.assertions.inference import AssertionModel


def main():

    model = AssertionModel()

    test_cases = [
        (
            "Thuốc trước khi nhập viện: metoprolol 25mg po bid",
            "metoprolol 25mg po bid",
        ),
        (
            "Bệnh nhân không dùng metoprolol",
            "metoprolol",
        ),
        (
            "Bệnh nhân đã dùng aspirin",
            "aspirin",
        ),
        (
            "Bệnh nhân được kê aspirin",
            "aspirin",
        ),
        (
            "Tiền sử bệnh nhân có tăng huyết áp",
            "tăng huyết áp",
        ),
        (
            "Bệnh nhân không có khó thở",
            "khó thở",
        ),
        (
            "X-quang ngực không ghi nhận gì bất thường",
            "không ghi nhận gì bất thường",
        ),
    ]

    for text, entity_text in test_cases:

        print()
        print("=" * 80)

        print(f"Text:   {text}")

        print(f"Entity: {entity_text}")

        # ------------------------------------------------------
        # Find entity position in original text
        # ------------------------------------------------------

        start = text.lower().find(entity_text.lower())

        if start == -1:

            print("ERROR: Entity not found in text")

            continue

        end = start + len(entity_text)

        print(f"Position: [{start}, {end}]")

        # ------------------------------------------------------
        # Run assertion model
        # ------------------------------------------------------

        result = model.predict(
            text=text,
            start=start,
            end=end,
        )

        print(f"Assertions: {result}")


if __name__ == "__main__":
    main()
