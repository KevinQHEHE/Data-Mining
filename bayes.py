# Import thư viện sys để dùng sys.exit khi cần thoát chương trình
import sys

# Khai báo dữ liệu gốc, mỗi bản ghi là một dict với các thuộc tính và nhãn phân lớp 'play'
data = [
    {"outlook": "sunny",    "terrain": "slope",      "temperature": "hot",  "humidity": "high",   "wind": "weak",   "play": "no"},
    {"outlook": "sunny",    "terrain": "undulating", "temperature": "hot",  "humidity": "low",    "wind": "strong", "play": "no"},
    {"outlook": "overcast", "terrain": "slope",      "temperature": "hot",  "humidity": "high",   "wind": "low",    "play": "no"},
    {"outlook": "rainy",    "terrain": "flat",       "temperature": "mild", "humidity": "high",   "wind": "weak",   "play": "no"},
    {"outlook": "rainy",    "terrain": "undulating", "temperature": "cool", "humidity": "low",    "wind": "weak",   "play": "yes"},
    {"outlook": "rainy",    "terrain": "flat",       "temperature": "cool", "humidity": "normal", "wind": "low",    "play": "no"},
    {"outlook": "overcast", "terrain": "undulating", "temperature": "cool", "humidity": "normal", "wind": "strong", "play": "yes"},
    {"outlook": "sunny",    "terrain": "flat",       "temperature": "mild", "humidity": "high",   "wind": "weak",   "play": "no"},
    {"outlook": "sunny",    "terrain": "slope",      "temperature": "cool", "humidity": "normal", "wind": "weak",   "play": "yes"},
    {"outlook": "rainy",    "terrain": "slope",      "temperature": "mild", "humidity": "low",    "wind": "weak",   "play": "no"},
    {"outlook": "sunny",    "terrain": "flat",       "temperature": "mild", "humidity": "normal", "wind": "low",    "play": "no"},
    {"outlook": "overcast", "terrain": "undulating", "temperature": "mild", "humidity": "high",   "wind": "strong", "play": "yes"},
    {"outlook": "overcast", "terrain": "undulating", "temperature": "hot",  "humidity": "normal", "wind": "weak",   "play": "yes"},
    {"outlook": "rainy",    "terrain": "slope",      "temperature": "mild", "humidity": "high",   "wind": "low",    "play": "no"},
    {"outlook": "sunny",    "terrain": "flat",       "temperature": "cool", "humidity": "normal", "wind": "strong", "play": "yes"},
    {"outlook": "overcast", "terrain": "flat",       "temperature": "mild", "humidity": "normal", "wind": "low",    "play": "no"},
    {"outlook": "rainy",    "terrain": "undulating", "temperature": "mild", "humidity": "high",   "wind": "strong", "play": "no"},
]

# Các tuple mới cần phân loại, mỗi tuple là một dict các thuộc tính (chưa có nhãn phân lớp)
tuples = {
    "Row18": {"outlook": "sunny",    "terrain": "flat", "temperature": "cool",  "humidity": "low", "wind": "strong"},
    "Row19": {"outlook": "overcast", "terrain": "undulating", "temperature": "cool", "humidity": "low", "wind": "low"},
    "Row20": {"outlook": "rainy",    "terrain": "slope", "temperature": "cool", "humidity": "low", "wind": "low"},
    "Row21": {"outlook": "overcast", "terrain": "undulating",      "temperature": "cool",  "humidity": "high", "wind": "strong"},
    "Row22": {"outlook": "rainy", "terrain": "undulating", "temperature": "cool", "humidity": "low", "wind": "low"},
}

def detailed_classification(dataset, X, class_attr):
    """
    Với một tập dữ liệu và một tuple X (dạng dict các thuộc tính),
    hàm này in ra chi tiết các bước tính toán xác suất cho từng lớp.
    """
    # 1. Tách dữ liệu thành các nhóm theo giá trị của thuộc tính phân lớp (class_attr)
    # Ví dụ: class_attr = 'play', sẽ tách thành 2 nhóm: 'yes' và 'no'
    classes = {}
    for record in dataset:
        class_value = record[class_attr]  # Lấy giá trị phân lớp của bản ghi, ví dụ 'no'
        if class_value not in classes:
            classes[class_value] = []     # Nếu chưa có nhóm này thì tạo mới
        classes[class_value].append(record)  # Thêm bản ghi vào nhóm tương ứng

    total = len(dataset)  # Tổng số bản ghi trong tập dữ liệu, ví dụ 17

    # 2. Tính xác suất tiên nghiệm (prior probability) cho từng lớp
    # Ví dụ: P(play='no') = số bản ghi 'no' / tổng số bản ghi
    prior = {}
    for c in classes:
        prior[c] = len(classes[c]) / total

    print(f"P({class_attr}):")
    for c in classes:
        count = len(classes[c])
        print(f"  P({class_attr} = '{c}') = {count}/{total} = {prior[c]:.6f}")

    print(f"\nTính P(X|{class_attr}) cho từng lớp:")

    # 3. Lấy danh sách các thuộc tính (trừ thuộc tính phân lớp)
    # Ví dụ: ['outlook', 'terrain', 'temperature', 'humidity', 'wind']
    attributes = []
    for attr in X.keys():
        if attr != class_attr:
            attributes.append(attr)

    # 4. Tính xác suất có điều kiện cho từng lớp
    # Ví dụ: P(outlook='overcast'|play='no') = số bản ghi 'no' có outlook='overcast' / tổng số bản ghi 'no'
    prod = {}
    for c in classes:
        prod[c] = 1  # Khởi tạo tích xác suất cho lớp c
        print(f"\nVới {class_attr} = '{c}':")
        total_c = len(classes[c])  # Số bản ghi thuộc lớp c, ví dụ 11 cho 'no'
        for attr in attributes:
            # Đếm số bản ghi trong lớp c có thuộc tính attr bằng giá trị X[attr]
            # Ví dụ: đếm số bản ghi 'no' có outlook='overcast'
            count_attr = 0
            for rec in classes[c]:
                if rec[attr] == X[attr]:
                    count_attr += 1
            if total_c > 0:
                cond_prob = count_attr / total_c
            else:
                cond_prob = 0
            prod[c] *= cond_prob  # Nhân xác suất vào tích
            print(f"  P({attr} = '{X[attr]}' | {class_attr} = '{c}') = {count_attr}/{total_c} = {cond_prob:.6f}")
        print(f"  => P(X|{class_attr} = '{c}') = {prod[c]:.6f}")

    # 5. Nhân với xác suất tiên nghiệm để ra xác suất hậu nghiệm chưa chuẩn hóa
    # Ví dụ: P(X|play='no') * P(play='no')
    final = {}
    print(f"\nNhân P(X|{class_attr}) với P({class_attr}):")
    for c in classes:
        final[c] = prod[c] * prior[c]
        print(f"  P(X|{class_attr} = '{c}') * P({class_attr} = '{c}') = {prod[c]:.6f} x {prior[c]:.6f} = {final[c]:.6f}")

    # 6. Chọn lớp có xác suất lớn nhất làm kết quả dự đoán
    # Ví dụ: nếu P(play='no'|X) > P(play='yes'|X) thì dự đoán là 'no'
    prediction = None
    max_prob = -1
    for c in final:
        if final[c] > max_prob:
            max_prob = final[c]
            prediction = c

    print(f"\n=> Kết luận: X = {X} được phân loại là: {class_attr} = '{prediction}'\n")
    return final, prediction

# Hàm main cho phép người dùng chọn thuộc tính phân lớp và phân loại các tuple mới
def main():
    class_attr = 'play'  # Chọn cố định thuộc tính phân lớp là 'play'
    print(f"\nClassifying predefined tuples using '{class_attr}' as the class attribute...\n")
    for label, X in tuples.items():
        print(f"--- Classification for {label} ---")
        final, prediction = detailed_classification(data, X, class_attr)

        # Thêm bản ghi mới với nhãn dự đoán vào tập dữ liệu
        new_record = X.copy()
        new_record[class_attr] = prediction
        data.append(new_record)
        print(f"Updated dataset with {label}: {new_record}\n")
    
    # In ra tập dữ liệu đã cập nhật + đánh số thứ tự
    print("Updated dataset:")
    for i, record in enumerate(data):
        print(f"Row {i+1}: {record}")
        
if __name__ == "__main__":
    main()