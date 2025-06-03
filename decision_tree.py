import math
import matplotlib.pyplot as plt
import csv
import networkx as nx

# -------------------------------
# 1. Đọc dữ liệu từ CSV file
# -------------------------------
def read_data_from_csv(file_path):
    data = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Clean up the column names by removing any excess whitespace
            cleaned_row = {}
            for key, value in row.items():
                # Extract the actual attribute name from column headers like "Outlook (G1)"
                clean_key = key.split('(')[0].strip() if '(' in key else key.strip()
                cleaned_row[clean_key] = value
            data.append(cleaned_row)
    return data

# -------------------------------
# 2. Hàm xây dựng cây quyết định (ID3)
# -------------------------------
def build_decision_tree(records, attribute_list, target_attribute):
    # Lấy tất cả nhãn phân lớp của các bản ghi, ví dụ: ['no', 'no', 'yes', ...]
    # Ví dụ: với 5 bản ghi đầu, class_labels = ['no', 'no', 'yes', 'yes', 'yes']
    class_labels = [row[target_attribute] for row in records]

    # Nếu tất cả bản ghi đều cùng một nhãn, trả về nhãn đó (nút lá)
    # Ví dụ: nếu class_labels = ['yes', 'yes', 'yes'], trả về 'yes'
    if class_labels.count(class_labels[0]) == len(class_labels):
        return class_labels[0]

    # Nếu không còn thuộc tính nào để chia, trả về nhãn xuất hiện nhiều nhất
    # Ví dụ: nếu còn 3 bản ghi ['yes', 'no', 'yes'], trả về 'yes'
    if not attribute_list:
        return get_majority_label(records, target_attribute)

    # Chọn thuộc tính tốt nhất để chia (có information gain cao nhất)
    # Ví dụ: chọn 'Outlook' nếu chia theo 'Outlook' giúp giảm entropy nhiều nhất
    best_attribute = choose_best_attribute(records, attribute_list, target_attribute)

    # Tạo nút cây với thuộc tính tốt nhất, ví dụ: {'Outlook': {}}
    decision_tree = {best_attribute: {}}
    # Lấy tất cả giá trị có thể của thuộc tính này, ví dụ: {'sunny', 'rainy', 'overcast'}
    attribute_values = set([row[best_attribute] for row in records])

    # Với mỗi giá trị của thuộc tính, xây cây con
    for value in attribute_values:
        # Lấy các bản ghi có giá trị thuộc tính bằng value
        # Ví dụ: value='sunny', subset là tất cả bản ghi có Outlook='sunny'
        subset = [record for record in records if record[best_attribute] == value]
        # if len(subset) == 0:
        #     # Nếu không có bản ghi nào, gán nhãn đa số của tập hiện tại
        #     decision_tree[best_attribute][value] = get_majority_label(records, target_attribute)
        #     print(f"Không có bản ghi nào cho {best_attribute} = {value}, gán nhãn đa số là {decision_tree[best_attribute][value]}")
        # else:
        # Đệ quy xây cây con với tập thuộc tính còn lại (loại bỏ best_attribute)
        # Ví dụ: sau khi chia theo 'Outlook', chỉ còn lại các thuộc tính khác
        new_attribute_list = [attr for attr in attribute_list if attr != best_attribute]
        subtree = build_decision_tree(subset, new_attribute_list, target_attribute)
        decision_tree[best_attribute][value] = subtree

    return decision_tree

def get_majority_label(records, target_attribute):
    # Đếm tần suất xuất hiện của từng nhãn, ví dụ: {'yes': 9, 'no': 5}
    label_frequency = {}
    for row in records:
        label = row[target_attribute]
        if label not in label_frequency:
            label_frequency[label] = 0
        label_frequency[label] += 1

    # Tìm nhãn có tần suất lớn nhất
    # Ví dụ: nếu {'yes': 9, 'no': 5} thì trả về 'yes'
    max_label = None
    max_count = -1
    for label, count in label_frequency.items():
        if count > max_count:
            max_count = count
            max_label = label
    return max_label

def choose_best_attribute(records, attribute_list, target_attribute):
    # Hàm tính entropy của một tập bản ghi
    def calculate_entropy(records):
        label_frequency = {}
        for row in records:
            label = row[target_attribute]
            if label not in label_frequency:
                label_frequency[label] = 0
            label_frequency[label] += 1

        total = len(records)
        entropy_value = 0.0
        for label in label_frequency:
            probability = label_frequency[label] / total
            # Công thức entropy: -sum(p * log2(p))
            entropy_value -= probability * math.log2(probability)
        return entropy_value

    # Hàm tính thông tin thu được khi chia theo một thuộc tính
    def calculate_information_gain(attribute):
        original_entropy = calculate_entropy(records)
        # Lấy tất cả giá trị có thể của thuộc tính, ví dụ: {'sunny', 'rainy', 'overcast'}
        attribute_values = set([row[attribute] for row in records])

        weighted_entropy = 0.0
        total = len(records)
        for value in attribute_values:
            # Lấy các bản ghi có giá trị thuộc tính bằng value
            # Ví dụ: value='sunny', subset là các bản ghi có Outlook='sunny'
            subset = [row for row in records if row[attribute] == value]
            # Tính entropy có trọng số cho từng nhánh
            weighted_entropy += (len(subset) / total) * calculate_entropy(subset)

        # Information gain = entropy ban đầu - entropy sau khi chia
        # Ví dụ: nếu entropy ban đầu là 1.0, entropy sau chia là 0.5, gain = 0.5
        return original_entropy - weighted_entropy

    # Lưu information gain của mỗi thuộc tính vào dictionary
    gain_values = {}
    for attribute in attribute_list:
        gain_values[attribute] = calculate_information_gain(attribute)
    
    # Tìm giá trị information gain lớn nhất
    max_gain = max(gain_values.values())
    
    # Tất cả thuộc tính có information gain bằng max_gain
    best_attributes = [attr for attr, gain in gain_values.items() if gain == max_gain]
    
    # Nếu có nhiều thuộc tính cùng có information gain cao nhất,
    # chọn thuộc tính đầu tiên theo thứ tự trong attribute_list
    for attribute in attribute_list:
        if attribute in best_attributes:
            return attribute
        
    # Safety return (không nên xảy ra)
    return best_attributes[0]

# -------------------------------
# 4. Vẽ cây quyết định bằng NetworkX
# -------------------------------
def build_graph(tree, G=None, parent=None, edge_label="", node_counter=[0]):
    """
    Chuyển cấu trúc tree sang đồ thị NetworkX, với mỗi node là duy nhất
    (kể cả khi nhãn lá giống nhau).
    
    - node_counter: list chứa bộ đếm node để mỗi lần gọi hàm sẽ tạo node_id mới.
    - Mỗi node sẽ có nhãn (label) để hiển thị, và cạnh (edge) sẽ có label là giá trị chia nhánh.
    """
    if G is None:
        G = nx.DiGraph()
    
    # Mỗi lần build_graph được gọi, ta tăng node_counter để tạo ID mới
    node_counter[0] += 1
    node_id = f"node_{node_counter[0]}"
    
    if isinstance(tree, str):
        # node lá
        G.add_node(node_id, label=tree, shape="ellipse")
        if parent is not None:
            G.add_edge(parent, node_id, label=edge_label)
    else:
        # node bên trong: key là thuộc tính
        key = list(tree.keys())[0]  # tên thuộc tính
        G.add_node(node_id, label=key, shape="box")
        # Nối với cha (nếu có) 
        if parent is not None:
            G.add_edge(parent, node_id, label=edge_label)
        # Duyệt các nhánh con
        for val, subtree in tree[key].items():
            build_graph(subtree, G, node_id, str(val), node_counter)
    
    return G

def get_subtree_size(G, node):
    """
    Tính số lá của subtree gốc tại 'node'.
    Nếu node không có con => subtree_size = 1 (chính nó).
    """
    children = list(G.successors(node))
    if not children:
        return 1
    return sum(get_subtree_size(G, c) for c in children)

def hierarchy_pos_custom(G, root=None, x_min=0, x_max=1, y=0, layer_height=1.2, pos=None):
    """
    Xếp node 'root' vào vị trí giữa [x_min, x_max] tại tọa độ y,
    rồi chia khoảng [x_min, x_max] cho các con dựa trên kích thước subtree của chúng.
    
    Tham số:
      - G: đồ thị NetworkX dạng cây (DiGraph).
      - root: node gốc (nếu None, tự tìm node không có cha).
      - x_min, x_max: khoảng không gian ngang có thể đặt node.
      - y: tọa độ dọc hiện tại (ví dụ 0, xuống dưới là số âm).
      - layer_height: khoảng cách dọc giữa các level.
      - pos: dict lưu kết quả {node: (x, y)}.
    Trả về:
      - pos: dict {node: (x, y)}.
    """
    if pos is None:
        pos = {}
    # Nếu root=None, tự tìm node không có cha => root
    if root is None:
        root_candidates = [n for n, d in G.in_degree() if d == 0]
        if len(root_candidates) > 0:
            root = root_candidates[0]
        else:
            root = list(G.nodes())[0]
    
    # Đặt node root vào giữa khoảng [x_min, x_max]
    x_root = (x_min + x_max) / 2
    pos[root] = (x_root, y)
    
    # Lấy danh sách con
    children = list(G.successors(root))
    if not children:
        return pos  # node lá => dừng
    
    # Tính tổng số lá của tất cả con để chia tỉ lệ
    total_size = 0
    subtree_sizes = []
    for child in children:
        s = get_subtree_size(G, child)
        subtree_sizes.append(s)
        total_size += s
    
    # Bắt đầu chia khoảng ngang [x_min, x_max] cho từng child
    x_start = x_min
    for child, size in zip(children, subtree_sizes):
        # tỉ lệ chiều rộng cho subtree = (size / total_size) * (x_max - x_min)
        w = (size / total_size) * (x_max - x_min)
        x_end = x_start + w
        
        # Gọi đệ quy cho child, xuống 1 level => y - layer_height
        hierarchy_pos_custom(G, root=child, x_min=x_start, x_max=x_end, 
                             y=y - layer_height, layer_height=layer_height, pos=pos)
        
        x_start = x_end  # Dịch sang phải cho child kế
    
    return pos

def plot_decision_tree_networkx(tree):
    """Xây dựng đồ thị NetworkX và vẽ cây quyết định."""
    # Xây dựng graph (chú ý tạo node_counter để không trùng node_id)
    G = build_graph(tree, node_counter=[0])
    
    # Tìm root (node không có incoming edges)
    root_node = None
    for n in G.nodes():
        if G.in_degree(n) == 0:
            root_node = n
            break
    
    # Gọi layout tùy chỉnh
    pos = hierarchy_pos_custom(G, root=root_node, x_min=0, x_max=1, y=0, layer_height=1.2)
    
    # Lấy label cho node và edge
    node_labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')
    
    plt.figure(figsize=(12, 8))
    
    # Vẽ node
    nx.draw(G, pos, labels=node_labels, with_labels=True,
            node_size=2500, node_color="#FFB6C1", font_size=10)
    
    # Vẽ edge với label
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.title("Cây Quyết Định ID3")
    plt.axis('off')  # Tắt trục
    plt.show()

# -------------------------------
# 5. Thực thi
# -------------------------------
if __name__ == "__main__":
    # Đọc dữ liệu từ CSV file
    csv_file_path = "data2.csv"
    data = read_data_from_csv(csv_file_path)
    
    # Chọn thuộc tính phân lớp là 'Play'
    target_attribute = "Play"
    # Danh sách thuộc tính dùng để xây cây (lấy từ dữ liệu, loại bỏ ID và thuộc tính phân lớp)
    attribute_list = [attr for attr in data[0].keys() if attr != "Id" and attr != target_attribute]

    print("Danh sách thuộc tính:", attribute_list)
    print("Số lượng bản ghi:", len(data))
    
    # Xây dựng cây quyết định từ dữ liệu
    decision_tree = build_decision_tree(data, attribute_list, target_attribute)
    print("Cây quyết định thu được:")
    print(decision_tree)

    # Vẽ cây quyết định bằng NetworkX
    plot_decision_tree_networkx(decision_tree)
