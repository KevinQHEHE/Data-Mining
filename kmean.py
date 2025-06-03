# kmeans.py

import numpy as np
import matplotlib.pyplot as plt


# 1. Chuẩn bị dữ liệu
data = [
    (1, 5), (3, 5), (2, 7), (3, 4), (5, 5),
    (2, 6), (4, 1), (4, 3), (4, 7), (4, 6),
    (7, 1), (6, 5), (8, 1), (4, 8), (8, 3),
    (12, 4), (3, 16), (7, 14)
]

initial_centroids = [(2, 7), (3, 5), (7, 1)]
MAX_ITERATIONS = 5

def euclidean_distance(p1, p2):
    """
    Tính khoảng cách Euclid giữa hai điểm p1 và p2.
    p1, p2 là tuple (x, y).
    """
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def assign_points_to_clusters(data_points, centroids):
    """
    Gán mỗi điểm trong data_points vào cụm có centroid gần nhất.
    Trả về list các cluster, mỗi cluster là 1 list điểm.
    Đồng thời in ra bảng khoảng cách và cluster được gán.
    """
    clusters = [[] for _ in centroids]
    # In header
    header = "Point".center(10) + " | " + \
             " | ".join([f"d{i+1}".center(8) for i in range(len(centroids))]) + \
             " | Assigned Cluster"
    print(header)
    print("-" * (13 + 12 * len(centroids)))
    
    for point in data_points:
        # Tính khoảng cách đến từng centroid
        distances = [euclidean_distance(point, c) for c in centroids]
        # Chọn centroid gần nhất
        # idx = chỉ số của khoảng cách nhỏ nhất trong mảng distances
        idx = np.argmin(distances)
        # In dòng dữ liệu
        dist_str = " | ".join(f"{d:^8.3f}" for d in distances)
        print(f"{str(point):^10} | {dist_str} | {idx+1}")
        # Gán điểm vào cluster dựa vào chỉ số idx
        clusters[idx].append(point)
    print()
    return clusters

def calculate_new_centroids(clusters):
    """
    Tính centroid mới bằng cách lấy trung bình các tọa độ
    của điểm trong mỗi cluster.
    Nếu cluster trống, giữ nguyên centroid (có thể xử lý tùy ý).
    """
    new_centroids = []
    for cluster in clusters:
        # Nếu cluster không rỗng, tính centroid mới
        if cluster:
            # Lấy tọa độ x
            xs = [p[0] for p in cluster]
            # Lấy tọa độ y
            ys = [p[1] for p in cluster]
            new_centroids.append((sum(xs)/len(xs), sum(ys)/len(ys)))
        else:
            new_centroids.append((0, 0))
    return new_centroids

def get_data_limits(data_points, centroids):
    """
    Tính giới hạn trục cho việc vẽ, dựa trên
    toàn bộ điểm + centroid, cộng thêm padding.
    """
    # Kết hợp tất cả các điểm và centroid
    pts = data_points + centroids
    # Tách tọa độ x và y
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # Padding 20%, đảm bảo tối thiểu 1 đơn vị
    pad_x = max((x_max - x_min)*0.2, 1)
    pad_y = max((y_max - y_min)*0.2, 1)
    return [x_min-pad_x, x_max+pad_x], [y_min-pad_y, y_max+pad_y]

def visualize_cluster_iteration(clusters, centroids, iteration, ax):
    # Danh sách các centroid hiện tại
    """
    Vẽ scatter plot cho từng cluster và các centroid tương ứng
    lên subplot ax, đánh dấu iteration.
    """
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    all_pts = []
    for i, cluster in enumerate(clusters):
        if not cluster: continue
        xs = [p[0] for p in cluster]; ys = [p[1] for p in cluster]
        # Vẽ các điểm dữ liệu
        ax.scatter(xs, ys, color=colors[i], label=f'Cluster {i+1}')
        # vẽ centroid
        cx, cy = centroids[i]
        ax.scatter(cx, cy, marker='x', s=100, linewidths=2,
                   color=colors[i])
        # Thêm tất cả các điểm trong cụm hiện tại
        all_pts += cluster
    ax.set_title(f'Iteration {iteration}')
    ax.set_xlabel('X-coordinate'); ax.set_ylabel('Y-coordinate')
    xlim, ylim = get_data_limits(all_pts, centroids)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.legend()
    ax.grid(True)

def run_kmeans(data_points, initial_centroids, max_iter):
    """
    Điều phối thuật toán K-means:
      1. Gán điểm vào cluster
      2. Tính centroid mới
      3. Vẽ kết quả mỗi vòngf
      4. Lặp đến khi hội tụ hoặc đạt max_iter
    Trả về clusters cuối và centroids cuối.
    """
    centroids = initial_centroids
    fig_rows = 2
    fig_cols = (max_iter + 1 + 1) // fig_rows
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(5*fig_cols, 5*fig_rows))
    axes = axes.flatten()
    
    # Lặp qua tối đa max_iter vòng lặp hoặc cho đến khi thuật toán hội tụ.
    for it in range(max_iter):
        print(f"\n--- Iteration {it+1} ---")
        clusters = assign_points_to_clusters(data_points, centroids)
        for idx, c in enumerate(clusters):
            print(f"Cluster {idx+1}: {c}")
        new_centroids = calculate_new_centroids(clusters)
        print(f"New centroids: {new_centroids}")
        visualize_cluster_iteration(clusters, centroids, it+1, axes[it])
        
        if new_centroids == centroids:
            print("Converged.")
            break
        centroids = new_centroids

    # Ẩn subplot thừa
    for j in range(it+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    return clusters, centroids

# 2. Chạy thuật toán KMeans
final_cluster, final_centroids = run_kmeans(data, initial_centroids, MAX_ITERATIONS)

# 3. in kết quả cuối 
print("\n ---Final Result---")
for i, cluster in enumerate(final_cluster):
    print(f"Cluster {i + 1}: {cluster}")

print(f"Centroid {i + 1}: {final_centroids[i]}")

# 4. Vẽ đồ thị
plt.show()