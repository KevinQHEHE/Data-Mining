import os

#   Nhóm 6
#   22110212	Vũ Xuân	Quang
# 	22110166	Chung Quang Đăng Khoa
# 	22110217	Hàng Diễm Quỳnh
# 	22110251	Nguyễn Nam Thùy Trinh

MENU = '''\nChọn bài toán muốn chạy:\n1. Decision Tree\n2. Bayes\n3. K-means\n0. Thoát\nNhập lựa chọn của bạn: '''

FILE_MAP = {
    '1': 'decision_tree.py',
    '2': 'bayes.py',
    '3': 'kmean.py',
}

def main():
    while True:
        choice = input(MENU)
        if choice == '0':
            print('Thoát chương trình.')
            break
        elif choice in FILE_MAP:
            print(f'Đang chạy {FILE_MAP[choice]}...')
            os.system(f'python {FILE_MAP[choice]}')
        else:
            print('Lựa chọn không hợp lệ, vui lòng thử lại.')

if __name__ == '__main__':
    main()
