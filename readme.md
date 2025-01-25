# Đóng góp đồ án

1. Xây dựng bộ nhận diện khuôn mặt sử dụng dữ liệu 2D và 3D tăng cường từ phương pháp Photometric Stereo.
2. Sử dụng triplet loss và magface làm loss function 
3. Sử dụng under sampling để tránh học thiên vị
4. Ảnh hưởng của backbone size đến hiệu suất của bộ nhận diện.
5. Điều chỉnh sự random giống nhau trên cả 3 loại ảnh mỗi khi dataloader load ảnh.

# Cấu trúc Project

```plaintext
concat/
├── checkpoint/                     # các experments (jupyter) và tensorboard logs + models
│   ├── hoang/                      # experment của mình
│   ├── lam/                        # experment của anh lâm khóa trước
├── 3D_Dataset/                     # dataset sau tiền xử lý
├── Dataset/                        # chia dataset thành 2 tập train và test (bản chất tập này là validate)
├── Gallery/                        # gallery set, không có Probe set
├── images/                         # kết quả so sánh Hoàng và Lâm
├── going_modular/                  # package multi task + magface để viết các experments đơn giản hơn
│   └── dataloader/                 # dataloader từng loại dữ liệu và data prefetch
│   └── loss/                       # cách tính multi task toàn mạng (focal loss + magface)
│   └── model/                      # kiến trúc mạng multi task
│   └── train_eval/                 # train loop + eval loop
│   └── utils/                      # các hàm phụ phục vụ huấn luyện như tính auc, acc, model checkpoint, early stopping, ...
├── preprocess/                     # tiền xử lý và phân tích dữ liệu từ dataset gốc (không quan tâm nếu đã có thư mục Dataset)
├── test_models/
│   └── test_model.ipynb            # expertment test dữ liệu trên tập gallery
│   └── triplet_model.ipynb         # expertment so sánh với project multi task
│   └── test.ipynb                  # experment đọc tensorboard log
├── .gitignore
└── README.md
```

Dataset download tại ![đây](https://www.kaggle.com/datasets/blueeyewhitedaragon/hoangvn-3dfacerecognition)

# Cách chạy project

**Đưa các file jupyter (experment) muốn chạy vào thư mục root của project này và chạy bình thường**
- Code có thể có 1 chút bug khi chạy, do trong quá trình làm đồ án mình đã sửa đổi rât nhiều để phù hợp với tính huống gần nhất. (chủ yếu nằm ở phần dataloader và trong jupyter, còn lại code bình thường)
- Nếu muốn tính thêm chỉ số accuracy, chỉnh lại phần comment ở file utils/roc_auc.py (nên làm với model thu được sau cùng chứ ko nên làm trong quá trình train)
- Chú ý cần đọc kỹ cẩn thận lại các đường dẫn lưu log và models.
- File requirements.txt ko hoàn chỉnh
- Muốn code nhanh, hay chạy trên máy cá nhân trước (wsl hoặc ubuntu) rồi mới chạy trên kaggle.
