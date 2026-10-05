# Báo cáo Lab Day 2 — DeepWeeds

## Tóm tắt

Cấu hình có macro-F1 validation cao nhất là **F01uncal / convnext_tiny**, macro-F1 val 0.9724, top-1 val 0.9794. Chung kết có macro-F1 test 0.9719 ± 0.0016 qua 3 seed. Top-1 test 0.9773 ± 0.0010. Đây là số liệu được tổng hợp từ các file result.json sinh bởi các lần chạy trong repo.

## Thiết lập và dữ liệu

- Fold: 0. Cấu hình chạy chọn theo validation; không gộp val vào train.
- Phân bố và kiểm tra giao split lưu trong `runs/<exp_id>/seed<seed>/config.json`.
- Cấu hình chi tiết, seed, tag trọng số, thời gian và chỉ số được lưu trong `results.xlsx`.

## So sánh và phân tích

Xem bảng Backbones, Training, Inference, Final và Latency trong `results.xlsx`; biểu đồ theo epoch nằm trong `curves/`. Chỉ diễn giải khác biệt giữa thí nghiệm có kiểm soát một yếu tố và đối chiếu độ lệch chuẩn giữa các seed.

### Recall và F1 theo lớp (trung bình các seed test)

| Lớp | Recall | F1 |
|---|---:|---:|
| Chinee Apple | 0.9454 | 0.9553 |
| Lantana | 0.9812 | 0.9692 |
| Parkinsonia | 0.9903 | 0.9816 |
| Parthenium | 0.9821 | 0.9837 |
| Prickly Acacia | 0.9812 | 0.9639 |
| Rubber Vine | 0.9769 | 0.9753 |
| Siam Weed | 0.9876 | 0.9808 |
| Snake Weed | 0.9510 | 0.9533 |
| Negatives | 0.9801 | 0.9838 |

### Ma trận nhầm lẫn chuẩn hóa theo lớp thật (trung bình các seed)

| Lớp thật \ Dự đoán | Chinee Apple | Lantana | Parkinsonia | Parthenium | Prickly Acacia | Rubber Vine | Siam Weed | Snake Weed | Negatives |
|---|---|---|---|---|---|---|---|---|---|
| Chinee Apple | 0.945 | 0.000 | 0.000 | 0.001 | 0.001 | 0.000 | 0.000 | 0.024 | 0.028 |
| Lantana | 0.000 | 0.981 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.009 | 0.009 |
| Parkinsonia | 0.005 | 0.000 | 0.990 | 0.000 | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 |
| Parthenium | 0.002 | 0.000 | 0.011 | 0.982 | 0.003 | 0.000 | 0.000 | 0.000 | 0.002 |
| Prickly Acacia | 0.000 | 0.000 | 0.008 | 0.003 | 0.981 | 0.000 | 0.000 | 0.000 | 0.008 |
| Rubber Vine | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.977 | 0.000 | 0.000 | 0.023 |
| Siam Weed | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.988 | 0.000 | 0.009 |
| Snake Weed | 0.008 | 0.007 | 0.000 | 0.003 | 0.000 | 0.000 | 0.005 | 0.951 | 0.026 |
| Negatives | 0.003 | 0.004 | 0.001 | 0.001 | 0.005 | 0.003 | 0.003 | 0.001 | 0.980 |

## Kết luận, hạn chế

Kết quả chỉ phản ánh fold 0, một GPU và tập dữ liệu DeepWeeds hiện tại. Chia ngẫu nhiên có thể lạc quan do ảnh cùng địa điểm/mùa; số seed và ngân sách tính toán hữu hạn. Không suy rộng kết luận ngoài miền dữ liệu này.
