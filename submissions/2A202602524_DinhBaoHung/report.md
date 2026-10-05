# Báo cáo Lab Day 2 — DeepWeeds

> Trạng thái: pipeline và kiểm tra dữ liệu đang được chuẩn bị. Không có metric mô hình nào được điền trước khi hoàn tất các lần chạy thực tế.

## Tóm tắt

Chưa chạy thí nghiệm huấn luyện hoặc đánh giá cuối. Hoàn thiện phần này từ kết quả trong `results.xlsx` sau khi notebook chạy xong.

## Dữ liệu và thiết lập

- Dataset: DeepWeeds, fold 0.
- 17.509 ảnh có trong thư mục `data/`; train/validation/test CSV được tải nguyên bản vào `data/labels/`.
- Kiểm tra hiện tại xác nhận split không giao nhau, hợp đủ 17.509 ảnh và mọi ảnh đều tồn tại.
- Chưa có metric validation/test.

## Kết quả

Chưa có kết quả chạy. Bảng so sánh backbone, ablation huấn luyện, inference, latency và kết quả cuối sẽ được sinh từ log thật.

## Kết luận và hạn chế

Chưa thể kết luận về backbone hay công thức tốt nhất trước khi chạy đủ thí nghiệm. Kết quả fold 0 không nên được suy rộng ra địa điểm/mùa khác; việc chia ngẫu nhiên có thể lạc quan.
