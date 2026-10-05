# Bài nộp cá nhân — Lab Day 2

- Sinh viên: Đinh Bảo Hùng
- MSSV: 2A202602524
- Notebook chính: `code/lab_day2.ipynb`
- Mục tiêu: chạy trên Google Colab để tránh lỗi CUDA/driver trên máy local và giữ workflow dễ tái lập.

## Chạy trên Colab (khuyến nghị)

1. Mở notebook `code/lab_day2.ipynb` trong Colab (upload từ repo hoặc mở qua GitHub link).
2. Mount Google Drive và đặt repo vào thư mục Drive để lưu dữ liệu chạy/điểm/biểu đồ:

```python
from google.colab import drive

drive.mount('/content/drive')
%cd /content/drive/MyDrive
```

3. Clone/fork repo và truy cập thư mục làm bài:

```bash
git clone https://github.com/<username>/K4-DAY02-HoVaTen-MSSV.git
cd K4-DAY02-HoVaTen-MSSV
```

4. Mở notebook `submissions/2A202602524_DinhBaoHung/code/lab_day2.ipynb` và chạy các ô từ trên xuống.
5. Nếu dữ liệu chưa có trong `data/`, notebook sẽ tự tải `labels/*.csv`; nếu ảnh chưa có, cho `IMAGES_DIR = "data"` hoặc mount `data/` từ Drive vào đúng vị trí.

## Cài đặt thư viện

```python
!pip install -q timm openpyxl fvcore
!pip install -q -r submissions/2A202602524_DinhBaoHung/code/requirements.txt
```

Dùng batch `8`, `num_workers=0` để phù hợp GPU Colab 4 GB/VRAM thấp và tránh quá tải RAM. Các biến này đã được thiết lập trong notebook và file `config.json` của mỗi `runs/<exp_id>/seed<seed>/`.

## Thứ tự chạy

Chạy notebook lần lượt:

- kiểm tra dữ liệu và fold
- smoke test pipeline
- 5 backbone
- huấn luyện ablation (≥ 3 trục)
- suy luận + latency (≥ 4 phương pháp)
- chạy vòng cuối cho seed 0/1/2
- `eval.py score` và `eval.py grade`
- sinh `results.xlsx`, `curves/`, `predictions/`, và `report.md`

## Lưu trữ kết quả

Nên lưu toàn bộ thư mục `submissions/2A202602524_DinhBaoHung/` trên Google Drive để giữ các ảnh biểu đồ, file dự đoán và checkpoint log. Không commit dataset nặng hoặc checkpoint lớn; chỉ commit code, `results.xlsx`, `curves/`, `predictions/` và báo cáo.

## Ghi chú

Bản thân notebook đã có logic tự nhận diện môi trường Colab/CPU/GPU; nếu chạy local, thay vì `drive.mount()` thì giữ nguyên `Path.cwd()` như hiện tại. Mọi số liệu phải đến từ lần chạy thật trên Colab/GPU, không dùng số giả từ bài báo hoặc slide.
