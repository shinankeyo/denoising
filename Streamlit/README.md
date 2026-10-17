# Image Denoising Streamlit - TV5

## 1. Cấu trúc thư mục

```text
denoising_streamlit_vscode/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── processed/
│       └── grayscale/
│           ├── 100098.jpg
│           ├── ...
│
└── results/
    ├── 01_Ket_qua_chi_tiet_6300_luot.csv
    ├── 02_Ket_qua_tong_hop_42_truong_hop.csv
    └── 03_Bo_loc_tot_nhat_theo_muc_nhieu.csv
```

`data/processed/grayscale/` lấy từ kết quả TV1.

`results/` lấy 3 file CSV từ TV4.

## 2. Cài thư viện trong VS Code

Mở Terminal:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Sau đó:

```bash
pip install -r requirements.txt
```

## 3. Chạy Streamlit

```bash
streamlit run app.py
```

Trình duyệt sẽ mở giao diện Streamlit.

## 4. Hai chế độ

### Demo một ảnh
- Chọn ảnh từ dataset hoặc upload ảnh.
- Chọn Gaussian/Salt-Pepper.
- Chọn 10%, 20%, 30%.
- Chọn một trong 7 cấu hình:
  - Mean 3x3
  - Mean 5x5
  - Gaussian 3x3
  - Gaussian 5x5
  - Median 3x3
  - Median 5x5
  - Bilateral
- Hiển thị:
  - Ảnh gốc
  - Ảnh nhiễu
  - Ảnh sau khử nhiễu
  - MSE
  - PSNR
  - SSIM
  - Processing Time
  - Histogram

### Kết quả thực nghiệm
Đọc trực tiếp kết quả TV4:
- 6300 lượt đánh giá
- 42 trường hợp tổng hợp
- 6 mức nhiễu
- 7 cấu hình bộ lọc

## 5. Lưu ý

Notebook TV4 hiện chưa đánh giá processing time trong bảng kết quả 6300 lượt. Giao diện Demo có đo thời gian xử lý cho từng lần chạy, nhưng không nên dùng thời gian Demo để thay thế benchmark 6300 lượt trong báo cáo nếu nhóm chưa chạy benchmark thống nhất.

## 6. Pipeline của nhóm

TV1
→ 150 ảnh BSDS500
→ resize 256x256
→ grayscale

TV2
→ Gaussian 10/20/30%
→ Salt-and-Pepper 10/20/30%

TV3
→ Mean 3x3/5x5
→ Gaussian 3x3/5x5
→ Median 3x3/5x5
→ Bilateral

TV4
→ MSE
→ PSNR
→ SSIM
→ 6300 lượt
→ 42 trường hợp tổng hợp

TV5
→ Streamlit Dashboard
→ Demo
→ So sánh
→ Trình bày kết quả
