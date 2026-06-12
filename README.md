# MDD Challenge - Speech Technology Project

Repository bài tập lớn môn Speech Technology cho bài toán Mispronunciation Detection and Diagnosis (MDD). Mục tiêu của project là dự đoán trực tiếp các edit phát âm trên chuỗi âm vị chuẩn, sau đó tái dựng chuỗi âm vị dự đoán và xuất file nộp Challenge theo định dạng `predict.zip`.

GitHub repository: https://github.com/An041003/MDD-Challenge.git

## Thành viên nhóm

- Nguyễn Bình An - 20210122E
- Phạm Đình Hải - 20241591E

## Bài toán

Input chính của mỗi mẫu gồm:

- File audio giọng nói.
- Chuỗi âm vị chuẩn `canonical`.
- Với tập train, có thêm chuỗi âm vị nhãn `transcript`.

Output cần sinh cho tập test là một file `results.csv` có đúng một cột:

```csv
predict
<predicted phoneme sequence>
```

File này được nén thành `predict.zip` và bên trong zip chỉ có `results.csv`.

Điểm Challenge được tính theo công thức:

```text
Score = 0.5 * F1 + 0.4 * (1 - DER) + 0.1 * (1 - PER)
```

## Hướng tiếp cận

Notebook chính của nhóm triển khai C-MED V1: Canonical-conditioned Mispronunciation Edit Decoder. Pipeline không sinh transcript bằng CTC trước rồi hậu xử lý về canonical, mà học trực tiếp token-level edit so với chuỗi âm vị chuẩn:

```text
audio waveform
-> Vietnamese Wav2Vec2 audio encoder
canonical phoneme sequence
-> phoneme embedding / canonical encoder
audio states + canonical states
-> cross-attention alignment
-> token-level edit heads
-> reconstruct predicted phoneme sequence
-> results.csv
-> predict.zip
```

Các thành phần chính:

- Speech backbone: Wav2Vec2 Vietnamese.
- Canonical phoneme encoder để biểu diễn chuỗi âm vị chuẩn.
- Cross-attention để align từng canonical token với acoustic evidence.
- `detection_head` dự đoán token đúng/sai.
- `operation_head` dự đoán KEEP/SUBSTITUTE trong C-MED V1.
- `replacement_head` dự đoán âm vị thay thế khi có SUBSTITUTE.
- `utterance_head` dự đoán câu có lỗi phát âm hay không.
- Calibration threshold và reconstruction ở mức edit, không fallback cả câu về canonical.

## Cấu trúc repository

```text
.
├── README.md
├── requirements.txt
├── Architecture_CMED.md
├── notebooks/
│   └── MDD_CMED_Kaggle.ipynb
├── src/
│   └── mdd_cmed/
│       ├── alignment.py
│       ├── config.py
│       ├── losses.py
│       ├── metrics.py
│       ├── model.py
│       ├── paths.py
│       ├── phonemes.py
│       └── submission.py
├── scripts/
│   └── create_cmed_kaggle_notebook.py
└── src/
    └── README.md
```

`notebooks/MDD_CMED_Kaggle.ipynb` là entrypoint chạy end-to-end trên Kaggle. Thư mục `src/mdd_cmed/` tách các thành phần lõi để code dễ đọc và tái sử dụng:

- `config.py`, `paths.py`: cấu hình thí nghiệm và path Kaggle.
- `phonemes.py`, `alignment.py`: xử lý chuỗi âm vị và tạo nhãn edit KEEP/SUB.
- `model.py`, `losses.py`: kiến trúc C-MED V1 và loss token-level.
- `metrics.py`: metric MDD nội bộ theo F1, DER, PER, Score.
- `submission.py`: kiểm tra và đóng gói `results.csv` / `predict.zip`.

Các thư mục dữ liệu, checkpoint, experiment output, file zip nộp bài và các thư mục kiểm tra tạm không được đưa lên GitHub.

## Cài đặt môi trường

Khuyến nghị chạy trên Kaggle Notebook hoặc Google Colab có GPU. Với môi trường local, tạo virtual environment rồi cài dependencies:

```bash
pip install -r requirements.txt
```

Nếu chạy notebook trên Kaggle/Colab, notebook có thể tự cài thêm một số package cần thiết như `transformers`, `librosa`, `soundfile`, `jiwer`, `pandas`, `numpy`, `tqdm`, `scikit-learn`.

## Chuẩn bị dữ liệu

Project giả định dữ liệu Challenge có cấu trúc tương tự:

```text
MDD-Challenge-2025-training-set/
├── audio_data/
│   └── train/
└── metadata/
    └── train_phones.csv
```

Các cột cần có trong `train_phones.csv`:

- `id`
- `path`
- `canonical`
- `transcript`

Trên Kaggle, notebook đang dùng các đường dẫn input:

```text
/kaggle/input/datasets/andesulaeta/mdd-data/MDD-Challenge-2025-training-set/MDD-Challenge-2025-training-set
/kaggle/input/datasets/andesulaeta/mdd-data/MDD-Challenge-2025-public-test/MDD-Challenge-2025-public-test
/kaggle/input/datasets/andesulaeta/mdd-data/MDD-Challenge-2025-private-test/MDD-Challenge-2025-private-test
```

Nếu dataset được mount ở vị trí khác, sửa các biến path ở phần cấu hình đầu notebook.

## Cách chạy notebook chính

1. Mở `notebooks/MDD_CMED_Kaggle.ipynb` trên Kaggle.
2. Bật GPU trong runtime.
3. Kiểm tra lại các path ở cell cấu hình dữ liệu.
4. Chạy tuần tự các cell:
   - kiểm tra môi trường,
   - audit dữ liệu và build vocab,
   - tạo edit labels KEEP/SUB,
   - tạo speaker-safe train/calibration/validation split,
   - tiny balanced overfit gate,
   - Stage A C-MED V1 training,
   - replacement-head refinement,
   - calibration threshold,
   - đánh giá validation và error analysis,
   - inference trên public/private test.
5. Sau inference, tạo `results.csv` chỉ gồm cột `predict`.
6. Nén `results.csv` thành `predict.zip`.

Output nộp Challenge mong muốn:

```text
/kaggle/working/results.csv
/kaggle/working/predict.zip
```

## Tạo lại notebook C-MED

Script sau dùng để sinh lại notebook C-MED Kaggle-only từ source Python:

```bash
python scripts/create_cmed_kaggle_notebook.py
```

Notebook sinh ra mặc định là:

```text
notebooks/MDD_CMED_Kaggle.ipynb
```

Notebook được giữ self-contained để tiện chạy trên Kaggle. Các module trong `src/mdd_cmed/` là bản tách riêng của các khối quan trọng, phục vụ đọc code, tái sử dụng và viết report.

## Đánh giá

Metric nội bộ dùng trong project được đặt trong:

```text
src/mdd_cmed/metrics.py
```

Các metric quan trọng khi báo cáo:

- F1
- DER
- PER
- official weighted score
- precision / recall

Sau khi chạy notebook, các metric validation, threshold, error analysis và submission manifest được ghi dưới `/kaggle/working`. Điểm public/private leaderboard phụ thuộc vào file submission cuối cùng.

## Nội dung technical report

Report PDF cuối cùng nên trình bày ngắn gọn trong giới hạn 20 trang:

- Giới thiệu bài toán MDD và công thức score.
- Mô tả dữ liệu, định dạng phoneme, train/test split.
- Pipeline tổng thể từ audio đến `predict.zip`.
- Kiến trúc C-MED, gồm canonical encoder, cross-attention và các edit heads.
- Chiến lược huấn luyện, tiny overfit gate, threshold calibration và checkpoint selection.
- Kết quả thực nghiệm và phân tích lỗi.
- Phân chia công việc của Nguyễn Bình An và Phạm Đình Hải.
- References cho pretrained model, Wav2Vec2, edit-based MDD và tài liệu Challenge.

## Ghi chú GitHub

Không commit các file/thư mục sau:

- Dữ liệu audio và metadata gốc.
- Checkpoint/model weight.
- `experiments/`, `reports/`, `data/submissions/`.
- `best_dev_predictions.csv`, `results.csv`, `predict.zip`.
- `Agent_CMED.md`.
- Các thư mục check/tạm như `__zipcheck/`, `__ordercheck/`, `__shiftcheck/`, `_tmp_zip_check/`.
