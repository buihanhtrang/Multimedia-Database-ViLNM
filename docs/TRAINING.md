# Hướng dẫn train ViLNM theo phiên trên máy cá nhân

Tài liệu này hướng dẫn train phiên bản đầy đủ của ViLNM (NTA + LA + JII) trên
CUHK-PEDES, và trên VN3K tiếng Việt với backbone SigLIP 2 (mục 9), bằng một GPU nhỏ (đã đo trên RTX 3050 6 GB). Quá trình train được chia
thành nhiều phiên ngắn, nên bạn vẫn dùng máy cho việc khác được. Nội dung gồm:
cài môi trường, đo thời gian, chạy và dừng/tiếp tục, theo dõi bằng MLflow, đánh
giá kết quả, và chuyển sang server khi cần.

> Muốn biết mô hình làm gì, xem [README.md](../README.md). Tài liệu này chỉ nói về
> cách chạy.

---

## 1. Tóm tắt nhanh

```bash
# 1 lần duy nhất
uv sync                                   # tạo .venv từ pyproject.toml / uv.lock
unset LD_LIBRARY_PATH                     # xem mục 2.2
.venv/bin/python tools/download_clip.py   # nếu chưa có pretrained/ViT-B-16.pt
.venv/bin/python tools/smoke_test.py      # 39/39 checks, ~2 phút

# mỗi phiên train (mặc định 4 epoch ≈ 2h45 trên RTX 3050)
tools/train_session.sh

# theo dõi
cat logs/CUHK-PEDES/vilnm_bs8_e30/status.json
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000

# dừng sớm một cách an toàn (từ terminal khác)
touch logs/CUHK-PEDES/vilnm_bs8_e30/STOP
```

Chạy lại `tools/train_session.sh` lúc nào cũng được. Script tự tiếp tục đúng chỗ
đã dừng, kể cả khi dừng giữa epoch.

---

## 2. Cài đặt môi trường

### 2.1 venv bằng uv

Dự án dùng [uv](https://docs.astral.sh/uv/), được khai báo trong
[pyproject.toml](../pyproject.toml) và khóa phiên bản trong `uv.lock`:

```bash
uv sync
```

Lệnh này tạo `.venv/` với Python 3.13, torch/torchvision bản **CUDA 13.0** (lấy
từ index của PyTorch), MLflow, `psutil` và `nvidia-ml-py` (để MLflow ghi chỉ số
GPU).

Nếu driver của máy khác đi, sửa URL index trong `pyproject.toml`
(`.../whl/cu128`, `.../whl/cpu`, ...) rồi chạy lại `uv sync`.

Không bắt buộc phải kích hoạt venv. Tài liệu này gọi thẳng `.venv/bin/python`.
Nếu thích, dùng `source .venv/bin/activate` hoặc `uv run python ...`.

### 2.2 Lỗi `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`

Trên máy này, conda có một hook
`~/miniconda3/etc/conda/activate.d/zz_cuda_ld_path.sh` đặt `LD_LIBRARY_PATH`
trỏ vào thư viện CUDA của **conda**. Venv sẽ nạp lẫn cuDNN của conda và hỏng ngay
ở lớp Conv đầu tiên. Có ba cách xử lý:

- dùng `tools/train_session.sh`, script đã tự `unset LD_LIBRARY_PATH`;
- gõ `unset LD_LIBRARY_PATH` (hoặc `conda deactivate`) trong terminal trước khi chạy;
- chạy với `env -u LD_LIBRARY_PATH .venv/bin/python ...`.

Kiểm tra:

```bash
unset LD_LIBRARY_PATH
.venv/bin/python -c "import torch; c=torch.nn.Conv2d(3,8,3).cuda(); print(c(torch.randn(1,3,32,32,device='cuda')).shape)"
```

### 2.3 Dữ liệu và trọng số

```
data/CUHK-PEDES/
├── imgs/CUHK01, CUHK03, Market, cam_a, cam_b, test_query, train_query
└── reid_raw.json      ← phải nằm ở đây (đã chép ra từ thư mục json giải nén)
pretrained/ViT-B-16.pt ← trọng số CLIP gốc của OpenAI, KHÔNG phải mô hình đã train
```

`train.json`, `val.json`, `test.json`, `segs/` và `text_attribute_graph/` là định
dạng của ViTAA, code này không dùng đến.

### 2.4 Kiểm tra trước khi train

```bash
.venv/bin/python tests/test_modules.py     # 19 unit test
.venv/bin/python tools/smoke_test.py       # 39 check end-to-end, ~2 phút
```

Smoke test kiểm tra luôn phần train theo phiên. Một run 3 epoch được cắt thành 3
phiên (một lần dừng ở cuối epoch, một lần dừng giữa epoch). Kết quả cuối phải
**trùng từng bit** với run chạy liền về trọng số, best R@1, caption NTA, và phải
là cùng một run MLflow. Smoke test cũng kiểm tra Ctrl+C có lưu được checkpoint
giữa epoch.

---

## 3. Đo trước: batch size và thời gian

```bash
unset LD_LIBRARY_PATH
.venv/bin/python tools/benchmark.py --eval                # thử batch 8 16 24 32
.venv/bin/python tools/benchmark.py --batch_sizes 8 12 --num_workers 4
```

Mỗi batch size chạy trong một tiến trình riêng, nên OOM ở batch này không ảnh
hưởng batch khác. Script chạy 10 bước khởi động và 40 bước đo thật (dữ liệu thật,
AMP, đủ loss). Kết quả lưu vào `logs/benchmark.json`.

**Kết quả đo ngày 15/10/2026 (RTX 3050 6 GB, desktop đang chiếm ~1,3 GB VRAM,
8 worker):**

| batch | VRAM đỉnh | giây/bước | ảnh/giây | 1 epoch | 30 epoch | 60 epoch |
|---|---|---|---|---|---|---|
| 8 | 3,6 / 6 GB | 0,281 | 28,5 | **~40 phút** | **~20 giờ** | ~40 giờ |
| 16, 24, 32 | OOM | | | | | |

- Một lượt đánh giá trên tập test (3.074 ảnh, 6.156 caption) mất khoảng 65 giây.
- Chờ dữ liệu chiếm khoảng 0% thời gian mỗi bước, tức GPU là nút thắt chứ không
  phải CPU. Có thể giảm `--num_workers` xuống 4 để CPU rảnh hơn cho việc khác.
- Lần chạy thử thật cho tốc độ khớp với benchmark (200 bước / 56 giây).

**Về batch 8 và batch 64 của bài báo.** JII và NDF là loss contrastive, lấy mẫu
âm từ chính các mẫu trong batch. Batch 8 chỉ có 7 mẫu âm cho mỗi mẫu, so với 63
trong bài báo. Vì vậy R@1 nhiều khả năng sẽ thấp hơn số liệu công bố.
Gradient accumulation **không** bù được điều này, vì mẫu âm vẫn chỉ đến từ từng
micro-batch. Có hai cách tăng batch thật:

1. **Gradient checkpointing** cho các khối transformer của CLIP: đổi khoảng
   20–30% tốc độ để giảm mạnh bộ nhớ activation, có thể đủ cho batch 16–32 trên
   6 GB. Code **chưa** có tính năng này.
2. **Train trên server** có GPU 24 GB trở lên (mục 8).

---

## 4. Chạy train theo phiên

### 4.1 Cách đơn giản: `tools/train_session.sh`

```bash
tools/train_session.sh                    # 4 epoch rồi dừng
EPOCHS=2 tools/train_session.sh           # phiên ngắn hơn
HOURS=7 EPOCHS=0 tools/train_session.sh   # chạy tối đa 7 giờ, dừng giữa epoch nếu cần
tools/train_session.sh --num_workers 4    # tham số thêm được chuyển thẳng cho train.py
```

Script thực hiện các bước sau:
- tự `--resume` nếu đã có `logs/<dataset>/<NAME>/last.pth`;
- bỏ `LD_LIBRARY_PATH`;
- chạy dưới `systemd-inhibit` để máy không sleep;
- bật MLflow.

Các biến môi trường cấu hình:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `NAME` | `vilnm_bs8_e30` | tên run, cũng là thư mục `logs/<dataset>/<NAME>/` |
| `DATASET` | `CUHK-PEDES` | |
| `NUM_EPOCH` | `30` | tổng số epoch của **cả run**, quyết định lịch LR cosine |
| `BATCH` | `8` | |
| `WORKERS` | `8` | |
| `EVAL_PERIOD` | `5` | đánh giá mỗi 5 epoch và ở epoch cuối |
| `EPOCHS` | `4` | số epoch tối đa **mỗi phiên** (0 = không giới hạn) |
| `HOURS` | `0` | số giờ tối đa mỗi phiên (0 = không giới hạn) |

### 4.2 Quy tắc quan trọng

1. **Không đổi `NUM_EPOCH`, `BATCH`, `lr`, seed hay các công tắc loss giữa các
   phiên của cùng một `NAME`.** Đổi những tham số này là thay đổi chính quá trình
   train. `train.py` sẽ in `WARNING ... changed since the checkpoint was written`
   nếu phát hiện. Muốn thử cấu hình khác thì dùng `NAME` khác.
2. **Muốn train ít epoch hơn thì đặt `NUM_EPOCH` nhỏ ngay từ đầu** (ví dụ 30),
   đừng đặt 60 rồi dừng ở 30. Lịch cosine tính theo `NUM_EPOCH`; dừng giữa
   chừng nghĩa là LR vẫn còn cao và mô hình chưa hội tụ gọn.
3. Ranh giới epoch là điểm dừng "đẹp", nhưng dừng giữa epoch cũng an toàn hoàn
   toàn: run đi tiếp đúng các batch chưa học.

### 4.3 Các cách dừng một phiên

Mọi cách dưới đây đều **lưu `last.pth` rồi mới thoát**:

| Cách | Khi nào dùng |
|---|---|
| `--stop_after_epochs N` | lên lịch trước: "tối nay chạy 4 epoch" |
| `--max_hours H` | lên lịch theo giờ: "chạy đến 7h sáng" |
| `touch logs/<dataset>/<NAME>/STOP` | cần máy ngay, từ một terminal khác. Dừng sau bước hiện tại (dưới 1 giây). |
| **Ctrl+C** (1 lần) | đang ở ngay terminal đó. Lần Ctrl+C thứ 2 sẽ thoát ngay **không lưu**. |
| `kill <pid>` (SIGTERM) | từ script hoặc từ `htop` |
| `--stop_after_steps N` | chủ yếu để test |

Thêm vào đó, cứ mỗi `--ckpt_every_minutes` phút (mặc định 30) checkpoint lại được
ghi một lần. Nếu mất điện hay máy treo, bạn mất tối đa khoảng 30 phút train.
File được ghi theo kiểu nguyên tử (ghi ra `.tmp` rồi đổi tên), nên sập đúng lúc
đang ghi cũng không làm hỏng checkpoint cũ.

### 4.4 Chạy nền khi đóng VS Code / terminal

`tmux` chưa được cài. Có hai lựa chọn:

```bash
# cách 1: nohup
nohup tools/train_session.sh > logs/session.out 2>&1 &
tail -f logs/session.out

# cách 2: tmux (cần sudo apt install tmux)
tmux new -s vilnm 'tools/train_session.sh'     # Ctrl+B rồi D để thoát ra, tmux attach -t vilnm để vào lại
```

### 4.5 Lịch chạy gợi ý (30 epoch, batch 8)

Tổng cộng khoảng 20 giờ GPU.

| Phương án | Cách chạy | Số phiên |
|---|---|---|
| Ban đêm | `HOURS=7 EPOCHS=0` mỗi tối | ~3 đêm |
| Phiên ngắn | `EPOCHS=4` (≈ 2h45) khi rảnh | ~8 phiên |
| Cuối tuần | `EPOCHS=0` chạy một mạch | ~20 giờ liên tục |

Trong lúc train, GPU bị chiếm gần hết. Desktop, trình duyệt và VS Code vẫn dùng
được, nhưng đừng chạy song song việc khác cần GPU (game, mô hình khác).

### 4.6 Bảo vệ máy

- **Nhiệt độ:** `watch -n 5 nvidia-smi`, hoặc xem `system/gpu_0_*` trong MLflow.
  Nếu GPU thường xuyên trên 85 °C, có thể giới hạn công suất bằng
  `sudo nvidia-smi -pl <watt>`. Train sẽ chậm hơn một chút nhưng mát hơn.
- **RAM:** mỗi worker giữ một bản sao dataset. 8 worker trên máy 30 GB là ổn;
  nếu máy bắt đầu dùng swap thì giảm `WORKERS`.
- **Ổ đĩa:** mỗi run chiếm khoảng 2,5 GB (`last.pth` 1,9 GB + `best.pth` 0,6 GB).
  Ổ hiện còn trống 654 GB.

### 4.7 Dùng `train.py` trực tiếp

`tools/train_session.sh` chỉ là lớp bọc quanh `train.py`. Lệnh tương đương:

```bash
unset LD_LIBRARY_PATH
.venv/bin/python train.py --dataset_name CUHK-PEDES --name vilnm_bs8_e30 \
    --num_epoch 30 --batch_size 8 --amp True --num_workers 8 --eval_period 5 \
    --stop_after_epochs 4 --mlflow True \
    --resume logs/CUHK-PEDES/vilnm_bs8_e30/last.pth      # bỏ dòng này ở phiên đầu
```

---

## 5. Theo dõi tiến độ và kết quả

### 5.1 Không cần công cụ gì

Mỗi run nằm trong thư mục `logs/<dataset>/<NAME>/`:

| File | Nội dung |
|---|---|
| `status.json` | trạng thái `running`, `paused` hoặc `finished`; epoch/bước hiện tại; best R@1; lý do dừng; thời gian epoch gần nhất |
| `train_log.txt` | toàn bộ log, nối tiếp qua các phiên |
| `configs.json` | tham số của phiên gần nhất |
| `last.pth` | trạng thái đầy đủ để resume |
| `best.pth` | trọng số có R@1 cao nhất (chỉ trọng số + kết quả) |

```bash
cat logs/CUHK-PEDES/vilnm_bs8_e30/status.json
grep -E "done in|new best|NTA epoch" logs/CUHK-PEDES/vilnm_bs8_e30/train_log.txt
```

### 5.2 MLflow

Bật bằng `--mlflow True` (`train_session.sh` đã bật sẵn). Dữ liệu lưu ở
`mlflow.db` và `mlruns/` ngay trong repo, không cần server, không gửi gì ra ngoài.
Telemetry của MLflow được tắt (xem phần ghi chú cuối mục này).

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
# mở http://127.0.0.1:5000  → experiment "CUHK-PEDES"
```

Những gì được ghi:

| Nhóm | Metric | Trục x |
|---|---|---|
| `train/*` | `loss`, `jii_loss`, `interc_loss`, `intrac_loss`, `ndf_loss`, `id_loss`, `lr`, mỗi `--log_period` bước | global step |
| `epoch/*` | `time_min`, `gpu_peak_mem_gb`, trung bình các loss, `nta_eliminated_ratio`, `nta_mean_noise`, số token bị mask/replace/delete | epoch |
| `eval/*` | `eval/t2i/R1 R5 R10 mAP mINP` (kết hợp), `eval/t2i_global/*`, `eval/t2i_local/*` | epoch |
| `system/*` | CPU, RAM, đĩa, GPU (bộ nhớ, % dùng, công suất), khoảng 10 giây một lần | thời gian |

Ngoài ra còn có:
- **Params:** toàn bộ tham số `train.py`.
- **Tags:** `state` (running/paused/finished), `best_R1`, `best_epoch`, `paused_at`, `output_dir`.
- **Artifacts khi run kết thúc:** `configs.json`, `train_log.txt`. Checkpoint
  **không** được chép vào MLflow để tránh nhân đôi 2,5 GB; đường dẫn tới
  checkpoint nằm ở tag `output_dir`.

Một run chia thành nhiều phiên vẫn chỉ là **một run MLflow**: run ID được lưu
trong `last.pth` và mở lại khi resume. Giữa hai phiên, run có trạng thái
`KILLED` kèm tag `state=paused`. Đây là trạng thái gần nhất với "tạm dừng" mà
MLflow có, không có nghĩa là lỗi.

> Ghi chú: MLflow 3 mặc định có một luồng telemetry gửi dữ liệu sử dụng về server
> của MLflow. Luồng này còn rút số ngẫu nhiên từ `random` của Python vào những
> thời điểm bất kỳ, và đã được kiểm chứng là làm resume không còn chính xác từng
> bit. `utils/tracker.py` tắt nó bằng `MLFLOW_DISABLE_TELEMETRY=true`.

### 5.3 Vì sao chưa dùng DVC

DVC dùng để quản lý phiên bản **dữ liệu và file lớn** (dataset, checkpoint) qua
git và một kho lưu trữ từ xa. Ở giai đoạn này chưa cần, vì:
- dataset cố định, không thay đổi;
- chỉ có một máy;
- MLflow đã lo phần log và kết quả.

DVC sẽ đáng thêm vào khi bạn chuyển sang server và cần đồng bộ `data/` cùng
checkpoint giữa nhiều máy, hoặc khi bắt đầu chỉnh sửa dataset (lọc caption,
thêm nhãn, ...).

---

## 6. Đánh giá (benchmark)

Khi `status.json` báo `"state": "finished"`:

```bash
unset LD_LIBRARY_PATH
.venv/bin/python test.py --config_file logs/CUHK-PEDES/vilnm_bs8_e30/configs.json \
                         --checkpoint  logs/CUHK-PEDES/vilnm_bs8_e30/best.pth
```

Script in ra R@1/5/10, mAP và mINP cho ba cách tính độ tương đồng: kết hợp
(`0.25·global + 0.75·local`, là con số so với bài báo), chỉ global, và chỉ local.
Có thể thử trọng số khác mà không cần train lại, ví dụ
`--sim_alpha 0.5 --sim_beta 0.5`.

`best.pth` cũng đánh giá được trong lúc run còn dở, vì nó được ghi mỗi khi R@1 tăng.

**Lưu ý phương pháp:** mặc định `--val_dataset test`, tức `best.pth` được chọn
theo tập test (giống IRRA và bài báo). Như vậy con số báo cáo hơi lạc quan.
Muốn chọn checkpoint một cách sạch hơn, dùng `--val_dataset val` khi train rồi
chạy `test.py` trên tập test.

---

## 7. Resume đảm bảo những gì

`last.pth` lưu đủ mọi thứ quá trình train phụ thuộc vào:

- trọng số mô hình;
- trạng thái optimizer Adam và scheduler LR;
- `GradScaler` của AMP;
- caption NTA hiện tại, mức nhiễu của từng token và RNG riêng của NTA;
- RNG của Python, NumPy, torch CPU và CUDA;
- vị trí trong epoch (epoch, bước);
- best R@1 tính đến lúc đó;
- run ID của MLflow.

Thứ tự dữ liệu của epoch *e* chỉ phụ thuộc vào `(seed, e)`. Khi resume giữa
epoch, sampler bỏ qua các batch đã học mà **không phải đọc lại ảnh**.

Mức độ chính xác của resume:

| Trường hợp | Kết quả |
|---|---|
| Dừng ở cuối epoch | giống hệt từng bit so với run chạy liền (đã kiểm chứng trên GPU với `num_workers` 0 và 2, và trên CPU) |
| Dừng giữa epoch, `num_workers=0` | giống hệt từng bit (đã kiểm chứng trên CPU và GPU) |
| Dừng giữa epoch, `num_workers>0` | thứ tự dữ liệu vẫn giống hệt, nhưng augmentation ngẫu nhiên của các batch còn lại trong epoch đó khác đi (worker được tạo lại với seed mới). Từ đó trọng số và caption NTA lệch nhẹ (đo được max\|diff\| ≈ 2e-4 sau 3 epoch). Về thống kê, đây chỉ tương đương một seed augmentation khác. Muốn giống hệt tuyệt đối thì dừng ở cuối epoch (`--stop_after_epochs`). |

Các lỗi của phiên bản trước đã được sửa:

- Sau resume, best R@1 bị reset về -1, nên lần eval đầu tiên luôn ghi đè
  `best.pth`, kể cả khi kết quả kém hơn.
- `--save_optimizer` mặc định là `False`, nên resume thực chất khởi động lại Adam.
  Giờ mặc định là `True`.
- Không lưu AMP scaler, trạng thái NTA, RNG, vị trí giữa epoch.
- Checkpoint không được ghi nguyên tử.

---

## 8. Khi chuyển sang server mạnh hơn

1. Chép repo, `data/CUHK-PEDES/` và `pretrained/ViT-B-16.pt` lên server (hoặc
   thêm DVC ở thời điểm này, xem mục 5.3).
2. `uv sync`. Nếu driver khác thì đổi index CUDA trong `pyproject.toml` trước.
3. Đo lại: `.venv/bin/python tools/benchmark.py --batch_sizes 32 64 --eval`.
4. Bắt đầu một **run mới** với `NAME` mới và batch lớn, ví dụ
   `NAME=vilnm_bs64_e60 BATCH=64 NUM_EPOCH=60 EPOCHS=0 tools/train_session.sh`.
   Không nên resume run batch 8 với batch 64: quá trình train sẽ trở thành một
   thứ lai, không so sánh được với bên nào.
5. Muốn gom kết quả về một chỗ, chạy MLflow server trên một máy rồi trỏ
   `--mlflow_uri http://<host>:5000`.

Một checkpoint resume được giữa hai máy khác GPU, nhưng không còn bảo đảm giống
từng bit, vì các kernel CUDA khác nhau giữa các thế hệ GPU.

---

## 9. Bộ VN3K (tiếng Việt, có nhiễu)

```
data/VN3K/
├── vn3k.json                      # 2.000 id / 4.302 ảnh / 8.604 caption train; 1.000 id / 2.000 ảnh test; không có val
├── images/
├── noise_sentence/VN3K-VI_{0.1..0.8}.npy                 # tráo cả câu sang người khác
└── noisy_color/vn3k_attribute_noise_{10..70}pct.json     # đổi từ chỉ màu
```

```bash
# sạch
.venv/bin/python train.py --dataset_name VN3K --name vn3k_clean ...
# nhiễu câu 20%
.venv/bin/python train.py --dataset_name VN3K --name vn3k_sent20 --noise_file noise_sentence/VN3K-VI_0.2.npy ...
# nhiễu màu "20%"
.venv/bin/python train.py --dataset_name VN3K --name vn3k_color20 --noise_file noisy_color/vn3k_attribute_noise_20pct.json ...
```

- Nhiễu chỉ áp lên tập train; tập test luôn sạch. Log in ra tỉ lệ caption thực sự
  bị thay đổi (`caption noise from ...`).
- **Tỉ lệ nhiễu thật của `noisy_color` cao hơn tên file**, vì nhiều caption ghi
  `is_real: true` vẫn bị đổi màu: file 10% thực tế là 28,2%, 20% là 36,1%,
  30% là 44,1%, 40% là 51,8%, 50% là 60,0%, 60% là 67,8%, 70% là 75,7%.
  Code dùng nội dung caption, không dựa vào cờ `is_real`. Khi báo cáo, ghi tỉ lệ
  thật hoặc nói rõ điều này.
- `noise_sentence` có tỉ lệ đúng như tên file.
- **CLIP (tiếng Anh) gần như không đọc được tiếng Việt.** Trung vị 97 token BPE
  mỗi caption và 77,8% caption bị cắt ở 77 token (CUHK: 24 token, 0,3%); zero-shot
  R@1 chỉ 0,18. Vì vậy VN3K dùng backbone **SigLIP 2** (mục 9.1).

### 9.1 Backbone SigLIP 2 (`--backbone siglip2`)

`google/siglip2-base-patch16-256` là model đa ngôn ngữ (tokenizer Gemma 256k), được
chọn sau khi sàng lọc zero-shot trên VN3K (`tools/zeroshot_encoders.py`, kết quả ở
`logs/zeroshot_VN3K.json`): R@1 18,68, so với CLIP 0,18 và ViCLIP-OT / ViSigLIP-OT
dưới 5,4. Chỉ 1% caption dài hơn 64 token.

SigLIP 2 khác CLIP ở ba chỗ. Code xử lý như sau (`model/siglip_model.py`):

| | CLIP | SigLIP 2 trong ViLNM |
|---|---|---|
| Vector toàn cục của ảnh | token [CLS] | đầu MAP (attention pooling), chính là vector SigLIP dùng để so với text |
| Vector toàn cục của câu | vị trí [EOS] | vị trí **cuối cùng** của chuỗi 64 token (đúng cách SigLIP được train) |
| Token cho LA / NTA | các token còn lại | 192 patch (ảnh); `head` áp lên từng vị trí (câu) |
| Dạng caption | `<sot> từ… <eot> 0…` (77) | `từ… <eos> <pad>…` (64), viết thường |
| Chuẩn hoá ảnh | mean/std của CLIP | 0,5 / 0,5 |

Ngoài ra:
- **NTA:** phép "random replace" chỉ lấy từ trong vốn từ của caption train, không
  lấy trong cả 256k token; phép "mask" dùng `<mask>` (id 4) của Gemma.
- **Bảng token embedding** (197M tham số) được đóng băng
  (`--freeze_token_embedding True`). Cách này tiết kiệm khoảng 2,4 GB cho gradient
  và trạng thái Adam.
- **Gradient checkpointing** (`--grad_ckpt True`) là bắt buộc trên 6 GB.
- **Lưới vị trí 16×16** được nội suy (bicubic) một lần sang kích thước train.
- Trọng số nằm ở `pretrained/hf` (HF_HOME). Lần đầu chạy sẽ tự tải nếu chưa có.

Kiểm tra trước khi train (khoảng 1 phút):

```bash
unset LD_LIBRARY_PATH
.venv/bin/python tools/verify_siglip_port.py
#  [1] khớp HF gốc (chênh 0)  [2] tokenize giống processor  [3] zero-shot qua pipeline
.venv/bin/python tests/test_modules.py        # 27 test, gồm 7 test SigLIP
```

**Độ phân giải ảnh.** Đây là quyết định cần chốt trước khi train. Zero-shot (chưa
train, chỉ dùng vector toàn cục) trên tập test VN3K:

| `--img_size` | 256 256 | 384 256 | 384 192 | 320 160 | 256 128 | 384 128 (paper) |
|---|---|---|---|---|---|---|
| R@1 | 18,30 | 16,92 | 14,78 | 11,42 | 7,98 | 6,85 |

SigLIP được train trên ảnh vuông, nên ảnh càng hẹp chiều ngang thì càng tụt; ảnh
VN3K có tỉ lệ cao/rộng trung vị khoảng 2,7. Fine-tune sẽ bù lại một phần.

**Bộ nhớ và thời gian** (RTX 3050 6 GB, `--amp True`, 8 worker, có gradient
checkpointing; không bật thì cả batch 8 cũng OOM). File kết quả ở
`logs/bench_siglip_*.json`:

| `--img_size` | batch | VRAM | s/bước | 1 epoch | 30 epoch |
|---|---|---|---|---|---|
| 384 128 | 16 | 4,2 GB | 0,46 | 4 phút | 2h02 |
| 384 128 | **32** | 4,5 GB | 0,76 | 3 phút | **1h41** |
| 384 128 | 48 | 4,6 GB | 1,06 | 3 phút | 1h35 |
| 384 128 | 64 | OOM | | | |
| 256 256 | 16 | 4,3 GB | 0,54 | 5 phút | 2h30 |
| 256 256 | **32** | 4,6 GB | 0,91 | 4 phút | **2h07** |
| 256 256 | 48 | OOM | | | |

Mỗi lần đánh giá mất khoảng 52 giây (đã tính vào cột 30 epoch, với 5 epoch đánh
giá một lần). Batch 32 là lựa chọn hợp lý: nhiều negative cho loss tương phản hơn
batch 8 khi train CUHK, mà vẫn còn dư khoảng 1,4 GB VRAM.

**Chạy:**

```bash
# một lượt đầy đủ, chia phiên như CUHK
NAME=vn3k_siglip_full DATASET=VN3K BACKBONE=siglip2 BATCH=32 NUM_EPOCH=30 EPOCHS=0 \
    tools/train_session.sh --img_size 384 128

# có nhiễu câu 50%
NAME=vn3k_siglip_sent50 DATASET=VN3K BACKBONE=siglip2 BATCH=32 NUM_EPOCH=30 EPOCHS=0 \
    tools/train_session.sh --img_size 384 128 --noise_file noise_sentence/VN3K-VI_0.5.npy

# đánh giá lại một checkpoint
.venv/bin/python test.py --config_file logs/VN3K/vn3k_siglip_full/configs.json \
    --checkpoint logs/VN3K/vn3k_siglip_full/best.pth
```

Mỗi lượt chạy ghi `best.pth` khoảng 1,5 GB và `last.pth` khoảng 3 GB (có trạng thái
optimizer). Xoá `last.pth` khi lượt chạy đã xong và không cần resume nữa.

Một lượt 30 epoch chỉ mất khoảng 2 giờ, nên có thể đặt `EPOCHS=0` (chạy một mạch)
thay vì chia phiên 4 epoch. Mọi tham số định hình lượt chạy (`--img_size`, batch,
backbone, file nhiễu…) phải giữ nguyên giữa các phiên của cùng một `NAME`.

### 9.2 Chạy ablation và xuất báo cáo

Lưới ablation gồm **6 cấu hình × các mức nhiễu**. Mỗi ô là một lượt `train.py`
bình thường, nằm ở `logs/VN3K/abl_<cấu hình>_<nhiễu>/`.

| Cấu hình | Tắt gì |
|---|---|
| `full` | không tắt |
| `no_nta` | `--use_nta False` |
| `nta_random` | `--nta_random True`: vẫn loại ~20% token nhưng chọn ngẫu nhiên, không theo ảnh (đối chứng: NTA có tác dụng vì chọn đúng từ nhiễu hay chỉ như dropout) |
| `no_la` | `--loss_on_local False --sim_alpha 1 --sim_beta 0` |
| `no_jii` | `--use_jii False` (chỉ còn NDF + ID) |
| `baseline` | tắt cả ba |

Mức nhiễu: `clean`, `sent10`…`sent80` (tráo câu), `color10`…`color70` (đổi màu).
NTA xử lý nhiễu *trong* câu (vài từ sai), nên nó được kiểm chứng trên nhiễu màu;
nhiễu tráo câu (cả câu sai) dùng cho LA/JII.

```bash
.venv/bin/python tools/ablation.py --status        # xem lượt nào xong/dở/chưa chạy
.venv/bin/python tools/ablation.py                 # đợt 1 (NTA): {full, no_nta, nta_random} × {clean, color30, color50}, ~17 h
.venv/bin/python tools/ablation.py --hours 6       # chỉ chạy 6 giờ rồi lưu và dừng
.venv/bin/python tools/ablation.py --configs full no_la no_jii baseline --noise clean sent50   # đợt sau
.venv/bin/python tools/ablation_report.py          # bảng + biểu đồ, chạy lúc nào cũng được
.venv/bin/python tools/ablation.py --keep_all ...  # giữ hết: last.pth + snapshots/ (~7 GB/lượt)
.venv/bin/python tools/nta_inspect.py --runs abl_full_color50 abl_no_nta_color50
```

- `--keep_all` (= `--keep_last` + `--save_snapshots True`): `snapshots/epoch_XX.pth` là
  weight ở mỗi lần đánh giá, `snapshots/nta_epoch_XX.pt` là r' và caption NTA tạo ra sau
  mỗi epoch. Muốn train tiếp quá 30 epoch: `--num_epoch 40 --resume .../last.pth`.
- `--keep_nta` (bản gọn cho việc nghiên cứu NTA, ~1–2 GB/lượt): giữ `nta_epoch_XX.pt` mọi
  epoch và weight epoch cuối (`--save_snapshots True --snapshot_weights final`);
  `best.pth` chỉ được giữ khi lần đánh giá tốt nhất không phải epoch cuối; `last.pth` bị xoá.
  `--rest_minutes 10` cho máy nghỉ giữa hai lượt. Ví dụ (mốc 20%, đánh giá mỗi epoch):
  `tools/ablation.py --configs full no_nta nta_random --noise sent20 color20 --keep_nta --rest_minutes 10 --eval_period 1`.
- `tools/nta_inspect.py` soi NTA bằng nhiễu màu (biết chính xác từ nào sai):
  `examples.png` (bản đồ nhiệt từ–ảnh, r' từng từ), `detection.png`, `top_words.png`,
  `summary.md`. Ngoài AUC còn hai phép đối chứng: AUC chỉ theo vị trí trong câu, và
  "cùng chỗ": r' của màu sai so với màu đúng ở cùng vị trí, cùng ảnh.
  `--ckpt snapshots/epoch_10.pth` để xem một epoch bất kỳ.
- `tools/module_metrics.py`: metric riêng cho từng module, tính từ weight epoch cuối
  (lưu đệm trong `<lượt>/module_metrics.json`, kết quả ở `results/module_metrics/`).
  JII: khoảng cách ảnh đúng − ảnh sai giống nhất, độ gom cụm cùng người. LA: từ chỉ
  bộ phận (tóc/áo/quần/giày) có nhìn đúng vùng ảnh không. Học thuộc nhiễu: phép thử
  sent50 trên tập train. Cần GPU rảnh (khoảng 2–3 phút/lượt); `--limit 30 --device cpu`
  để thử nhanh.
- `tools/sim_weights.py`: chấm lại các lượt có LA với trọng số
  S = α·S_global + (1−α)·S_local khác nhau (mặc định α ∈ {0; 0,25; 0,5; 0,75; 1}),
  không train lại. Lưu đệm trong `<lượt>/sim_weights.json`, kết quả ở `results/sim_weights/`.

- Gọi lại `tools/ablation.py` bao nhiêu lần cũng được: lượt đã xong thì bỏ qua,
  lượt đang dở thì resume.
- **Dừng:** nhấn Ctrl+C, hoặc `touch logs/VN3K/STOP_ABLATION` từ terminal khác.
  Lượt đang chạy sẽ lưu giữa epoch rồi thoát.
- Khi một lượt xong, `last.pth` (3 GB) bị xoá. Những file được giữ lại: `best.pth`,
  `history.json` (kết quả mọi lần đánh giá) và `nta_final.pt` (điểm nhiễu NTA cuối).
  Thêm `--keep_last` nếu muốn giữ `last.pth`.
- MLflow: experiment `VN3K-ablation`.

Báo cáo nằm ở `results/ablation_VN3K/`:

| File | Nội dung |
|---|---|
| `report.md` | ma trận R@1 / mAP / mINP; mức tụt khi bỏ từng module; global và local; NTA bắt nhiễu |
| `results.csv` | số liệu thô, mỗi lượt một dòng |
| `r1_vs_noise.png` | R@1 theo tỉ lệ nhiễu thật, mỗi cấu hình một đường |
| `delta_r1.png` | R@1(Đầy đủ) − R@1(bỏ module), theo từng mức nhiễu |
| `curves.png` | R@1 qua các epoch |

Các quy ước trong báo cáo:
- **Số chính lấy ở epoch cuối.** Best-on-test ghi trong ngoặc. VN3K không có tập
  val, nên chọn epoch tốt nhất trên tập test sẽ làm đẹp số liệu, và mức đẹp lên
  khác nhau giữa các cấu hình.
- **Trục nhiễu dùng tỉ lệ đo được**, không dùng tên file (xem lưu ý về `noisy_color` ở trên).
- **NTA bắt nhiễu** chỉ tính cho nhiễu màu. Với nhiễu màu ta biết chính xác từ nào
  bị đổi, nên so được với điểm r' mà NTA gán cho từng token:
  - AUC 0,5 là đoán bừa, 1,0 là hoàn hảo.
  - Top-1 là tỉ lệ token "nhiễu nhất" theo NTA đúng là từ bị đổi.
  - Với nhiễu câu không tính được, vì r' được trừ trung bình trong từng câu
    (Eq. 4), nên không so sánh được giữa các câu.

---

## 10. Xử lý sự cố

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` | `LD_LIBRARY_PATH` của conda (mục 2.2) → `unset LD_LIBRARY_PATH` |
| `torch.OutOfMemoryError` | giảm `BATCH`; đóng ứng dụng khác đang dùng VRAM (`nvidia-smi`); chạy `tools/benchmark.py` |
| `'data/CUHK-PEDES/reid_raw.json' is not available` | chép `reid_raw.json` ra thẳng `data/CUHK-PEDES/` |
| `WARNING --num_epoch changed since the checkpoint was written` | đang resume với tham số khác. Dừng lại, dùng đúng tham số cũ hoặc đổi `NAME`. |
| `... holds weights only (saved with --save_optimizer False)` | checkpoint cũ hoặc tạo với `--save_optimizer False`; resume được nhưng optimizer/NTA bắt đầu lại từ đầu |
| Phiên vừa bắt đầu đã dừng ngay | file `STOP` cũ được tự xoá khi phiên bắt đầu; nếu vẫn dừng, kiểm tra `--max_hours` / `--stop_after_*` |
| Ctrl+C mà không dừng | đang chờ bước hiện tại hoặc đang eval (tối đa khoảng 1 phút). Ctrl+C lần 2 sẽ thoát ngay nhưng không lưu. |

---

## 11. Bảng tham số mới

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--stop_after_epochs N` | 0 | xong N epoch trong phiên này thì lưu và thoát |
| `--stop_after_steps N` | 0 | sau N bước thì lưu (giữa epoch) và thoát |
| `--max_hours H` | 0 | quá H giờ thì lưu và thoát |
| `--ckpt_every_minutes M` | 30 | ghi `last.pth` giữa epoch mỗi M phút (0 = chỉ ghi ở cuối epoch) |
| `--save_optimizer` | **True** (trước đây False) | lưu đủ trạng thái để resume chính xác |
| `--mlflow` | False | bật MLflow |
| `--mlflow_uri` | `sqlite:///mlflow.db` | nơi lưu; có thể là `http://host:5000` |
| `--mlflow_experiment` | tên dataset | |
| `--mlflow_system_metrics` | True | ghi CPU/RAM/GPU |
| `--noise_file` | (trống) | file nhiễu `.npy`/`.json` cho caption train (mục 9) |
| `--backbone` | `clip` | `siglip2` cho VN3K (mục 9.1) |
| `--hf_model` | `google/siglip2-base-patch16-256` | model SigLIP 2 trên Hugging Face |
| `--freeze_token_embedding` | True | siglip2: đóng băng bảng token 256k |
| `--grad_ckpt` | True | siglip2: gradient checkpointing (bắt buộc trên 6 GB) |
| `--text_length` | 0 = theo backbone | 77 cho clip, 64 cho siglip2 |

Thay đổi hành vi so với trước:
- epoch cuối cùng luôn được đánh giá, kể cả khi không chia hết cho `--eval_period`;
- DataLoader dùng `pin_memory` khi có CUDA.
