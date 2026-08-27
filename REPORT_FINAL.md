# Báo Cáo Kỹ Thuật FinOps: Tối Ưu Hóa Chi Phí GPU & Hạ Tầng AI (NimbusAI)

> **Tác giả:** Mai Việt Anh 
> **Dự án:** NimbusAI GPU Infrastructure Cost Optimization  
> **Bài Lab:** Lab 25 — GPU FinOps Optimization Workshop  
> **Thời điểm dữ liệu:** Tháng 6/2026 Snapshot  

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

NimbusAI đã tiến hành kiểm toán toàn diện hạ tầng tính toán GPU và nhật ký sử dụng inference token. Qua việc áp dụng phương pháp luận GPU FinOps đa tầng (đo lường bằng đơn vị `$/1M-token` thay vì `$/GPU-giờ` truyền thống), chúng tôi đã xác định được các điểm lãng phí nghiêm trọng và thiết lập lộ trình tối ưu hóa với kết quả vượt bậc:

| Chỉ số FinOps | Trước tối ưu (Baseline) | Sau tối ưu (Optimized) | Mức cắt giảm / Tiết kiệm |
|---|---|---|---|
| **Tổng chi phí hàng tháng** | **$27,133 / tháng** | **$14,626 / tháng** | **-$12,507 / tháng (Tiết kiệm 46.1%)** |
| **Đơn giá Inference (`$/1M-token`)** | **$6.488 / 1M token** | **$1.126 / 1M token** | **Giảm 82.6% đơn giá phục vụ** |
| **Chi phí Inference hàng ngày** | $48.87 / ngày | $8.48 / ngày | -$40.39 / ngày |
| **Tổng lượng token phục vụ** | 7,533,027 token / ngày | 7,533,027 token / ngày | Giữ nguyên 100% throughput |
| **Tag Coverage** | 92.0% | 92.0% | **Đủ điều kiện Chargeback (≥80%)** |
| **Dấu chân Carbon (Interruptible)** | 1,606.3 kg CO2e / tháng | 126.8 kg CO2e / tháng | **Giảm 92.1% lượng phát thải CO2** |

```
Chi phí GPU Hàng Tháng ($ USD):
Baseline:  ████████████████████████████ $27,133
Optimized: █████████████░░░░░░░░░░░░░░ $14,626  (-46.1%)
```

---

## 2. Phân Tích Chi Tiết 4 Đòn Bẩy Tiết Kiệm (FinOps Levers Breakdown)

Biểu đồ thác nước (Waterfall Chart) tại `outputs/savings.png` và bảng phân bổ dưới đây thể hiện đóng góp của từng đòn bẩy trong tổng số **$12,507/tháng** tiết kiệm được:

| Đòn bẩy FinOps (Lever) | Tiết kiệm (USD/tháng) | Tỷ trọng trong tổng tiết kiệm | Cơ chế tác động & Nguyên nhân |
|---|---|---|---|
| **1. Purchasing Strategy (Spot + Reserved)** | **$10,040 / tháng** | **80.3%** | **Đóng góp lớn nhất.** Tận dụng Spot instances (chiết khấu ~40-60%) cho 5 job training/eval gián đoạn được (`interruptible=1`) với checkpointing overhead chỉ 3%. Cam kết Reserved 3 năm (chiết khấu 45%) cho 3 job inference chạy 24/7 (vượt điểm hòa vốn 55%). |
| **2. Inference Optimization (Cascade/Cache/Batch)** | **$1,212 / tháng** | **9.7%** | Cắt giảm 82.6% chi phí token bằng bộ 3 đòn bẩy: Cascade định tuyến model nhỏ ($0.20/$0.40 per 1M), Prompt Caching giảm 90% chi phí input lặp lại, và Batch API giảm 50% cho tác vụ offline. |
| **3. Right-sizing Over-provisioned GPUs** | **$655 / tháng** | **5.2%** | Phát hiện các GPU bị "GPU-Util Lie" (MFU < 30% dù GPU-Util > 90%) để hạ cấp xuống GPU phù hợp (H100 -> A100, A10G -> L4), tránh trả tiền cho phần cứng thừa. |
| **4. Kill Idle GPUs** | **$600 / tháng** | **4.8%** | Tắt bỏ các instance GPU bị bỏ quên sau khi huấn luyện xong (lãng phí $20/ngày = $600/tháng trên instance H100 chạy idle 8 giờ). |

---

## 3. Bản Chất Kỹ Thuật Của "GPU-Util Lie" & Tác Động Tài Chính

### 3.1. GPU nào bị "GPU-Util Lie"?
Trong quá trình kiểm toán hạ tầng (Mission 1), hệ thống phát hiện 2 GPU rơi vào nhóm "nói dối":
1. **`gpu-h100-4`**: `gpu_util_pct = 98.2%`, nhưng **`MFU = 0.194` (~19.4%)** và **`MBU = 0.207`**.
2. **`gpu-a10g-1`**: `gpu_util_pct = 96.9%`, nhưng **`MFU = 0.268` (~26.8%)** và **`MBU = 0.302`**.

### 3.2. Tại sao `nvidia-smi` GPU-Util 98% nhưng MFU chỉ ~20%?
- Lệnh `nvidia-smi` hoặc NVIDIA DCGM đo lường chỉ số **GPU-Util %** dựa trên **tỷ lệ thời gian mà xung nhịp GPU (kernel execution clock) có hoạt động** trong chu kỳ lấy mẫu (sample window). Nó chỉ trả lời câu hỏi: *"GPU có đang bận không?"* chứ **hoàn toàn không đo lường hiệu suất tính toán thực tế**.
- **Nguyên nhân gốc rễ (Root Causes):**
  1. **Memory Bandwidth Stall:** Trong giai đoạn *LLM Decode (Token Generation)*, cường độ tính toán (Arithmetic Intensity) rất thấp (~1–2 FLOP/byte), mô hình bị nghẽn băng thông bộ nhớ HBM nghiêm trọng (Memory-bound). Tensor Cores phải nhàn rỗi chờ nạp trọng số từ VRAM, nhưng clock GPU vẫn chạy nên `nvidia-smi` ghi nhận 98% utilization.
  2. **Kernel Launch Overhead & Small Batch Size:** Chạy batch size nhỏ hoặc nhiều kernel nhỏ rời rạc khiến GPU mất thời gian điều phối CPU-GPU driver interaction thay vì thực thi phép tính ma trận lớn.
  3. **I/O & Communication Wait:** Trong distributed training/inference, GPU bị block bởi barrier mạng (AllReduce/NCCL) hoặc đọc dữ liệu từ ổ đĩa chậm.
  4. **Non-Tensor Core Ops:** Workload sử dụng các phép toán scalar/vector FP32 hoặc custom kernels không tận dụng được ma trận phần cứng chuyên dụng FP16/BF16 Tensor Cores của H100.

### 3.3. Tác động tài chính
- Doanh nghiệp đang trả trọn gói **$2.50 / GPU-giờ** cho H100 (tương đương **$1,800 / tháng / GPU**) với kỳ vọng nhận được sức mạnh 990 TFLOPs FP16, nhưng thực tế chỉ thu về ~192 TFLOPs (1/5 năng lực phần cứng).
- Đây là dạng **chi phí ẩn vô hình (invisible waste)** lớn nhất trong các công ty AI nếu chỉ giám sát qua dashboard CPU/GPU Utilization thông thường.

---

## 4. Báo Cáo 5 Phần Mở Rộng "Your Turn" Đã Thực Hiện

Chúng tôi đã hoàn thiện toàn diện cả **5/5 phần mở rộng** với mã nguồn thực tế và số liệu đo lường cụ thể:

### Extension 1: Nâng Cấp Chính Sách `recommend_tier()` Theo Interruption Risk & Duration
- **Vị trí code:** `finops/pricing.py` & `missions/m3_purchasing.py`.
- **Cải tiến logic:** 
  1. Tích hợp đánh giá rủi ro gián đoạn theo loại GPU: H100 có tỷ lệ thu hồi spot thấp (~3–5%), trong khi A10G/L4 trên commodity cloud có thể chịu tỷ lệ ngắt cao hơn (~10–15%).
  2. Duration-Awareness: Đối với các job ngắn hạn (`job_days < 90`), chính sách ngăn chặn cam kết Reserved 3 năm để tránh rủi ro "shelf-ware", ưu tiên Spot hoặc On-Demand.
- **Kết quả đo lường:** Xác định chính xác 5 job phù hợp với Spot, 3 job inference 24/7 ổn định cam kết Reserved 3 năm, tiết kiệm **$10,040/tháng (39.1%)** chi phí mua GPU.

### Extension 2: Right-sizing Theo MBU & Chi Phí VRAM (`$/GB-VRAM`)
- **Vị trí code:** `finops/metrics.py` & `missions/m1_efficiency_audit.py`.
- **Phân tích kinh tế VRAM:**
  | GPU Type | VRAM (GB) | Giá On-Demand ($/hr) | Đơn giá VRAM (`$/GB-hr`) | Peak BW (TB/s) | Đơn giá Băng thông (`$/(TB/s)-hr`) |
  |---|---|---|---|---|---|
  | **MI300X** | 192 GB | $1.95 | **$0.0102** | 5.30 TB/s | **$0.3679** |
  | **A100** | 80 GB | $1.79 | $0.0224 | 2.00 TB/s | $0.8950 |
  | **B200** | 192 GB | $5.09 | $0.0265 | 8.00 TB/s | $0.6362 |
  | **H100** | 80 GB | $2.50 | $0.0312 | 3.35 TB/s | $0.7463 |
  | **A10G** | 24 GB | $1.00 | $0.0417 | 0.60 TB/s | $1.6667 |
- **Đo lường Right-sizing:** Với GPU `gpu-h100-4` (achieved BW chỉ 0.69 TB/s), chuyển sang A100 (2.0 TB/s BW) giúp tiết kiệm ngay **$511.20/tháng/GPU**. Tổng tiềm năng right-sizing toàn bộ pool GPU theo MBU lên tới **$3,924.00/tháng**.

### Extension 3: Kinh Tế Học Prompt Caching (`cache_is_worth_it`)
- **Vị trí code:** `finops/pricing.py` & `missions/m2_inference_levers.py`.
- **Công thức Break-Even:**
  $$\text{Break-Even Reads} = \frac{\text{Write Cost / 1M}}{\text{Read Price / 1M} \times (1 - \text{Discount})} = \frac{1.25 \times P_{\text{in}}}{P_{\text{in}} \times 0.90} \approx 1.39 \text{ lần đọc}$$
- **Kết quả đo lường:** 
  - Trong dataset `token_usage.csv`, 100% request (2,400/2,400) có token prefix tái sử dụng với tổng 1,703,990 cached tokens.
  - Hàm `cache_is_worth_it()` trả về `True` (vì số lần đọc trung bình của hệ thống là 3.5x > 1.39x break-even), chứng minh chính sách Prompt Caching mang lại lợi ích ròng cực lớn.

### Extension 4: Phân Tích & Kiểm Soát Ngân Sách Reasoning Traffic
- **Vị trí code:** `missions/m2_inference_levers.py`.
- **Kết quả đo lường thực tế:**
  - Reasoning queries (`is_reasoning=1`) chỉ chiếm **8.4% tổng số requests** (201/2,400) và **16.5% tổng tokens** (1,241,156 tokens).
  - Tuy nhiên, do nhân tố năng lượng 80×, Reasoning tiêu thụ tới **29.79 kWh / ngày** so với 1.89 kWh / ngày của standard traffic — **chiếm tới 94.0% tổng năng lượng tiêu thụ toàn hệ thống inference!**
- **Đề xuất Routing Rule:** Giới hạn Reasoning chỉ kích hoạt cho bài toán phức tạp (khi confidence score của Small Model < 0.75), thiết lập quota 10% traffic giúp bảo vệ ngân sách và giảm tải hạ tầng.

### Extension 5: Lập Lịch Nhận Thức Carbon (Carbon-Aware Scheduling)
- **Vị trí code:** `finops/sustainability.py` & `missions/m3_purchasing.py`.
- **Bảng So Sánh 5 Vùng Điện Toán (Tổng điện năng 4,227 kWh/tháng cho các job interruptible):**
  | Vùng (Region) | Cường độ Carbon (`gCO2/kWh`) | Giá điện (`$/kWh`) | Phát thải CO2 (`kgCO2e/tháng`) | Tiền điện thực tế (`$/tháng`) | Đặc điểm lưới điện |
  |---|---|---|---|---|---|
  | **europe-north1** (Na Uy) | **30** | $0.090 | **126.8 kg** | $380.43 | Thủy điện sạch nhất |
  | **us-east-wa** (Washington) | 90 | **$0.055** | 380.4 kg | **$232.49** | Rẻ nhất về tiền điện |
  | **us-west-2** (Oregon) | 120 | $0.070 | 507.2 kg | $295.89 | Cân bằng giá/sạch |
  | **us-east-1** (Virginia) | 380 | $0.120 | 1,606.3 kg | $507.24 | Mặc định (nhiều than/khí) |
  | **europe-central2** (Ba Lan) | 660 | $0.180 | 2,789.8 kg | $760.86 | Dơ nhất & đắt nhất |
- **Kết quả điều phối:** Chuyển toàn bộ 5 job training/eval gián đoạn từ `us-east-1` sang `europe-north1` giúp **cắt giảm 1,479.5 kg CO2e / tháng (Giảm 92.1%)** và tiết kiệm thêm **$126.81 / tháng** tiền điện.

---

## 5. Khuyến Nghị Chiến Lược Cho Ban Giám Đốc NimbusAI

Nếu đảm nhận vị trí FinOps Lead tại NimbusAI, 3 hành động tiên quyết tôi sẽ triển khai ngay trong Quý tới:

### 1. Hành Động 1: Thiết Lập Quản Trị Tagging Tự Động & Kích Hoạt Cơ Chế Chargeback
- **Mục tiêu:** Chuyển từ "Showback" (xem cho biết) sang "Chargeback" (trừ ngân sách thực tế của từng phòng ban).
- **Thực thi:** Tận dụng tag coverage hiện tại đã đạt 92% (vượt ngưỡng 80%), áp dụng định dạng chuẩn [FOCUS 1.x](outputs/focus_export.csv) để xuất hóa đơn chi tiết cho 4 team (`assistant`, `search`, `eval`, `rag`). Bắt buộc mọi instance mới phải có tag `team` và `project` qua CI/CD gatekeeper.


### 2. Hành Động 2: Tái Cấu Trúc Hợp Đồng Mua GPU (Spot Checkpointing + Reserved Commitments)
- **Mục tiêu:** Thu hồi ngay **$10,040 / tháng** lãng phí mua sắm.
- **Thực thi:**
  - Chuyển 100% batch training, fine-tuning và eval sang **Spot Instances** kết hợp cơ chế tự động checkpoint mỗi 30 phút.
  - Ký cam kết **Reserved 3 năm** cho cụm GPU phục vụ inference 24/7 (nơi duty cycle đạt 100% > break-even 55%).
  - Cài đặt daemon tự động thu hồi (auto-termination script) để terminate mọi instance idle quá 15 phút.

### 3. Hành Động 3: Triển Khai Cổng Định Tuyến Chi Phí Thông Minh (FinOps Inference Gateway)
- **Mục tiêu:** Giảm đơn giá phục vụ token xuống dưới **$1.20 / 1M token**.
- **Thực thi:**
  - Tích hợp proxy gateway (tương tự `LiteLLM` / vLLM Router) để tự động hóa: **Cascade Routing** (đưa prompt đơn giản về Small Model), **Prompt Caching** (cache system prompts & context documents), và **Batching** cho các request không yêu cầu thời gian thực.
  - Đặt hard budget cap theo API Key và giới hạn reasoning token theo task complexity.

---

## 6. Phụ Lục: Giải Đáp 5 Câu Hỏi Kiểm Tra Hiểu Biết Bản Chất (Oral Check Q&A)

1. **"GPU-Util 98% có nghĩa là GPU đang làm việc hiệu quả không? Tại sao?"**  
   *Trả lời:* **Không.** `GPU-Util 98%` từ `nvidia-smi` chỉ phản ánh xung nhịp GPU có lệnh đang chạy (thời gian bận), chứ không phản ánh lượng tính toán có ích (TFLOPs). Trong các tác vụ decode bộ nhớ nghẽn (Memory-bound) hoặc kernel launch rời rạc, GPU dành phần lớn thời gian chờ nạp dữ liệu từ HBM VRAM. Chỉ có chỉ số **MFU (Model FLOPs Utilization)** và **MBU (Model Bandwidth Utilization)** mới phản ánh đúng hiệu quả phần cứng.

2. **"Tại sao cần ≥ 80% tag coverage mới dám chargeback?"**  
   *Trả lời:* Chargeback là hành động tài chính trừ tiền trực tiếp vào P&L của từng team. Nếu tag coverage < 80%, một lượng lớn chi phí (>20%) là "vô thừa nhận" (untagged). Việc phân bổ mù quáng chi phí untagged sẽ gây xung đột nội bộ, mất niềm tin vào hệ thống FinOps và làm sai lệch chỉ số ROI của sản phẩm.

3. **"Nếu công ty bạn có 70% workload interruptible, bạn sẽ tối ưu purchasing như thế nào?"**  
   *Trả lời:* Với 70% workload có thể gián đoạn, chiến lược tối ưu là sử dụng **Spot Instances kết hợp công nghệ Checkpointing thường xuyên** (như PyTorch Lightning / DeepSpeed Checkpoint). Chi phí Spot rẻ hơn 40–60%, trong khi chi phí rework khi bị ngắt chỉ chiếm ~3% tổng thời gian tính toán, mang lại tỷ suất tiết kiệm ròng vượt trội so với On-Demand hoặc Reserved.

4. **"Đo bằng $/GPU-hr vs $/1M-token — khi nào con số này cho kết quả trái ngược nhau?"**  
   *Trả lời:* Trái ngược khi một đội thuê GPU đắt hơn theo giờ (ví dụ H100 giá $2.5/hr so với A10G giá $1.0/hr) nhưng nhờ tối ưu hóa phần mềm (vLLM, TensorRT-LLM, FlashAttention, batching lớn) đạt MFU cao và phục vụ được lượng token gấp 10 lần. Khi đó, `$/GPU-hr` cao hơn 2.5× nhưng `$/1M-token` lại **rẻ hơn gấp 4 lần**. Do đó, `$/1M-token` là thước đo chân thực duy nhất cho hiệu quả kinh tế của sản phẩm AI.

5. **"Tại sao LLM decode là memory-bound còn prefill là compute-bound?"**  
   *Trả lời:* 
   - **Prefill (Prompt processing):** Xử lý toàn bộ chuỗi đầu vào song song qua phép nhân ma trận $(N \times D) \times (D \times D)$, cường độ tính toán (Arithmetic Intensity) cao (~455 FLOP/byte trên H100), tận dụng tối đa Tensor Cores $\rightarrow$ **Compute-bound**.
   - **Decode (Token generation):** Sinh từng token tuần tự theo cơ chế autoregressive. Với mỗi token mới sinh ra, toàn bộ trọng số mô hình và KV-Cache phải được nạp lại từ bộ nhớ HBM vào SRAM/Register chỉ để thực hiện vài phép tính trên 1 vector $(1 \times D)$, cường độ tính toán cực thấp (~1–2 FLOP/byte) $\rightarrow$ **Memory-bound**.
