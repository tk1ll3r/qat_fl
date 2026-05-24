# QAT-FL: Ghi Chú Rebuild Từ Paper

Paper: **Communication Efficient Federated Learning With Quantization-Aware Training Design**

DOI: `10.1109/TMLCN.2025.3635050`

Ghi chú về phạm vi:

- Các phần **mô hình hệ thống, QAT-FL, công thức hội tụ, non-uniform quantization, adaptive-bits** được rút trực tiếp từ nội dung paper.
- Các phần **kiến trúc repo, checklist implementation, logging, test, quyết định engineering** là phần đề xuất thêm để tự xây lại mã nguồn. Paper không cung cấp repo hay cấu trúc code cụ thể.
- Một số công thức trong IEEE text dùng ký hiệu hơi dễ nhầm, đặc biệt `s`: có lúc được dùng như số mức lượng tử hóa, có lúc người đọc dễ hiểu nhầm thành scale. Khi triển khai code nên tách thành `num_levels`, `num_bits`, `scale`, `zero_point`.

## 1. Chủ Đề Chính

Paper giải quyết bài toán **giảm chi phí truyền thông trong Federated Learning bằng lượng tử hóa mô hình**, nhưng khác với phần lớn hướng trước đó ở chỗ: paper không chỉ lượng tử hóa sau khi train xong local model, mà đưa **Quantization-Aware Training (QAT)** vào quá trình huấn luyện liên kết.

Tên phương pháp chính:

```text
QAT-FL: Quantization-Aware Training Federated Learning
```

Ý tưởng một câu:

```text
Client huấn luyện mô hình sao cho mô hình quen trước với lỗi lượng tử hóa,
rồi mới gửi model delta đã lượng tử hóa về server.
```

Mục tiêu:

- Giữ chi phí truyền thông thấp bằng low-bit quantization.
- Giảm mất mát accuracy do lượng tử hóa.
- Làm mô hình hội tụ tốt hơn so với Post-Training Quantization trong FL.
- Mở rộng thêm non-uniform quantization và adaptive-bits quantization.

## 2. Bài Toán Federated Learning Trong Paper

Paper xét centralized Federated Learning gồm:

```text
N client phân tán
1 central server
```

Client `i` có dataset cục bộ `D_i`, kích thước `D_i`. Toàn bộ dữ liệu:

```text
D = union_i D_i
|D| = sum_i |D_i|
```

Paper giả định các dataset không giao nhau:

```text
D_i ∩ D_j = ∅, với i != j
```

Loss cục bộ của client `i`:

```text
F_i(w) = (1 / D_i) * sum_{j in D_i} f(w, x_j, y_j)
```

Loss toàn cục trong paper:

```text
F(w) = (1 / N) * sum_{i in N} F_i(w)
```

Mục tiêu tối ưu:

```text
w* = argmin_w F(w)
```

Trong FL truyền thống, mỗi global round gồm hai bước:

1. **Local update**: client cập nhật mô hình trên dữ liệu cục bộ.
2. **Global aggregation**: server tổng hợp mô hình từ các client.

Local SGD:

```text
w~_{k-1,t}^{(i)}
  = w~_{k-1,t-1}^{(i)} - eta * grad F_i(w~_{k-1,t-1}^{(i)})
```

FedAvg theo kích thước dữ liệu trong phần system model:

```text
w_{k,0} = sum_{i=1..N} (D_i / D) * w~_{k-1,tau}^{(i)}
```

Sau đó server gửi lại global model cho client:

```text
w~_{k,0}^{(i)} = w_{k,0}
```

Điểm nghẽn: mỗi round cần truyền model parameters giữa client và server. Khi mô hình lớn và bandwidth hạn chế, truyền thông trở thành chi phí chính.

## 3. Vì Sao Cần Quantization

Quantization biến tham số float thành biểu diễn ít bit hơn.

Ví dụ:

```text
float32 -> int8 / int4 / int2
```

Trong FL, client có thể gửi model hoặc model delta ở dạng đã lượng tử hóa để giảm số bit truyền đi.

Với vector float `v in R^d`, quá trình lượng tử hóa gồm:

```text
v -> q -> v'
```

Trong đó:

- `v`: vector float trước lượng tử hóa.
- `q`: vector fixed-point/integer sau lượng tử hóa.
- `v'`: vector float sau giải lượng tử hóa.
- `scale`: bước lượng tử hóa.
- `zero_point`: điểm lệch để biểu diễn số 0.

Uniform stochastic quantization trong paper được mô tả bằng dạng:

```text
q_i =
  floor(v_i / scale) + z
  hoặc
  floor(v_i / scale) + z + 1
```

Dequantization:

```text
v'_i = (q_i - z) * scale
```

Với QSGD/uniform linear quantizer, paper nêu:

```text
scale = (v_max - v_min) / (q_max - q_min)
z     = q_max - round(v_max / scale)
```

Ví dụ 8-bit:

```text
symmetric:     q_min = -128, q_max = 127
non-symmetric: q_min = 0,    q_max = 255
```

Số bit cần truyền cho vector tham số kích thước `d` theo paper:

```text
C_s = d * ceil(log2(s)) + d + 32
```

Trong đó paper dùng:

- `s`: số mức lượng tử hóa/quantization levels.
- `d * ceil(log2(s))`: bit để biểu diễn chỉ số mức lượng tử hóa.
- `d`: bit dấu cho từng tham số.
- `32`: bit để truyền scalar/scale.

Khi viết code nên tránh dùng `s` chung chung:

```text
num_levels = 2^num_bits
num_bits   = số bit lượng tử hóa
scale      = bước lượng tử hóa
zero_point = offset
```

## 4. PTQ-FL Và Điểm Yếu

Phần lớn quantized FL trước paper này nằm ở mức **Post-Training Quantization (PTQ)**.

Luồng PTQ-FL:

```text
client train local full precision
-> lượng tử hóa model/model delta sau khi train
-> gửi về server
-> server giải lượng tử hóa
-> FedAvg
```

Vấn đề:

- Model không được huấn luyện để thích nghi với lượng tử hóa.
- Khi số bit thấp, lỗi lượng tử hóa lớn.
- Lỗi lượng tử hóa làm loss tăng, accuracy giảm.
- Với low-bit quantization, training có thể hội tụ chậm hoặc không hội tụ tốt.

Đây là lý do paper đưa QAT vào FL.

## 5. Tư Duy Chính Của QAT-FL

Thay vì:

```text
train full precision -> cuối round mới lượng tử hóa để gửi
```

QAT-FL làm:

```text
train bình thường -> train thêm với fake quantization -> gửi delta đã lượng tử hóa
```

Fake quantization mô phỏng lượng tử hóa ngay trong forward pass. Nhờ đó, trong lúc train, mô hình đã thấy trước lỗi lượng tử hóa sẽ xảy ra khi truyền thông.

Kết quả mong muốn:

- Mô hình học các trọng số thân thiện hơn với lượng tử hóa.
- Khi gửi low-bit model delta, sai số gây hại ít hơn.
- Accuracy sau lượng tử hóa cao hơn PTQ-FL.

## 6. Quantization-Aware Training Và STE

QAT chèn fake-quantization node vào neural network.

Forward:

```text
w_q     = Quantize(w)
w_tilde = Dequantize(w_q)
output  = layer(x, w_tilde)
```

Backward gặp vấn đề vì hàm `round`, `floor`, `clamp` gần như không khả vi. Nếu backprop đúng toán học, gradient qua quantization gần như bằng 0.

Paper dùng **Straight-Through Estimator (STE)**:

```text
∂L / ∂w_tilde_{i,l} = ∂L / ∂w_{i,l}
```

Diễn giải:

- Forward vẫn dùng trọng số đã fake-quantize.
- Backward coi quantization như identity function.
- Gradient vẫn cập nhật được tham số gốc.

Trực giác paper nêu:

- QAT giúp mô hình tránh các sharp/narrow local minima nhạy với lượng tử hóa.
- Mô hình có xu hướng tìm vùng nghiệm phẳng hơn, nơi quantized weights vẫn cho loss thấp.

## 7. Thiết Kế QAT-FL Trong Paper

Trong mỗi global round `k`:

1. Server chọn ngẫu nhiên `r` client trong tổng số `N` client.
2. Server gửi global model `w_k` cho các client được chọn.
3. Mỗi client train `tau` local update round bình thường.
4. Mỗi client train thêm `M` QAT round với fake-quantization.
5. Client gửi model delta đã lượng tử hóa:

```text
Q(w_{k,tau+M}^{(i)} - w_k)
```

6. Server giải lượng tử hóa và tổng hợp:

```text
w_{k+1}
  = w_k + (1/r) * sum_{i in S_k} Q_de(Q(w_{k,tau+M}^{(i)} - w_k))
```

Local update bình thường:

```text
w_{k,t+1}^{(i)}
  = w_{k,t}^{(i)}
    - eta_{k,t} * grad_tilde f_i(w_{k,t})
```

QAT local update:

```text
w_{k,tau+m+1}^{(i)}
  = w_{k,tau+m}^{(i)}
    - eta_{k,tau+m} * grad_tilde f_i(Q_fake(w_{k,tau+m}))
```

Trong đó:

```text
t = 0, 1, ..., tau - 1
m = 0, 1, ..., M - 1
```

## 8. Algorithm QAT-FL

Pseudo-code bám theo Algorithm 1 của paper:

```text
Input:
  K: tổng số global iterations
  eta: learning rate
  tau: số local update round bình thường
  M: số local QAT round
  r: số client được chọn mỗi round

Khởi tạo global model w_0

for k = 0..K-1:
    Chọn ngẫu nhiên r client: S_k
    Server gửi w_k cho các client trong S_k

    for mỗi client i trong S_k:
        w_local = w_k

        for t = 0..tau-1:
            Tính stochastic gradient không chèn fake-quantization
            w_local = w_local - eta * grad_tilde f_i(w_local)

        for m = 0..M-1:
            Tính stochastic gradient với fake-quantization
            w_local = w_local - eta * grad_tilde f_i(Q_fake(w_local))

        delta = w_local - w_k
        Client gửi Q(delta) về server

    Server giải lượng tử hóa các delta
    w_{k+1} = w_k + average(dequantized_deltas)
```

Điểm cần giữ đúng khi reproduce:

- QAT-FL gửi **model delta**, không phải toàn bộ full model.
- Tổng local compute nên so sánh công bằng với PTQ-FL. Paper so sánh PTQ-FL với `tau + M` local update round.
- QAT round chỉ là một phần trong local training, không phải fine-tune tập trung sau khi FL kết thúc.

## 9. Non-Uniform Quantization

Paper nói uniform quantization có thể không tối ưu vì tham số mô hình thường phân bố không đều. Nhiều trọng số tập trung quanh một số vùng nhất định, ví dụ quanh 0.

Do đó paper mở rộng sang **non-uniform quantization**.

Forward non-uniform trong paper:

```text
q_i =
  0          nếu v_i < b_1
  j          nếu b_j <= v_i < b_{j+1}
  2^n - 1    nếu v_i >= b_{2^n - 1}
```

Paper liên hệ cách này với Lloyd-Max quantization.

Điều kiện Lloyd-Max:

```text
b_j = (l_j + l_{j+1}) / 2
```

```text
l_j =
  integral_{b_{j-1}}^{b_j} w * phi(w) dw
  /
  integral_{b_{j-1}}^{b_j} phi(w) dw
```

Trong đó:

- `b_j`: ngưỡng/threshold.
- `l_j`: reconstruction level.
- `phi(w)`: phân bố của vector cần lượng tử hóa.

Ý nghĩa:

- Vùng nào có nhiều tham số hơn thì có thể đặt nhiều mức lượng tử hóa hơn.
- Lỗi lượng tử hóa giảm khi phân bố tham số lệch mạnh.

Phần paper nói có thể dùng các phương pháp lặp nhanh, ví dụ dựa trên K-means, để tính threshold và reconstruction levels.

## 10. Non-Uniform STE

Với non-uniform quantization, paper không dùng STE identity đơn giản như uniform quantization. Paper thêm một hệ số slope trong backward.

Backward:

```text
∂L / ∂w_tilde_{i,l}
  = ∂L / ∂w_{i,l} * (1 / a_j)
```

Trong đó:

```text
a_j = l_{j+1} - l_j
```

Diễn giải:

- Nếu các reconstruction levels cách nhau không đều, gradient cũng nên phản ánh độ rộng khoảng lượng tử hóa.
- Paper đề xuất tận dụng Lloyd-Max để tránh phải học tham số slope bằng heuristic hoặc neural network phức tạp.

Ghi chú implementation thêm:

- Khi code thực tế, cần chặn `a_j` quá nhỏ để tránh gradient explosion.
- Có thể dùng `eps` và `clamp` cho slope.
- Đây là quyết định engineering, paper không đưa code cụ thể.

## 11. Adaptive-Bits QAT-FL

Paper nhận xét:

- Fixed low-bit: tiết kiệm truyền thông nhưng hội tụ kém hơn.
- Fixed high-bit: hội tụ tốt hơn nhưng tốn truyền thông.

Do đó paper đề xuất **adaptive-bits quantization QAT-FL**: tự điều chỉnh số mức lượng tử hóa/số bit theo quá trình training.

Từ phân tích bound theo communicated bits, paper đưa:

```text
s_k* = sqrt(f(w_0) / f(w_k)) * s_0
```

Số bit tại global round `k`:

```text
n_b^k = ceil(log2(s_k*))
```

Trong đó:

- `s_0`: số mức lượng tử hóa ban đầu.
- `f(w_0)`: loss ban đầu.
- `f(w_k)`: loss tại round `k`.
- `n_b^k`: số bit lượng tử hóa tại round `k`.

Ý nghĩa:

- Khi training mới bắt đầu, loss còn cao, có thể dùng ít bit.
- Khi loss giảm và mô hình cần cập nhật tinh hơn, số bit tăng dần.
- Cách này nhằm đạt cùng hoặc tốt hơn fixed-bit QAT-FL với ít communicated bits hơn.

Ghi chú implementation thêm:

```text
num_levels_k = sqrt(initial_loss / current_loss) * initial_num_levels
num_bits_k   = ceil(log2(num_levels_k))
```

Nên giới hạn:

```text
initial_bits <= num_bits_k <= max_bits
```

Giới hạn này là đề xuất engineering để tránh số bit nhảy quá cao hoặc quá thấp; paper không mô tả chi tiết code.

## 12. Giả Định Hội Tụ

Paper chứng minh hội tụ dưới các giả định sau.

### Assumption 1: Quantizer không chệch

```text
E[Q(w) | w] = w
```

Variance bị chặn:

```text
E[||Q(w) - w||^2 | w] <= q ||w||^2
```

### Assumption 2: Loss L-smooth

Với mọi `i`:

```text
||grad f_i(w) - grad f_i(w_hat)|| <= L ||w - w_hat||
```

### Assumption 3: Stochastic gradient không chệch

```text
E[grad_tilde f_i(w)] = grad f_i(w)
```

Variance bị chặn:

```text
E[||grad_tilde f_i(w) - grad f_i(w)||^2] <= sigma^2
```

Paper chủ yếu chứng minh với i.i.d data distribution. Với non-IID, paper nói mở rộng không đơn giản vì hướng gradient cục bộ của các client bị phân tán. Tuy vậy, paper vẫn thực nghiệm trên FEMNIST non-IID để so sánh QAT-FL với PTQ-FL trong cùng điều kiện.

## 13. Bound Hội Tụ Chính

Vì neural network thường là non-convex, paper dùng trung bình norm bình phương của gradient làm chỉ báo hội tụ:

```text
(1 / (K(tau + M))) *
sum_{k=0..K-1} [
  sum_{t=0..tau-1} E||grad f(w_bar_{k,t})||^2
  +
  sum_{m=0..M-1} E||grad f(w_bar_{k,tau+m})||^2
]
```

Bound trong Theorem 1:

```text
<=
2(f(w_0) - f*) / (eta K(tau + M))
+ eta L(1 + q) * ( (n-r)/(r(n-1)) * 4 sigma^2 + sigma^2/n )
+ eta^2 * sigma^2/n * (n+1)(tau+M-1)L^2
```

Cách đọc:

- Term 1 giảm khi số global round `K` tăng.
- Term 2 phụ thuộc quantization variance `q`, stochastic gradient variance `sigma^2`, và số client được chọn `r`.
- Term 3 phụ thuộc learning rate, variance và số local/QAT steps.

Paper so sánh với PTQ-FL và nói bound nhìn bề ngoài tương tự, nhưng khác ở phần bản chất:

- PTQ-FL tối ưu full-precision model rồi mới lượng tử hóa.
- QAT-FL trong QAT round tối ưu loss với fake-quantized model.

Vì vậy QAT-FL trực tiếp tối ưu mô hình trong điều kiện sẽ bị lượng tử hóa, nên thực nghiệm cho loss thấp hơn và accuracy tốt hơn.

## 14. Bound Theo Communicated Bits

Paper xét thêm tổng số bit truyền thông `B`.

Với QSGD:

```text
q_s = d / s^2
```

Ý nghĩa:

- Tăng số mức lượng tử hóa `s` làm giảm quantization variance.
- Nhưng tăng `s` làm tăng số bit truyền mỗi round.
- Adaptive-bits cân bằng hai chiều này.

Corollary của paper dẫn tới lựa chọn:

```text
s_k* = sqrt(f(w_0) / f(w_k)) * s_0
```

và:

```text
n_b^k = ceil(log2(s_k*))
```

## 15. Setup Thực Nghiệm Trong Paper

Paper dùng:

```text
framework: Python + PyTorch
FL clients: 100 nodes
aggregation: FedAvg
optimizer: mini-batch SGD
models: CNN
datasets: MNIST, CIFAR-10, FEMNIST
```

Datasets:

- `MNIST`: phân loại chữ số viết tay.
- `CIFAR-10`: phân loại ảnh màu 10 lớp.
- `FEMNIST`: phức tạp hơn, có non-IID split theo writer, gồm 62 lớp.

Learning rate trong paper:

```text
MNIST:
  local update learning rate = 0.002
  QAT learning rate          = 0.002

CIFAR-10:
  local update learning rate = 0.01
  QAT learning rate          = 0.01

FEMNIST:
  local update learning rate = 0.01
  QAT learning rate          = 0.01
```

Số local update + QAT round:

```text
MNIST:    tau + M = 4
CIFAR-10: tau + M = 6
```

Quantization bits:

```text
MNIST fixed bits: 3, 4, 5
MNIST adaptive initial bits: 2

CIFAR-10 fixed bits: 6, 7, 8
CIFAR-10 adaptive initial bits: 6
```

Baselines trong paper:

- PTQ-FL với QSGD.
- PTQ-FL với non-uniform quantization.
- QAT-FL với QSGD.
- QAT-FL với non-uniform quantization.
- Adaptive-bits QAT-FL.

Paper có nhắc đã thử SignSGD và FedPAQ, nhưng không chọn SignSGD làm baseline chính vì độ nhạy learning rate và kết quả không ổn định trong thí nghiệm của họ. FedPAQ được xem gần với PTQ-FL fixed-bit uniform quantization.

## 16. Kết Quả Paper Báo Cáo

Kết quả chính:

- QAT-FL có training loss thấp hơn PTQ-FL.
- QAT-FL có testing accuracy cao hơn PTQ-FL trong cùng mức bit.
- Trên MNIST, QAT-FL cải thiện khoảng `2%-9%` accuracy.
- Trên CIFAR-10, QAT-FL cải thiện khoảng `2%-4%` accuracy.
- Trên FEMNIST non-IID, QAT-FL vẫn cho xu hướng tốt hơn PTQ-FL trong cùng setting.
- Non-uniform QAT-FL tốt hơn non-uniform PTQ-FL.
- Adaptive-bits QAT-FL đạt hiệu quả truyền thông tốt hơn fixed-bit QAT-FL.

Paper cũng nêu ví dụ về inference efficiency trên FEMNIST:

- PTQ inference khoảng `1.499 ms`.
- QAT-FL 4-bit inference khoảng `605 us`.
- Convolution layers giảm từ khoảng `800 us` xuống `170 us`.

## 17. Những Phần Cần Tự Thiết Kế Khi Xây Repo

Paper không cung cấp repo. Các phần dưới đây là đề xuất thêm để triển khai lại.

### Kiến trúc repo đề xuất

```text
qat_fl/
  configs/
    mnist_qat.yaml
    cifar10_qat.yaml
    femnist_qat.yaml
  data/
    partition.py
    femnist.py
  models/
    cnn_mnist.py
    cnn_cifar.py
    cnn_femnist.py
  quantization/
    uniform.py
    qsgd.py
    lloyd_max.py
    fake_quant.py
    ste.py
    adaptive_bits.py
  fl/
    client.py
    server.py
    fedavg.py
    trainer.py
  experiments/
    run_mnist.py
    run_cifar10.py
    run_femnist.py
  utils/
    metrics.py
    seed.py
    logging.py
    checkpoint.py
  tests/
    test_quantizers.py
    test_fake_quant.py
    test_fedavg.py
  train.py
  README.md
  QAT.md
```

### Module quan trọng

- `quantization/fake_quant.py`: fake quantization module.
- `quantization/ste.py`: STE/autograd trick.
- `quantization/qsgd.py`: uniform/QSGD quantizer cho communication delta.
- `quantization/lloyd_max.py`: non-uniform Lloyd-Max/K-means quantizer.
- `quantization/adaptive_bits.py`: cập nhật số bit theo loss.
- `fl/client.py`: local SGD + QAT local update.
- `fl/server.py`: chọn client + aggregate.
- `fl/fedavg.py`: FedAvg.

## 18. Gợi Ý Implement Fake Quantization

Đây là phần đề xuất code, không phải code trong paper.

Trong PyTorch có thể dùng STE trick:

```python
w_fake = w + (w_dequantized - w).detach()
```

Ý nghĩa:

- Forward value là `w_dequantized`.
- Backward gradient đi qua như identity đối với `w`.

Pseudo-code:

```python
def fake_quantize_ste(w, num_bits):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    w_min = w.min()
    w_max = w.max()
    scale = (w_max - w_min) / (qmax - qmin + eps)
    zero_point = qmax - torch.round(w_max / scale)

    q = torch.round(w / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    w_dequantized = (q - zero_point) * scale

    return w + (w_dequantized - w).detach()
```

Cần chú ý:

- Nếu `w_max == w_min`, `scale` gần 0, cần `eps`.
- Nên bắt đầu với per-layer/per-tensor quantization.
- Communication quantization và fake quantization nên cùng logic để tránh mismatch.

## 19. Baseline Nên Implement

Thứ tự triển khai hợp lý:

1. **FedAvg full precision**

```text
client train local -> gửi full precision delta -> server FedAvg
```

2. **PTQ-FL uniform/QSGD**

```text
client train local full precision -> quantize delta -> dequantize -> FedAvg
```

3. **QAT-FL uniform/QSGD**

```text
normal local SGD -> QAT local SGD -> quantize delta -> FedAvg
```

4. **Non-uniform QAT-FL**

```text
normal local SGD -> non-uniform fake quant QAT -> non-uniform communication quantization
```

5. **Adaptive-bits QAT-FL**

```text
num_bits_k từ adaptive rule
-> QAT với num_bits_k
-> quantize delta với num_bits_k
```

## 20. Metrics Nên Log

Đây là phần đề xuất thêm để dễ reproduce và debug.

```text
round
train_loss
test_loss
test_accuracy
num_bits
num_levels
communication_bits_this_round
cumulative_communication_bits
quantization_error_norm
relative_quantization_error
model_delta_norm
selected_client_ids
```

Quantization error:

```text
||Q_de(Q(delta)) - delta||_2
```

Relative quantization error:

```text
||Q_de(Q(delta)) - delta||_2 / (||delta||_2 + eps)
```

## 21. Các Quyết Định Engineering Cần Rõ

Các điểm này là phần tự thiết kế khi rebuild, không phải paper đưa code sẵn.

### Weighted hay uniform FedAvg

Paper system model viết weighted aggregation theo dataset size, còn Algorithm 1 dùng trung bình trên selected clients:

```text
w_{k+1} = w_k + (1/r) * sum_i dequantized_delta_i
```

Repo nên hỗ trợ cả hai:

```text
aggregation = "uniform"
aggregation = "weighted_by_data_size"
```

Khi reproduce Algorithm 1, dùng uniform average trên selected clients.

### Quantize weights hay delta

Paper gửi:

```text
Q(w_local - w_global)
```

Do đó implementation nên quantize **model delta**.

### Fake quantize weights hay cả activations

Paper mô tả QAT nói chung có thể chèn fake-quantization sau convolution/fully connected layers và nhắc weights/activations. Nhưng bài toán truyền thông trong FL tập trung vào model parameters.

Đề xuất triển khai:

```text
phase 1: fake quantize weights
phase 2: thêm activation fake quantization nếu cần
```

### Per-tensor hay per-layer

Paper không chốt chi tiết code. Đề xuất:

```text
per-layer/per-parameter-tensor quantization
```

Lý do: mỗi layer có range tham số khác nhau.

## 22. Checklist Triển Khai

Phase 1:

- Xây FedAvg full precision.
- Load MNIST.
- Chia IID cho 100 clients.
- Train CNN nhỏ.
- Kiểm tra loss giảm và accuracy tăng.

Phase 2:

- Implement uniform/QSGD quantizer cho model delta.
- Thêm PTQ-FL baseline.
- Log communication bits và quantization error.

Phase 3:

- Implement fake quantization với STE.
- Thêm `M` QAT local rounds.
- So sánh PTQ-FL và QAT-FL với cùng tổng `tau + M`.

Phase 4:

- Thêm CIFAR-10.
- Thêm FEMNIST hoặc non-IID partition thay thế nếu FEMNIST khó tải.

Phase 5:

- Implement Lloyd-Max/K-means non-uniform quantizer.
- Implement non-uniform STE.

Phase 6:

- Implement adaptive-bits rule.
- Vẽ loss/accuracy theo cumulative communicated bits.

## 23. Thí Nghiệm Tối Thiểu Để Kiểm Tra Rebuild

Đề xuất bắt đầu với MNIST:

```text
N = 100 clients
r = 10 selected clients per round
K = 50 global rounds
tau + M = 4
optimizer = SGD
learning rate = 0.002
model = small CNN
bits = 4
```

PTQ-FL:

```text
tau = 4
M = 0
bits = 4
```

QAT-FL:

```text
tau = 2
M = 2
bits = 4
```

Kỳ vọng:

```text
QAT-FL có final loss thấp hơn hoặc test accuracy cao hơn PTQ-FL
trong cùng số bit và cùng tổng local steps.
```

Adaptive-bits:

```text
initial_bits = 2
max_bits = 5 hoặc 8
num_bits_k = ceil(log2(sqrt(loss_0 / loss_k) * 2^initial_bits))
```

Nên vẽ:

```text
x-axis: cumulative communicated bits
y-axis: train loss / test accuracy
```

## 24. Những Điểm Dễ Sai Khi Reproduce

- So sánh không công bằng nếu PTQ-FL và QAT-FL có tổng local steps khác nhau.
- Fake quantization và communication quantization dùng scale/logic khác nhau.
- Low-bit quantization rất nhạy với learning rate.
- Non-IID làm FedAvg dao động mạnh hơn, cần chạy nhiều seed.
- Adaptive bits dùng loss quá nhiễu sẽ làm số bit nhảy thất thường.
- Lloyd-Max/K-means mỗi client mỗi round có thể chậm; nên làm sau uniform QAT-FL.

## 25. Tóm Tắt Một Câu

QAT-FL giảm chi phí truyền thông trong Federated Learning bằng cách cho client huấn luyện với fake quantization trước khi gửi quantized model delta, nhờ đó mô hình thích nghi với low-bit quantization và đạt loss/accuracy tốt hơn PTQ-FL dưới cùng communication budget.
