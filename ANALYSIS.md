# Phân tích kết quả — Day 17: Memory Systems for AI Agent

## 1. Kết quả benchmark

### 1.1. Standard Benchmark (`data/conversations.json`)

10 hội thoại ngắn (~10 lượt), user `dungct`, có `recall_questions` hỏi ở thread mới.

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| -------- | ----------------- | ----------------------- | -------------------- | ---------------- | --------------------- | ----------- |
| Baseline | 2,809             | 21,480                  | 0.036                | 0.705            | 0                     | 0           |
| Advanced | 1,975             | 21,841                  | 0.214                | 0.764            | 205                   | 0           |

### 1.2. Long-Context Stress Benchmark (`data/advanced_long_context.json`)

1 hội thoại 16 lượt rất dài, user `dungct_stress`, có nhiều đoạn tin tức dài.

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| -------- | ----------------- | ----------------------- | -------------------- | ---------------- | --------------------- | ----------- |
| Baseline | 551               | 24,593                  | 0.000                | 0.700            | 0                     | 0           |
| Advanced | 255               | 15,841                  | 0.333                | 0.775            | 194                   | 1           |

---

## 2. Phân tích

### 2.1. Vì sao Advanced tốn hơn Baseline ở hội thoại ngắn

Ở Standard Benchmark, Advanced xử lý **21,841 prompt tokens** so với **21,480** của Baseline — cao hơn khoảng **2%**.

Nguyên nhân: Advanced luôn kéo theo `User.md` trong prompt ở mỗi lượt, kể cả khi hội thoại còn ngắn và chưa cần compact. Đây là **"memory tax"** — cái giá cố định phải trả để có persistent memory xuyên session.

Đổi lại, Advanced đạt cross-session recall **0.214** so với **0.036** của Baseline. Ở hội thoại ngắn, trade-off là: tốn thêm một ít token để có khả năng nhớ dài hạn.

**Kết luận:** Compact memory **không phải lúc nào cũng thắng**. Ở hội thoại ngắn, chi phí `User.md` chưa được bù đắp bởi tiết kiệm compact, nên Advanced có thể tốn hơn Baseline về prompt tokens.

### 2.2. Vì sao compact thắng lớn ở hội thoại dài

Ở Long-Context Stress Benchmark, Advanced chỉ xử lý **15,841 prompt tokens** so với **24,593** của Baseline — **giảm 35.6%**.

Lý do:

- **Baseline** phải kéo theo **toàn bộ** lịch sử thread ở mỗi lượt. Prompt cost tăng **tuyến tính** theo số lượt → ở hội thoại dài, prompt phình rất nhanh.
- **Advanced** thay thế lịch sử cũ bằng một summary ngắn, chỉ giữ lại `keep_messages` gần nhất. Prompt cost bị **chặn trên** (bounded), không tăng vô hạn theo số lượt.

Cột `Compactions = 1` xác nhận compact memory **thực sự kích hoạt** trong stress test — không chỉ là code có mà không chạy.

**Điểm mấu chốt:** Compact memory chủ yếu tối ưu **`prompt tokens processed`**, không phải `agent tokens only`. Vì nó cắt ngữ cảnh **đầu vào**, không cắt câu trả lời **đầu ra**. Đây là lý do hai chỉ số này cần được đọc riêng.

### 2.3. Ba lớp memory và vai trò của chúng

| Lớp            | Công cụ                         | Lưu gì                                          | Vòng đời                                     | Agent có            |
| -------------- | ------------------------------- | ----------------------------------------------- | -------------------------------------------- | ------------------- |
| **Short-term** | `CompactMemoryManager.messages` | message gần nhất                                | theo `thread_id`                             | Baseline + Advanced |
| **Persistent** | `User.md` (`UserProfileStore`)  | fact ổn định: tên, nơi ở, nghề, style, sở thích | theo `user_id`, bền vững                     | Chỉ Advanced        |
| **Compact**    | `CompactMemoryManager.summary`  | tóm tắt message cũ                              | theo `thread_id`, tự trigger khi vượt ngưỡng | Chỉ Advanced        |

- **Cross-session recall** đến từ persistent memory (`User.md`), không phải từ short-term hay compact.
- **Tiết kiệm prompt tokens** đến từ compact memory, không phải từ persistent memory.
- Hai lớp này độc lập — Advanced cần cả hai để vừa nhớ đúng vừa kiểm soát chi phí.

### 2.4. Rủi ro và guardrail

**Rủi ro 1 — Memory file phình to**

Nếu `upsert_fact()` chỉ **thêm** fact mới mà không ghi đè, `User.md` sẽ phình vô hạn theo số lượt.

Giải pháp: `upsert_fact()` ghi đè theo **key**. Key `location` chỉ có một giá trị tại mọi thời điểm → file bounded.

**Rủi ro 2 — Lưu sai fact (false positive)**

Câu hỏi như `"Hiện tại mình làm nghề gì?"` có thể bị regex nghề nghiệp bắt nhầm và ghi `"nghề gì"` vào `User.md`, đè lên fact thật `"backend engineer"`.

Guardrail đã thêm ba tầng:

1. `extract_profile_updates()` **bỏ qua** message kết thúc bằng `?` — câu hỏi thuần không mang fact mới.
2. `_looks_like_question()` chặn value chứa từ để hỏi (`"gì"`, `"nào"`, `"đâu"`, `"sao"`).
3. `upsert_fact()` từ chối ghi value giống câu hỏi — belt-and-suspenders phòng khi extract lọt.

**Rủi ro 3 — Correction không được xử lý**

Khi user đổi nơi ở từ Đà Nẵng sang Huế, hoặc đổi nghề, agent phải **giữ fact mới nhất**, không giữ đồng thời fact cũ và mới.

Giải pháp: `upsert_fact()` ghi đè theo key → correction được xử lý tự động. Đây là **conflict handling** cơ bản.

---

## 3. Bonus đã triển khai

### 3.1. Conflict handling

**Vấn đề giải quyết:** User đổi nơi ở / nghề nghiệp. Nếu agent chỉ thêm fact mới, recall sẽ trả về thông tin cũ sai.

**Cách hoạt động:** `upsert_fact()` dùng key làm định danh duy nhất. Key `location` luôn có đúng một dòng `- location: <value>` trong `User.md`.

**Cải thiện:** Cross-session recall luôn phản ánh **fact mới nhất**.

**Rủi ro:** Nếu user nói `"Mình thích cả Đà Nẵng và Huế"` thì một trong hai sẽ bị ghi đè. Đây là hạn chế của mô hình key-value phẳng. Guardrail: `_NOISE_LOCATIONS` chặn các địa điểm chỉ xuất hiện trong ngữ cảnh không ổn định (ví dụ `"Hà Nội"` chỉ là nơi đi họp).

### 3.2. Confidence guardrail cho `extract_profile_updates`

**Vấn đề giải quyết:** Câu hỏi và câu khẳng định dùng chung từ vựng. Regex dễ bắt nhầm câu hỏi thành fact.

**Cách hoạt động:**

- Message kết thúc bằng `?` → không extract.
- Value chứa từ để hỏi → từ chối.

**Cải thiện:** Giảm false positive trong `User.md`, từ đó tăng recall chính xác.

**Rủi ro:** Có thể bỏ sót fact hợp lệ nếu user đặt câu hỏi tu từ có chứa fact (hiếm). Chấp nhận được vì ưu tiên precision hơn recall cho persistent memory.

---

## 4. Kết luận

Hệ thống memory ba lớp cho phép Advanced Agent:

1. **Nhớ xuyên session** nhờ `User.md` — recall tăng từ 0.036 lên 0.214 (Standard) và từ 0 lên 0.333 (Stress).
2. **Kiểm soát chi phí token** ở hội thoại dài nhờ compact memory — prompt tokens giảm 35.6% trong stress test.
3. **Không đánh đổi sai** ở hội thoại ngắn — chỉ tốn thêm ~2% prompt tokens, chấp nhận được so với lợi ích recall.

Hệ thống mạnh hơn Baseline nhưng cũng phức tạp hơn, và cần guardrail để tránh lưu sai fact. Đây là trade-off cốt lõi của memory system trong production.
