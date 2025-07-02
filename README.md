# Dự đoán Chi phí Bảo hiểm Y tế bằng Machine Learning

Đây là một dự án phân tích và xây dựng mô hình Machine Learning nhằm mục đích dự đoán chi phí bảo hiểm y tế dựa trên các thuộc tính cá nhân và lối sống.

Dự án này được thực hiện như một phần của bài kiểm tra giữa kỳ môn Học Máy, với các bước cải tiến để tối ưu hóa hiệu suất và đánh giá mô hình một cách toàn diện.

---

## 🎯 Mục tiêu dự án

-   **Phân tích khám phá (EDA):** Tìm hiểu các yếu tố có ảnh hưởng lớn nhất đến chi phí bảo hiểm.
-   **Xây dựng mô hình:** Xây dựng, so sánh và lựa chọn mô hình hồi quy có khả năng dự đoán tốt nhất.
-   **Tối ưu hóa:** Tinh chỉnh siêu tham số của mô hình tốt nhất để cải thiện hiệu suất.
-   **Diễn giải kết quả:** Đánh giá mô hình cuối cùng bằng các độ đo có ý nghĩa về mặt kinh doanh (RMSE, MAE tính bằng USD).

---

## 📂 Cấu trúc Project

```
├── data/
│ └── Insurance.csv # Bộ dữ liệu gốc
├── notebooks/
│ └── MidTerm_Test.ipynb # Notebook chứa toàn bộ quy trình
├── .gitignore 
├── LICENSE
└── README.md
```
---

## 📊 Bộ dữ liệu

Bộ dữ liệu `Insurance.csv` chứa 1338 mẫu với các thuộc tính sau:

-   `age`: Tuổi (năm).
-   `sex`: Giới tính (`male`, `female`).
-   `bmi`: Chỉ số khối cơ thể (kg/m²).
-   `children`: Số con cái phụ thuộc.
-   `smoker`: Tình trạng hút thuốc (`yes`, `no`).
-   `region`: Vùng sinh sống ở Mỹ (`northeast`, `northwest`, `southeast`, `southwest`).
-   **`charges` (Target)**: Chi phí bảo hiểm y tế hàng năm (USD).

---

## 🛠️ Quy trình thực hiện

1.  **Làm sạch dữ liệu**:
    -   Kiểm tra và xử lý các giá trị thiếu (không có trong bộ dữ liệu này).
    -   Phát hiện và xóa 1 dòng dữ liệu bị trùng lặp.

2.  **Phân tích khám phá dữ liệu (EDA)**:
    -   Phân phối của biến mục tiêu `charges` bị lệch phải, do đó đã áp dụng **biến đổi Log (`log1p`)** để đưa về dạng phân phối chuẩn hơn.
    -   Trực quan hóa mối quan hệ giữa các biến độc lập và biến mục tiêu, phát hiện ra `smoker` và `age` là hai yếu tố có ảnh hưởng mạnh nhất.

3.  **Tiền xử lý & Feature Engineering**:
    -   **Scaling:** Áp dụng `StandardScaler` cho các biến số (`age`, `bmi`, `children`).
    -   **Encoding:** Áp dụng `OneHotEncoder` cho các biến phân loại (`sex`, `smoker`, `region`).
    -   Toàn bộ quá trình được đóng gói trong một `Pipeline` để đảm bảo tính nhất quán và tránh rò rỉ dữ liệu.

4.  **So sánh các mô hình**:
    -   Sử dụng **5-Fold Cross-Validation** để đánh giá và so sánh hiệu suất của 5 mô hình hồi quy phổ biến.
    -   Mô hình `GradientBoostingRegressor` cho kết quả R² trung bình cao nhất và ổn định nhất.

5.  **Tinh chỉnh siêu tham số (Hyperparameter Tuning)**:
    -   Sử dụng `GridSearchCV` để tìm kiếm bộ tham số tối ưu cho mô hình `GradientBoostingRegressor`.

6.  **Đánh giá cuối cùng**:
    -   Huấn luyện mô hình đã được tối ưu trên toàn bộ tập huấn luyện.
    -   Đánh giá hiệu suất cuối cùng trên tập kiểm tra (dữ liệu chưa từng thấy).

---

## 📈 Kết quả

Mô hình **Gradient Boosting Regressor** sau khi được tinh chỉnh đã cho kết quả ấn tượng trên tập kiểm tra:

| Độ đo | Giá trị | Diễn giải |
| :--- | :--- | :--- |
| **R-squared (R²)** | 0.8067 | Mô hình giải thích được khoảng 80.67% sự biến thiên của chi phí bảo hiểm. |
| **RMSE (USD)** | $5,234.52 | Sai số trung bình theo căn bậc hai của mô hình là khoảng $5,234.52. |
| **MAE (USD)** | $2,495.66 | **Trung bình, mô hình dự đoán sai lệch khoảng $2,495.66 so với chi phí thực tế.** |

---

## 🚀 Cách sử dụng

Để chạy lại project này trên máy của bạn, hãy làm theo các bước sau:

1.  **Clone repository:**
    ```bash
    git clone https://github.com/YourUsername/Machine-Learning_Medical-Insurance-Prediction.git
    cd Machine-Learning_Medical-Insurance-Prediction
    ```
    *(Thay `YourUsername` bằng tên người dùng GitHub của bạn)*

2.  **Tạo môi trường ảo (khuyến nghị):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Windows: venv\Scripts\activate
    ```

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Chạy Jupyter Notebook:**
    ```bash
    jupyter notebook notebooks/MidTerm_Test.ipynb
    ```
