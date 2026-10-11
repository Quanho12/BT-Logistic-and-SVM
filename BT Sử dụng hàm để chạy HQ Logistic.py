import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Dữ liệu đầu vào (X cần chuyển về ma trận cột có kích thước N x d)
X = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]).reshape(-1, 1)

y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# 2. Khởi tạo mô hình
# C = 1e5 hoặc penalty=None để tắt/giảm thiểu regularization (chuẩn hóa),
# giúp kết quả tương đồng nhất với phương pháp tối ưu Maximum Likelihood thuần túy trong bài code gốc
model = LogisticRegression(penalty=None, solver='lbfgs')

# 3. Huấn luyện mô hình
model.fit(X, y)

# 4. Trích xuất tham số: w0 (bias/intercept) và w1 (hệ số góc)
w0 = model.intercept_[0]
w1 = model.coef_[0][0]

print("Hệ số bias (w0):", w0)
print("Hệ số đặc trưng (w1):", w1)
print(f"Vector trọng số tương ứng w: [[{w0:.4f}], [{w1:.4f}]]")

# 5. Ví dụ dự đoán xác suất và nhãn cho dữ liệu mới (ví dụ: học 1.5h và 4.0h)
X_test = np.array([[1.5], [4.0]])
prob = model.predict_proba(X_test) # Trả về [P(y=0), P(y=1)]
pred = model.predict(X_test)       # Trả về nhãn 0 hoặc 1

for hours, p, label in zip(X_test.flatten(), prob[:, 1], pred):
    print(f"Học {hours} giờ -> Xác suất đỗ: {p*100:.2f}% -> Kết quả dự đoán: {label}")