# Intelligent Insurance Claims Triage System

Xử lý dữ liệu cho bài toán binary classification trên bộ dữ liệu BNP Paribas Cardif Claims Management.
## Dữ liệu

```text
Train: 114,321 × 133
Test:  114,393 × 132
ID: ID
Target: target
```

`ID` được lưu riêng và không dùng làm feature. Từ 131 predictors ban đầu, expert preprocessing giữ lại 25 features gồm 12 numerical và 13 categorical.

## Xử lý features

- **Numerical:** giữ nguyên giá trị và NaN, không standardize hoặc impute.
- **Categorical:** missing được biểu diễn bằng `__MISSING__`; category chưa gặp trong train được mã hóa thành `-1`.
- **Categorical 1-way:** encode 13 categorical features thành `int32` codes.
- **Categorical 2-way:** tạo mọi cặp, ví dụ `cat2__v22__v24`.
- **Categorical 3-way:** tạo các tổ hợp ba chiều bắt buộc chứa `v22`.
- **Categorical 11-way:** tạo các tổ hợp 11 chiều bắt buộc chứa `v22`, có giới hạn số combination để kiểm soát bộ nhớ.
- **Numerical-to-categorical:** giảm hai chữ số thập phân cuối rồi encode, ví dụ `rnum__v10`.
- **Numerical pair sums:** tạo tổng từng cặp, ví dụ `num2sum__v10__v12`.

Category mappings chỉ được fit trên train; test và dữ liệu mới chỉ được transform. Target encoding OOF đã được triển khai nhưng mặc định tắt.

## Feature matrix cuối cùng

| Feature group | Số lượng |
|---|---:|
| Original numerical | 12 |
| Numerical pair sums | 66 |
| Categorical 1-way | 13 |
| Categorical 2-way | 78 |
| Categorical 3-way | 66 |
| Categorical 11-way | 66 |
| Numerical-to-categorical | 12 |
| **Tổng** | **313** |

```text
X_train_expert: 114,321 × 313
X_test_expert:  114,393 × 313
```

Dữ liệu model-ready được lưu trong `data/processed/`; preprocessing state được lưu tại `models/artifacts/bnp_feature_pipeline.joblib`.
