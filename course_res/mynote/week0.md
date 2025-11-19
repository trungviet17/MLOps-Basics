# Ghi chú Week 0: Lightning 

## Tổng quan 
**Lightning** : thư viện bọc lại pytorch sao cho phù hợp hơn với môi trường product 

Các thành phần chính của Lightning khi xây dựng một quy trình training bao gồm: 

1. Dataset : có thể cung cấp bởi thư viện bên ngoài / tự custom 
2. DataModule : đóng vai trò tương tự như với DataLoader trong torch => 