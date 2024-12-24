albedo:
    - hoang: 0.9580, 0.9575
    - lâm: 0.9485 0.9502
normalmap:
    - hoàng: 0.9386, 0.9394
    - lâm: 0.8976, 0.9177
deopthmap:
    - hoàng: 0.8733, 0.8825

concat:
    - hoàng: 0.9737, 0.9747
    - lâm: 0.8999, 0.9271

concat-unfreeze:
    - epoch cuối:
        - cosine: 0,9493 (210)
        - euclidean: 0,9607 (357)
    - best: 
        - cosine: 0,9659 (210)
        - euclidean: 0,9686 (327)
    - Giả định: mỗi mạng con được tối ưu để học 1 embedding chứa thông tin xác định identity của khuôn mặt với từng chiều khuôn mặt. Việc unfreeze lại làm cho trọng số mạng này bị thay đổi theo hướng.
        - Tích cực (mạng concat cải thiện): Việc unfreeze sẽ giúp,
            - Các mạng backbone bổ sung thông tin và khả năng nhận diện thông tin khuôn mặt tốt hơn, qua đó gián tiếp làm backbone cải thiện hơn, embedding chất lượng hơn.
            - Mạng concat tìm ra tổ hợp embedding tối ưu giữa 3 mạng
        - Tiêu cực (mạng concat tệ hơn): Việc unfreeze làm cho
            - Backbone đang được tối ưu để học embedding sẽ bị gây rối bởi thông tin từ các mạng backbone khác gây suy giảm chất lượng embedding mà backbone này tạo ra. Các backbone ảnh hưởng qua lại lẫn nhau làm cho embedding sinh ra của toàn mạng backbone kém đi => concat tệ đi
        - Thục nghiệm chứng minh concat khi unfreeze tệ hơn => Phải freeze lại.