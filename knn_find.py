import numpy as np
from collections import Counter

def knn_predict(X_train, y_train, x_new, k):
    """
    ฟังก์ชันทำนายกลุ่มด้วย k-NN
    """
    # 1. คำนวณระยะห่าง (Euclidean Distance)
    distances = np.linalg.norm(X_train - x_new, axis=1)
    
    # 2. เรียงลำดับระยะทางจากน้อยไปมาก แล้วดึง Index ของ K ตัวแรก
    nearest_indices = np.argsort(distances)[:k]
    
    # 3. ดึงชื่อกลุ่มของเพื่อนบ้าน K ตัวนั้น
    k_nearest_labels = y_train[nearest_indices]
    
    # 4. นับคะแนนโหวต (Majority Vote)
    most_common = Counter(k_nearest_labels).most_common(1)
    
    return most_common[0][0]

if __name__ == "__main__":
    # ชุดข้อมูลตามโจทย์ในสไลด์ (9 จุด, 3 กลุ่มสี)[cite: 2]
    X_train = np.array([
        [4.0, 5.0], [3.0, 6.0], [2.0, 5.0],  # blue (จุดที่ 1, 2, 3)[cite: 2]
        [6.0, 6.0], [5.5, 6.2], [7.0, 7.0],  # green (จุดที่ 4, 5, 6)[cite: 2]
        [5.0, 3.2], [4.2, 3.5], [5.0, 2.0]   # red (จุดที่ 7, 8, 9)[cite: 2]
    ])
    
    y_train = np.array([
        "blue", "blue", "blue",
        "green", "green", "green",
        "red", "red", "red"
    ])
    
    # จุด X ตรงกลางตามโจทย์[cite: 2]
    X_target = np.array([5.0, 5.0])
    
    print(f"จุดพยากรณ์ X: {X_target.tolist()}\n" + "="*35)
    
    # ทดสอบค่า k ตามโจทย์ในสไลด์ (k=1, k=3, k=6)[cite: 2]
    k_list = [1, 3, 6]
    for k in k_list:
        result = knn_predict(X_train, y_train, X_target, k)
        print(f"เมื่อ k = {k} --> X อยู่ในกลุ่ม: '{result}'")