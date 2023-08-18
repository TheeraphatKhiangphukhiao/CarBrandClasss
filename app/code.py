import pickle
import json

# ทำการเปิดไฟล์ hogtest.json เพื่อนำข้อมูลค่า Hog หรือเอกลักษณ์ของรูปภาพเข้ามาเพื่อทำการ predict ว่าเป็นยี่ห้อรถอะไร
# with open(r"hogtest.json", "r") as json_file:
#     data = json.load(json_file) # ทำการโหลดข้อมูล json มาเก็บไว้ใน data

# hots = data['Hog'] # นำข้อมูล json ที่ key Hog มาเก็บไว้ใน hots
# m = pickle.load(open(r'..\model\ClassifierCarModel.pkl', 'rb')) # ทำการโหลด model ที่ได้ทำการเรียนรู้ไว้มาใช้งาน

def predictcar(m,HOG):
    result = m.predict(HOG) # ทำการ predict เพื่อทำนายยี่ห้อรถยนต์
    return result[0]

#print(predictcar([hots])) # เรียกใช้ฟังก์ชัน predictcar เพื่อทำการ predict โดยส่ง model=m เเละข้อมูลค่า Hog=hots