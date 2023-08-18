# libarry FastAPI ในการสร้าง server
from fastapi import FastAPI, Request

from fastapi.middleware.cors import CORSMiddleware
import pickle
# libarry สำหรับ Request Body
from pydantic import BaseModel
# libarry แปลง base64
import numpy as np
import requests
# เข้าถึง Folder app เรียกใช้ file code เพื่อใช้งาน Method predictModel
import os
from app.code import predictcar

# สร้าง object ของ FastAPI
app = FastAPI()

# เพิ่ม middleware เพื่อจัดการ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # กำหนดให้ทุกโดเมนสามารถเข้าถึงได้
    allow_methods=['*'],  # กำหนดให้สามารถใช้ทุก HTTP method
    allow_headers=['*']   # กำหนดให้สามารถใช้ทุก header
)

@app.get("/")
def read_root():
    return {"Class Brand Car"}


# 1. สร้าง api เพื่อรอรับ base64 ของภาพ
# 2. ส่ง base64 ที่ได้ ให้กับ 'http://localhost:8080/api/gethog/' จะได้รับค่า HOG มา
# 3. นำค่า HOG ที่ได้ มาเข้า Model เพื่อหาว่า ค่า hog นั้นคือ brand(ยี่ห้อของรถ)
# 4. ส่งกลับ Brand ของรถที่ได้จากการนำเข้า model
@app.post("/api/carbrand")
async def read_image(image: Request):

    # ส่ง base64 ที่ได้ ให้กับ 'http://localhost:8080/api/gethog/' เพื่อเอาค่า HOG กลับมา
    ########### IPAddress ของ Containers เมื่อต้องการให้ Containers สื่อสารกัน ############
    ########### คำสั่งดู IPAddress ของ Containers ที่ run อยู่ --> docker inspect <รหัสของ Containers นั้น เช่น 9095c139ada5764c7e1460c930ea9f3fb1e26d696061c1e97d7b4a230dde14aa>
    poth_gethog = 'http://172.17.0.2:80/api/genhog'
    m = pickle.load(open(os.getcwd()+r'/model/ClassifierCarModel.pkl', 'rb')) # ทำการโหลด model ที่ได้ทำการเรียนรู้ไว้มาใช้งาน
    data = await image.json() # สร้าง json

    # เรียกใช้ api โดยส่ง base64 ที่รับมานั้น ไปอีกทีเพื่อหาเอกลักษณ์ของรูปภาพค่า HOG
    hog = requests.post(poth_gethog, json=data)

    # ค่าที่ตอบ hog กลับมานั้นจะเป็นแบบ json ดังนั้นต้องแปลงให้เป็น list
    # ค่าที่ตอบกลับมา จะมี 2 ค่า คือ HOG Length และ HOG vector แต่เราต้องการแค่ HOG vector
    hog = hog.json()['vector']
    brand =  predictcar(m,[hog]) 

    return {'Brain is' : brand}