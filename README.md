# Automated Visual Quality Control (AVQC)

## About the Project
This project was built to solve a simple problem — checking product quality manually is slow and inconsistent. So I tried to automate it using computer vision.

AVQC takes an image of a product and tells whether it is defective or not. The idea is to reduce human effort and make inspection faster.

---

## What it does
- Takes image input from user  
- Processes it on backend  
- Uses a trained model to detect defects  
- Shows result (defective / non-defective)  

---

## Tech Used
- Frontend: HTML, CSS, JS  
- Backend: Flask (Python)  
- Libraries: OpenCV, NumPy,yolo,ultralytics,,flaskcors,sqlite

---

## Project Structure

AVQC/
- backend/
  - app.py
  - templates/
  - static/
  - model/
- frontend/
  - index.html
  - style.css
  - script.js
- dataset/
- requirements.txt
- README.md

---

## How to Run

Clone the repo:
git clone https://github.com/your-username/AVQC.git  
cd AVQC  

Install dependencies:
pip install -r requirements.txt  

Run:
python app.py  

Open browser:
http://127.0.0.1:5000/  

---

## How it works
1. User uploads image  
2. Flask backend receives it  
3. Model processes the image  
4. Output is generated  
5. Result shown on UI  

---

## Current Limitations
- Model accuracy is not perfect  
- Works on limited dataset  
- No real-time camera support yet  

---

## Future Plans
- Improve model accuracy  
- Add live camera detection  
- Better UI  
- Deploy online  

---

## Why I made this
I wanted to explore how computer vision can be used in real-world problems like manufacturing. This project is more of a learning + practical implementation.

---

## Author
Pratyaksh Gupta
