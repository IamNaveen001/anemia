#  Anemia Prediction from Eye Images

A Flask web application that predicts **anemia** using **machine learning** by analyzing eye images uploaded by users. The model is trained to detect signs of low hemoglobin levels based on visual cues in the eye region.

## 📷 How It Works

1. User uploads a clear image of their eye.
2. The image is preprocessed and extract RGB values from the conjunctiva region and fed into a trained ML model (e.g., RandomForest).
3. The model predicts whether the person is likely anemic or not.
4. The result is displayed instantly on the webpage.

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, Bootstrap
- **Image Handling:** OpenCV / Pillow




## 🚀 Getting Started

📂 Project Structure

anemia/
├── static/              # CSS and image assets
├── templates/           # HTML files
├── model/               # Trained model (.h5 or .pkl)
├── uploads/             # Folder to temporarily store uploaded images
├── app.py               # Main Flask application
├── requirements.txt     # Required Python packages
└── README.md            # Project overview


### 1. Clone the repository
```bash
git clone https://github.com/IamNaveen001/anemia.git
cd anemia
```
Install dependencies
```
pip install -r requirements.txt
```
Run the Flask app
```
python app.py

```
Visit in browser
```
http://127.0.0.1:5000
```

📸 Screenshots

![Screenshot 2025-04-21 102814](https://github.com/user-attachments/assets/6a1717d5-9801-4571-898d-91a8159e6517)



![Screenshot 2025-04-21 102943](https://github.com/user-attachments/assets/b29a824e-7794-4eaf-8ce3-01ed4bde5e28)



![Screenshot 2025-04-21 103041](https://github.com/user-attachments/assets/974db10c-c638-4e4c-8186-48bc3226c8c6)


