# 🛣️ AI-Powered Pothole Detection and Reporting System

### Project Status: **Active / Deployment Ready (v1.0.0)**

This system provides an end-to-end automated solution for detecting
potholes and reporting road damage. It integrates computer vision, cloud
storage, and a full-stack architecture to help authorities and users
identify and manage road issues efficiently.

## 🌟 Key Features

-   **Real-Time Pothole Detection** using a custom-trained **YOLOv8**
    model.
-   **Persistent Data Storage** with **AWS RDS (PostgreSQL)**.
-   **Secure Media Storage** using **AWS S3** for all images/videos.
-   **Admin Dashboard** to view, filter, and export reports.
-   **Geolocation Extraction** using EXIF metadata or manual user input.
-   **Fully Cloud Deployed** Backend (Render) + Frontend (Vercel).

## 🏗️ Architecture Overview

  ------------------------------------------------------------------------
  Component            Technology               Description
  -------------------- ------------------------ --------------------------
  ML Model             YOLOv8 (Python)          Detects potholes in
                                                user-uploaded media

  Backend              FastAPI, Uvicorn         Processes uploads, runs
                                                YOLO, interacts with DB &
                                                S3

  Database             AWS RDS (PostgreSQL)     Stores report metadata

  Storage              AWS S3                   Stores raw image/video
                                                files

  Frontend             HTML, JavaScript,        Upload UI + Admin
                       Tailwind CSS             Dashboard

  Deployment           Render (API), Vercel     Global CDN and reliable
                       (UI)                     hosting
  ------------------------------------------------------------------------

## 🛠️ Local Installation

### 1. Prerequisites

Install:

-   Python 3.10+
-   Git
-   AWS IAM User with S3 Access
-   AWS RDS PostgreSQL instance

### 2. Clone the Repository

``` bash
git clone https://github.com/sirajv127/pothole-detection-system.git
cd pothole-detection-system/backend
pip install -r requirements.txt
```

### 3. Create `.env` File

Inside `backend/.env`:

``` ini
# --- AWS S3 ---
AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
S3_BUCKET_NAME="AWS_BUCKET_NAME"
S3_REGION="ap-south-1"

# --- AWS RDS ---
DATABASE_URL="postgresql://pothole_admin:YourPass@database-1xxx.rds.amazonaws.com:5432/postgres"
```

Place YOLO weights here:

    backend/weights/best.pt

### 4. Run Backend Locally

``` bash
uvicorn main:app --reload
```

API available at:

    http://127.0.0.1:8000

## 🚀 Deployment Guide

### Backend Deployment (Render)

1.  Connect GitHub repository.
2.  Add all `.env` values in Environment Variables.
3.  Set Start Command:

``` bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

4.  Deploy.

### Frontend Deployment (Vercel)

1.  Upload the `frontend` folder to Vercel.
2.  Update API URL in `frontend/index.html`:

``` javascript
const API_BASE_URL = "https://your-render-backend.onrender.com";
```

3.  Deploy.

## ❓ Troubleshooting

  -----------------------------------------------------------------------
  Issue               Reason                 Solution
  ------------------- ---------------------- ----------------------------
  Dashboard shows     Wrong API URL          Update `API_BASE_URL`
  blank                                      

  RDS not connecting  Security groups        Open Port 5432
                      blocked                

  Images not          Wrong S3 keys/region   Check IAM permissions
  uploading                                  

  YOLO not loading    Incorrect weights path Use backend/weights/best.pt
  -----------------------------------------------------------------------

## 📁 Project Structure

    pothole-detection-system/
    │
    ├── backend/
    │   ├── main.py
    │   ├── database.py
    │   ├── models.py
    │   ├── detect.py
    │   ├── weights/
    │   │   └── best.pt
    │   ├── utils/
    │   └── .env
    │
    └── frontend/
        ├── index.html
        ├── admin.html
        ├── styles.css
        └── script.js

## 🤝 Contribution

Pull requests and issue reports are welcome.

-   **Author:** Your Name
-   **License:** MIT
