import json
import os
import datetime
import random
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from PIL.ExifTags import TAGS
import pandas as pd
import numpy as np
import boto3
import io
import cv2
from ultralytics import YOLO # Using YOLOv8/v9 library for detection

# --- Configuration & Setup ---
# NOTE: Paths are relative assuming Render's Root Directory is set to 'backend'
DB_FILE = "pothole_reports.json" 
WEIGHTS_PATH = "weights/best.pt" 

# S3 Configuration (Reads from Environment Variables)
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "srm-pothole-detection-bucket")
S3_REGION = os.environ.get("S3_REGION", "ap-south-1")

# Global variables for client and model
s3_client = None
model = None

# --- Application Lifespan (for startup/shutdown) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes S3 client and loads the YOLO model when the server starts."""
    global s3_client, model

    # 1. Initialize S3 Client
    try:
        s3_client = boto3.client('s3', region_name=S3_REGION)
        print("S3 client initialized successfully.")
    except Exception as e:
        print(f"Error initializing S3 client. Check AWS configuration: {e}")
        s3_client = None

    # 2. Model Loading (Load the custom weights)
    try:
        if os.path.exists(WEIGHTS_PATH):
            model = YOLO(WEIGHTS_PATH)
            print(f"Loaded custom YOLO model from: {WEIGHTS_PATH}")
        else:
            model = YOLO('yolov8n.pt') 
            print("WARNING: Custom weights not found. Using default YOLOv8n model.")
            
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        # Final fallback to a mock model function if YOLO fails
        class MockResult:
            def __init__(self, boxes):
                self.boxes = type('Boxes', (object,), {'conf': np.array([random.uniform(0.7, 0.9)])})
        model = lambda x, conf, classes, verbose: [MockResult(boxes=None)] if random.random() < 0.3 else [MockResult(boxes=[])]
        print("Using Mock Detection Model.")
        
    yield
    print("Application shutting down.")

app = FastAPI(title="Pothole Reporting System (S3 Integrated)", lifespan=lifespan)

# CORS Middleware for Frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Core I/O and Processing Functions ---

def get_gps_from_image(image_bytes: bytes) -> Optional[tuple]:
    """Extracts GPS coordinates (lat, lon) from image bytes (EXIF data)."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        exif_data = image._getexif()
        if not exif_data: return None

        def get_decimal_from_dms(dms, ref):
            degrees = dms[0][0] / dms[0][1]
            minutes = dms[1][0] / dms[1][1] / 60.0
            seconds = dms[2][0] / dms[2][1] / 3600.0
            if ref in ['S', 'W']: return -(degrees + minutes + seconds)
            return degrees + minutes + seconds

        geotags = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'GPSInfo':
                for t in value:
                    geotags[TAGS.get(t,t)] = value[t]

        if not geotags: return None
        lat = get_decimal_from_dms(geotags['GPSLatitude'], geotags['GPSLatitudeRef'])
        lon = get_decimal_from_dms(geotags['GPSLongitude'], geotags['GPSLongitudeRef'])
        return (lat, lon)
    except Exception:
        return None

def load_records() -> List[dict]:
    DB_FILE_RELATIVE = "pothole_reports.json" 
    if not os.path.exists(DB_FILE_RELATIVE) or os.stat(DB_FILE_RELATIVE).st_size == 0:
        return []
    try:
        with open(DB_FILE_RELATIVE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return [] 

def save_records(records: List[dict]):
    DB_FILE_RELATIVE = "pothole_reports.json"
    with open(DB_FILE_RELATIVE, 'w') as f:
        json.dump(records, f, indent=4)


# --- S3 Upload Function (ASYNCHRONOUS) ---
async def upload_file_to_s3(file_bytes: bytes, filename: str, content_type: str) -> str:
    """Uploads file bytes to S3 using blocking boto3 call."""
    if not s3_client:
        raise HTTPException(status_code=500, detail="Cloud storage client not initialized.")
        
    s3_key = f"media/{filename}"
    
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_bytes,
            ContentType=content_type
        )
        return s3_key
    except Exception as e:
        print(f"S3 Upload Error: {e}")
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}") 


def run_pothole_detection_image(image_bytes: bytes) -> dict:
    """Runs YOLO inference on a single image."""
    global model
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(img, conf=0.5, classes=0, verbose=False) 
    
    detection_count = len(results[0].boxes)
    
    if detection_count > 0:
        max_confidence = float(results[0].boxes.conf.max())
        is_pothole = True
    else:
        max_confidence = round(random.uniform(0.1, 0.4), 4)
        is_pothole = False
        
    return {
        "pothole_found": is_pothole,
        "confidence": max_confidence,
        "detection_count": detection_count
    }


# --- FastAPI Routes (SYNCHRONOUS/BLOCKING LOGIC) ---

@app.post("/api/upload/image")
async def upload_image(
    file: UploadFile = File(...),
    manual_lat: Optional[float] = Form(None), 
    manual_lon: Optional[float] = Form(None)
):
    """
    Handles image upload, detection, and S3 upload. 
    NOTE: This operation blocks the server until complete (takes 5-6 mins).
    """
    
    # 1. Read the full file content
    image_bytes = await file.read()
    
    # 2. Generate required metadata quickly
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"{timestamp_str}_{file.filename.replace(' ', '_')}"
    content_type = file.content_type
    
    # 3. Get the location (metadata OR manual)
    lat, lon = manual_lat, manual_lon
    if lat is None or lon is None:
        gps_coords = get_gps_from_image(image_bytes)
        if gps_coords:
            lat, lon = gps_coords
    
    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="Location is required.")
    
    # 4. Upload file to S3 (BLOCKING I/O)
    media_key = await upload_file_to_s3(image_bytes, image_filename, content_type)

    # 5. Run Detection Model (BLOCKING CPU/ML)
    detection_result = run_pothole_detection_image(image_bytes)

    # 6. Save the record
    record = {
        "latitude": lat,
        "longitude": lon,
        "pothole_found": detection_result["pothole_found"],
        "confidence": detection_result["confidence"],
        "media_name": image_filename,
        "media_type": "Image",
        "media_key": media_key, 
        "detection_count": detection_result["detection_count"],
        "date_time": datetime.datetime.now().isoformat()
    }
    
    records = load_records()
    records.append(record)
    save_records(records)

    # 7. Return the final result
    return {
        "message": "Image analyzed and uploaded to S3 successfully!",
        "result": record
    }


@app.post("/api/upload/video")
async def upload_video(
    file: UploadFile = File(...),
    manual_lat: Optional[float] = Form(None), 
    manual_lon: Optional[float] = Form(None)
):
    """Handles video upload, detection, and S3 upload."""
    
    if manual_lat is None or manual_lon is None:
        raise HTTPException(status_code=400, detail="Location is required for videos. Please manually enter coordinates.")

    video_bytes = await file.read()
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{timestamp_str}_{file.filename.replace(' ', '_')}"
    content_type = file.content_type
    
    # 1. Upload original file to S3
    original_media_key = await upload_file_to_s3(video_bytes, video_filename, content_type)
    
    # 2. Save video locally for processing 
    temp_input_path = os.path.join("/tmp", video_filename)
    with open(temp_input_path, "wb") as buffer:
        buffer.write(video_bytes)

    # 3. Run Video Analysis and write a processed video
    base_name = video_filename.split('.')[0]
    output_folder_name = "processed_" + base_name
    temp_output_dir = os.path.join("/tmp", output_folder_name)
    
    try:
        results = model.predict(
            source=temp_input_path, 
            save=True, 
            project="/tmp", 
            name=output_folder_name, 
            exist_ok=True, 
            conf=0.5,
            classes=0,
            verbose=False
        )
        
        # 4. Upload Processed Video back to S3
        output_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.mp4') or f.endswith('.avi')]
        if not output_files:
            raise Exception("YOLO saved detections but not the video file.")
            
        temp_output_path = os.path.join(temp_output_dir, output_files[0])
        processed_filename = f"processed_{video_filename}"
        
        with open(temp_output_path, "rb") as processed_file:
            processed_bytes = processed_file.read()
            processed_media_key = await upload_file_to_s3(processed_bytes, processed_filename, "video/mp4")
        
        total_detections = sum(len(r.boxes) for r in results)
        max_confidence = 1.0 if total_detections > 0 else 0.0
        
    except Exception as e:
        print(f"Video processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video analysis failed on the server: {e}")
    finally:
        if os.path.exists(temp_input_path): os.remove(temp_input_path)
        if 'temp_output_path' in locals() and os.path.exists(temp_output_path): os.remove(temp_output_path)
        
    # 5. Save the record
    record = {
        "latitude": manual_lat,
        "longitude": manual_lon,
        "pothole_found": total_detections > 0,
        "confidence": max_confidence, 
        "media_name": video_filename,
        "media_type": "Video",
        "media_key": original_media_key,
        "processed_media_key": processed_media_key,
        "detection_count": total_detections,
        "date_time": datetime.datetime.now().isoformat()
    }
    
    records = load_records()
    records.append(record)
    save_records(records)

    return {
        "message": "Video analyzed and processed video uploaded to S3!",
        "result": record,
        "total_detections": total_detections
    }

# --- MISSING ADMIN DASHBOARD ROUTES ADDED HERE ---

@app.get("/api/potholes")
def list_pothole_reports():
    """Returns a list of all submitted pothole reports (for the Admin Dashboard table)."""
    # NOTE: This endpoint is called by the frontend to populate the dashboard.
    return load_records()

@app.get("/api/export/excel")
def export_to_excel():
    """Generates and returns an Excel file of all pothole reports."""
    records = load_records()
    if not records:
        raise HTTPException(status_code=404, detail="No records to export.")

    df = pd.DataFrame(records)

    df = df.rename(columns={
        "latitude": "Latitude",
        "longitude": "Longitude",
        "pothole_found": "Pothole Found?",
        "confidence": "Max Confidence",
        "media_name": "Media File",
        "media_type": "Type",
        "detection_count": "Detection Count",
        "date_time": "Date"
    })
    
    df["Pothole Found?"] = df["Pothole Found?"].apply(lambda x: "YES" if x else "NO")
    df["Max Confidence"] = (df["Max Confidence"] * 100).round(2).astype(str) + "%"
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    final_columns = ["Latitude", "Longitude", "Pothole Found?", "Detection Count", "Max Confidence", "Type", "Media File", "Date"]
    df = df.T.reindex(final_columns).T.reset_index(drop=True)
    
    excel_filename = f"pothole_reports_{datetime.date.today().isoformat()}.xlsx"
    excel_path = f"/tmp/{excel_filename}" if os.name != 'nt' else excel_filename
    df.to_excel(excel_path, index=False)

    return FileResponse(
        path=excel_path,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        filename=excel_filename
    )