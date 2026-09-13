import os
from dotenv import load_dotenv
import sys
import uuid
import base64
import urllib.request
import numpy as np
import cv2
import datetime
import asyncpg
import asyncio
import gc
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import HTMLResponse

if int(cv2.__version__.split(".")[0]) >= 5:
    raise RuntimeError("OpenCV 5.0+ dropped Caffe model support. Downgrade your environment.")

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI()

load_dotenv()
postgres_url = os.getenv("DATABASE_URL")
DB_URL = os.getenv("DATABASE_URL", postgres_url)
db_pool = None
ai_task_queue = asyncio.Queue()
workers = []
ai_semaphore = asyncio.Semaphore(1)

FACE_PROTO, FACE_MODEL = "deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel"
GENDER_PROTO, GENDER_MODEL = "gender_deploy.prototxt", "gender_net.caffemodel"
AGE_PROTO, AGE_MODEL = "age_deploy.prototxt", "age_net.caffemodel"

MODEL_URLS = {
    FACE_PROTO: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    FACE_MODEL: "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    GENDER_PROTO: "https://raw.githubusercontent.com/Isfhan/age-gender-detection/master/gender_deploy.prototxt",
    GENDER_MODEL: "https://raw.githubusercontent.com/Isfhan/age-gender-detection/master/gender_net.caffemodel",
    AGE_PROTO: "https://raw.githubusercontent.com/Isfhan/age-gender-detection/master/age_deploy.prototxt",
    AGE_MODEL: "https://raw.githubusercontent.com/Isfhan/age-gender-detection/master/age_net.caffemodel"
}

for file_name, url in MODEL_URLS.items():
    if not os.path.exists(file_name):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(file_name, 'wb') as out_file:
            out_file.write(response.read())

face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
for net in [face_net, gender_net, age_net]:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

waiting_males, waiting_females = [], []
active_rooms, user_rooms, client_ips = {}, {}, {}
monitor_connections = set()
SCAM_WORDS = ["crypto", "invest", "cashapp", "venmo", "telegram", "whatsapp", "paypal", "bitcoin", "scam", "hack"]

def get_active_user_count():
    chat_clients = set(client_ips.keys())
    return max(len(monitor_connections), len(chat_clients))

async def broadcast_active_count():
    count = get_active_user_count()
    payload = {"type": "active_users", "count": count}
    
    for ws in list(monitor_connections):
        try:
            await ws.send_json(payload)
        except Exception:
            pass
            
    for ws in list(client_ips.keys()):
        try:
            await ws.send_json(payload)
        except Exception:
            pass

def decode_base64_image(base64_str: str):
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        np_arr = np.frombuffer(base64.b64decode(base64_str), np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def compress_image_b64(img):
    if img is None: return None
    try:
        small = cv2.resize(img, (160, 120))
        _, buffer = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception: return None

def analyze_frame(img, user_gender):
    if img is None: return False, False, False, "unknown"
    try:
        global face_net, gender_net, age_net
        h, w = img.shape[:2]
        
        blob_face = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        face_net.setInput(blob_face)
        detections = face_net.forward()
        
        face_found = False
        best_box = None
        max_conf = 0
        
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.60 and conf > max_conf: 
                max_conf = conf
                best_box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
                face_found = True
        
        is_kid = False
        predicted_gender = "unknown"
        is_nudity = False
        x1 = y1 = x2 = y2 = 0
        
        if face_found:
            startX, startY, endX, endY = best_box
            pad_x = int((endX - startX) * 0.15)
            pad_y = int((endY - startY) * 0.20)
            x1, y1 = max(0, startX - pad_x), max(0, startY - pad_y)
            x2, y2 = min(w, endX + pad_x), min(h, endY + pad_y)
            
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size > 0:
                blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), (78.4, 87.8, 114.9), swapRB=False)
                
                age_net.setInput(blob)
                age_preds = age_net.forward()[0]
                minor_prob = float(np.sum(age_preds[0:3]))
                is_kid = bool(minor_prob > 0.80)
                
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()[0]
                male_conf = float(gender_preds[0])
                female_conf = float(gender_preds[1])
                
                # Increased threshold to 85% to counter strong pink/colored room lighting bias
                if male_conf > 0.85:
                    predicted_gender = "male"
                elif female_conf > 0.85:
                    predicted_gender = "female"
                else:
                    predicted_gender = user_gender
            
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)

        lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv = np.array([25, 180, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

        combined_skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        
        if face_found:
            mask_h = y2 - y1
            neck_extension = int(y2 + (mask_h * 0.40))
            combined_skin_mask[max(0, y1-30):min(h, neck_extension), max(0, x1-20):min(w, x2+20)] = 0 
        
        torso_mask = combined_skin_mask[int(h*0.40):h, 0:w]
        if torso_mask.size > 0:
            skin_ratio = np.sum(torso_mask > 0) / torso_mask.size
            is_nudity = bool(skin_ratio > 0.62) 
        
        return face_found, is_kid, is_nudity, predicted_gender
    except Exception:
        return False, False, False, "unknown"

async def ai_background_worker():
    loop = asyncio.get_running_loop()
    while True:
        try:
            task = await ai_task_queue.get()
            client_ip, ws, user_gender, room_id, img = task
            
            if ws.client_state.name != "CONNECTED":
                ai_task_queue.task_done()
                continue

            async with ai_semaphore:
                face_found, is_kid, is_nudity, predicted_gender = await loop.run_in_executor(None, analyze_frame, img, user_gender)
            
            status_text = "No Face"
            if is_nudity:
                status_text = "Nudity Detected"
            elif face_found:
                status_text = predicted_gender.capitalize()
                if is_kid:
                    status_text += " (Minor)"

            try: await ws.send_json({"type": "ai_status", "payload": f"AI: {status_text}"})
            except Exception: pass

            if is_nudity:
                img_b64 = compress_image_b64(img)
                await execute_ban(client_ip, ws, room_id, "Explicit/Nudity content detected.", duration_hours=12, img_data=img_b64)
            elif is_kid:
                img_b64 = compress_image_b64(img)
                await execute_ban(client_ip, ws, room_id, "Minors are strictly prohibited.", duration_hours=87600, img_data=img_b64)
            elif not face_found:
                try: await ws.send_json({"type": "warning", "payload": "⚠️ Warning: Face not visible! Please stay in the camera view."})
                except Exception: pass
            elif predicted_gender != user_gender and predicted_gender != "unknown":
                img_b64 = compress_image_b64(img)
                await execute_ban(client_ip, ws, room_id, f"Gender Mismatch (Selected: {user_gender}, Detected: {predicted_gender})", duration_hours=12, img_data=img_b64)

            del img
            gc.collect()
            ai_task_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception:
            ai_task_queue.task_done()

async def execute_ban(client_ip, ws, room_id, reason, duration_hours=12, img_data=None):
    await ban_user(client_ip, reason, duration_hours, img_data)
    try:
        await asyncio.sleep(0.2)
        await ws.close()
    except Exception: pass

    for chat_ws, ip in list(client_ips.items()):
        if ip == client_ip:
            try:
                r_id = user_rooms.get(chat_ws)
                if r_id and r_id in active_rooms:
                    for client in active_rooms[r_id]:
                        if client != chat_ws:
                            await client.send_json({"type": "system", "payload": "Stranger was banned for safety violations."})
                            await client.send_json({"type": "peer_disconnected"})
                            await asyncio.sleep(0.2)
                            await client.close()
                    active_rooms.pop(r_id, None)
                await asyncio.sleep(0.2)
                await chat_ws.close()
            except Exception: pass

@app.on_event("startup")
async def startup():
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(DB_URL, statement_cache_size=0, max_inactive_connection_lifetime=300)
        async with db_pool.acquire() as conn:
            await conn.execute('CREATE TABLE IF NOT EXISTS banned_ips (ip VARCHAR(255) PRIMARY KEY, reason TEXT, is_banned BOOLEAN DEFAULT TRUE, expires_at TIMESTAMP, image_data TEXT)')
            try: await conn.execute('ALTER TABLE banned_ips ADD COLUMN expires_at TIMESTAMP')
            except Exception: pass
            try: await conn.execute('ALTER TABLE banned_ips ADD COLUMN image_data TEXT')
            except Exception: pass
            await conn.execute('CREATE TABLE IF NOT EXISTS ads (id SERIAL PRIMARY KEY, ad_content TEXT, is_active BOOLEAN DEFAULT TRUE)')
    except Exception: pass
    for _ in range(4): workers.append(asyncio.create_task(ai_background_worker()))

@app.on_event("shutdown")
async def shutdown():
    if db_pool: await db_pool.close()
    for worker in workers: worker.cancel()

async def is_banned(ip: str):
    if not db_pool: return None
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT reason FROM banned_ips WHERE ip = $1 AND is_banned = TRUE AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)", ip)
            return row["reason"] if row else None
    except Exception as e:
        print(f"DB Ban Check Error: {e}")
        return None

async def ban_user(ip: str, reason: str, duration_hours: int = 12, img_data: str = None):
    if not db_pool: return
    expires = datetime.datetime.utcnow() + datetime.timedelta(hours=duration_hours)
    try:
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO banned_ips (ip, reason, is_banned, expires_at, image_data) 
                VALUES ($1, $2, FALSE, $3, $4) 
                ON CONFLICT (ip) DO UPDATE SET is_banned = FALSE, reason = EXCLUDED.reason, expires_at = EXCLUDED.expires_at, image_data = EXCLUDED.image_data
            """, ip, reason, expires, img_data)
    except Exception as e: 
        print(f"DB Ban Insert Error: {e}")

@app.get("/")
async def serve_frontend():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f: return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Error: index.html not found!</h1>", status_code=404)

@app.get("/api/ad")
async def get_ad():
    return {"ad_content": "<div style='color:#fff;'>[ Default Ad Banner ]</div>"}

@app.get("/api/stats")
async def get_online_stats():
    return {"active_users": get_active_user_count()}

@app.get("/admin/bans")
async def view_banned_images():
    if not db_pool:
        return HTMLResponse("Database not connected.")
    
    html_content = "<h2>Moderation Review Queue</h2><table border='1' style='text-align:left; color:white; background:#1e1e1e; border-collapse:collapse; width:100%;'><tr><th style='padding:8px;'>IP</th><th style='padding:8px;'>Reason</th><th style='padding:8px;'>Expires At</th><th style='padding:8px;'>Evidence Snapshot</th></tr>"
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT ip, reason, expires_at, image_data FROM banned_ips WHERE image_data IS NOT NULL ORDER BY expires_at DESC")
        for row in rows:
            img_tag = f"<img src='data:image/jpeg;base64,{row['image_data']}' width='160' style='border-radius:4px;' />" if row['image_data'] else "No Image"
            html_content += f"<tr><td style='padding:8px;'>{row['ip']}</td><td style='padding:8px;'>{row['reason']}</td><td style='padding:8px;'>{row['expires_at']}</td><td style='padding:8px;'>{img_tag}</td></tr>"
            
    html_content += "</table>"
    return HTMLResponse(content=f"<body style='background:#121212; font-family:sans-serif; padding:20px;'>{html_content}</body>")

@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    client_ip = websocket.client.host
    try:
        await websocket.accept()
        monitor_connections.add(websocket)
        await broadcast_active_count()

        if await is_banned(client_ip):
            await websocket.close()
            return
        while True:
            data = await websocket.receive_json()
            img = decode_base64_image(data.get("image", ""))
            user_gender = data.get("gender", "male").lower()
            current_room = next((user_rooms.get(ws) for ws, ip in client_ips.items() if ip == client_ip), None)
            await ai_task_queue.put((client_ip, websocket, user_gender, current_room, img))
    except Exception:
        pass
    finally:
        monitor_connections.discard(websocket)
        await broadcast_active_count()

@app.websocket("/ws/chat/{gender}")
async def websocket_chat(websocket: WebSocket, gender: str):
    client_ip = websocket.client.host
    loop = asyncio.get_running_loop()
    try:
        await websocket.accept()
        ban_reason = await is_banned(client_ip)
        if ban_reason:
            await websocket.send_json({"type": "error", "payload": f"⛔ Banned: {ban_reason}"})
            await websocket.close()
            return

        client_ips[websocket] = client_ip
        await broadcast_active_count()
        user_gender = gender.lower()
        is_verified = False

        init_data = await websocket.receive_json()
        if init_data.get("type") == "verify":
            if not init_data.get("policy_accepted"):
                await websocket.send_json({"type": "error", "payload": "❌ Must accept policies."})
                await websocket.close()
                return
            
            img = decode_base64_image(init_data.get("image", ""))
            
            async with ai_semaphore:
                face_found, is_kid, is_nudity, predicted_gender = await loop.run_in_executor(None, analyze_frame, img, user_gender)
                
            if not face_found:
                await websocket.send_json({"type": "error", "payload": "❌ No human face detected."})
                await websocket.close()
                return

            if is_kid:
                await ban_user(client_ip, "Minors are strictly prohibited.", duration_hours=87600, img_data=None)
                await websocket.close()
                return

            if is_nudity:
                img_b64 = compress_image_b64(img)
                await ban_user(client_ip, "Explicit/Nudity content detected.", duration_hours=12, img_data=img_b64)
                await websocket.close()
                return

            if predicted_gender != user_gender and predicted_gender != "unknown":
                img_b64 = compress_image_b64(img)
                await ban_user(client_ip, f"Gender Mismatch (Selected: {user_gender}, Detected: {predicted_gender})", duration_hours=12, img_data=img_b64)
                await websocket.close()
                return

            del img
            gc.collect()

            is_verified = True
            await websocket.send_json({"type": "status", "payload": "✅ Verified. Joining matchmaking..."})

        if not is_verified: return

        if user_gender == "male":
            if waiting_females:
                partner_ws = waiting_females.pop(0)
                room_id = str(uuid.uuid4())
                active_rooms[room_id] = [websocket, partner_ws]
                user_rooms[websocket] = user_rooms[partner_ws] = room_id
                await websocket.send_json({"type": "match_start", "role": "initiator", "partner_gender": "Female"})
                await partner_ws.send_json({"type": "match_start", "role": "receiver", "partner_gender": "Male"})
            else:
                waiting_males.append(websocket)
                await websocket.send_json({"type": "status", "payload": "Searching for a user..."})
        else:
            if waiting_males:
                partner_ws = waiting_males.pop(0)
                room_id = str(uuid.uuid4())
                active_rooms[room_id] = [partner_ws, websocket]
                user_rooms[websocket] = user_rooms[partner_ws] = room_id
                await partner_ws.send_json({"type": "match_start", "role": "initiator", "partner_gender": "Female"})
                await websocket.send_json({"type": "match_start", "role": "receiver", "partner_gender": "Male"})
            else:
                waiting_females.append(websocket)
                await websocket.send_json({"type": "status", "payload": "Searching for a male user..."})

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            current_room = user_rooms.get(websocket)
            if not current_room or current_room not in active_rooms: continue

            if msg_type in ["offer", "answer", "candidate"]:
                for client in active_rooms[current_room]:
                    if client != websocket: await client.send_json(data)
            elif msg_type == "text":
                text_payload = data.get("payload", "").lower()
                if any(word in text_payload for word in SCAM_WORDS):
                    await ban_user(client_ip, "Scam/Spam violations.", duration_hours=87600)
                    for client in active_rooms[current_room]:
                        if client != websocket:
                            await client.send_json({"type": "system", "payload": "Stranger banned for scamming."})
                            await client.send_json({"type": "peer_disconnected"})
                            await client.close()
                    await websocket.close()
                    break
                for client in active_rooms[current_room]:
                    if client != websocket: await client.send_json({"type": "message", "payload": data.get("payload")})
            elif msg_type == "report":
                for client in active_rooms[current_room]:
                    if client != websocket:
                        await ban_user(client_ips.get(client), "Reported by user.", duration_hours=12)
                        await websocket.send_json({"type": "system", "payload": "User banned."})
                        await client.close()
                await websocket.send_json({"type": "peer_disconnected"})
                break
    except Exception:
        pass
    finally:
        if websocket in waiting_males: waiting_males.remove(websocket)
        if websocket in waiting_females: waiting_females.remove(websocket)
        client_ips.pop(websocket, None)
        current_room = user_rooms.pop(websocket, None)
        if current_room and current_room in active_rooms:
            partners = active_rooms.pop(current_room)
            for client in partners:
                if client != websocket:
                    user_rooms.pop(client, None)
                    try:
                        await client.send_json({"type": "peer_disconnected"})
                        await client.close()
                    except Exception: pass
        await broadcast_active_count()