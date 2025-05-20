import time
import click
import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
from collections import deque
import os
from datetime import datetime

model = YOLO("yolov8n.pt")
class_names = model.names

allowed_classes = ['person', 'cell phone', 'remote', 'book', 'mouse', 'laptop', 'earphones']
allowed_ids = [k for k, v in class_names.items() if v in allowed_classes]
timeout_seconds = 10

warning_limit = 50
warning_count = 0
screenshot_dir = "screenshots"
os.makedirs(screenshot_dir, exist_ok=True)

log_file = "warning_log.txt"

@click.group()
def cli():
    pass

@click.command()
def start():
    global warning_count
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        click.echo("Error: Could not open webcam.")
        return
    
    last_person_detection_time = time.time()
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    YAW_THRESH = 25    
    PITCH_THRESH = 25  
    AWAY_LIMIT = 2 

    calib_buffer_y = []
    calib_buffer_p = []
    calibrated = False
    baseline_yaw = 0.0
    baseline_pitch = 0.0

    prev_frame_time = 0
    new_frame_time = 0

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                click.echo("Error: Could not read frame.")
                break

            new_frame_time = time.time()
            fps = int(1 / (new_frame_time - prev_frame_time + 1e-5))
            prev_frame_time = new_frame_time

            results = model(frame, verbose=False)[0]
            conf_mask = results.boxes.conf > 0.5
            cls_mask = np.isin(results.boxes.cls.cpu().numpy().astype(int), allowed_ids)
            keep_mask = np.logical_and(conf_mask.cpu().numpy(), cls_mask)
            filtered_boxes = results.boxes[keep_mask]
            results.boxes = filtered_boxes

            current_classes = [int(cls) for cls in filtered_boxes.cls]
            filtered_classes = [class_names[c] for c in current_classes]

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_face = face_detection.process(frame_rgb)

            alerts = []
            triggered_warning = False

            person_count = filtered_classes.count("person")
            has_cell = "cell phone" in filtered_classes
            has_book = "book" in filtered_classes
            has_remote_mouse = "remote" in filtered_classes or "mouse" in filtered_classes
            has_laptop = "laptop" in filtered_classes
            has_earphones = "earphones" in filtered_classes

            if person_count == 0:
                elapsed = time.time() - last_person_detection_time
                alerts.append(f"No person detected terminating in {(timeout_seconds - elapsed):.1f}")
                if elapsed > timeout_seconds:
                    alerts.append("No person detected for too long.")
                    triggered_warning = True
            else:
                last_person_detection_time = time.time()

            if person_count > 1:
                alerts.append("Multiple persons detected!")
                triggered_warning = True
            if has_cell:
                alerts.append("Mobile phone detected!")
                triggered_warning = True
            if has_book:
                alerts.append("Book detected!")
                triggered_warning = True
            if has_remote_mouse:
                alerts.append("Remote or mouse detected!")
                triggered_warning = True
            if has_laptop:
                alerts.append("Laptop detected!")
                triggered_warning = True
            if has_earphones:
                alerts.append("Earphones detected!")
                triggered_warning = True

            if results_face.detections:
                for detection in results_face.detections:
                    mp_drawing.draw_detection(frame, detection)
                    k = detection.location_data.relative_keypoints
                    if len(k) < 5:
                        continue
                    h, w, _ = frame.shape
                    image_points = np.array([
                        [k[0].x * w, k[0].y * h],
                        [k[1].x * w, k[1].y * h],
                        [k[2].x * w, k[2].y * h],
                        [k[3].x * w, k[3].y * h],
                        [k[4].x * w, k[4].y * h]
                    ], dtype="double")
                    model_points = np.array([
                        [0.0, 0.0, 0.0],
                        [-30.0, -30.0, -30.0],
                        [30.0, -30.0, -30.0],
                        [0.0, 30.0, -30.0],
                        [20.0, 30.0, -30.0]
                    ])
                    focal_length = w
                    center = (w / 2, h / 2)
                    camera_matrix = np.array([
                        [focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]
                    ], dtype="double")
                    dist_coeffs = np.zeros((4, 1))
                    success, rvec, tvec = cv2.solvePnP(
                        model_points, image_points, camera_matrix, dist_coeffs,
                        flags=cv2.SOLVEPNP_EPNP
                    )
                    if success:
                        rmat, _ = cv2.Rodrigues(rvec)
                        proj = np.hstack((rmat, tvec))
                        angles = cv2.decomposeProjectionMatrix(proj)[6]
                        pitch, yaw, roll = [float(a) for a in angles]
                        if not calibrated:
                            calib_buffer_y.append(yaw)
                            calib_buffer_p.append(pitch)
                            if len(calib_buffer_y) >= 30:
                                baseline_yaw = np.median(calib_buffer_y)
                                baseline_pitch = np.median(calib_buffer_p)
                                calibrated = True
                            continue

                        adj_yaw = yaw - baseline_yaw
                        adj_pitch = pitch - baseline_pitch

                        if abs(adj_yaw) > YAW_THRESH or abs(adj_pitch) > PITCH_THRESH:
                            alerts.append("Looking away from screen!")
                            triggered_warning = True

            if triggered_warning:
                warning_count += 1
                ts_file = datetime.now().isoformat().replace(":", "-")
                screenshot_path = os.path.join(screenshot_dir, f"warning_{warning_count}_{ts_file}.jpg")

                # Draw warnings on the screenshot
                for i, alert in enumerate(alerts):
                    cv2.putText(frame, alert, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite(screenshot_path, frame)

                log_entry = f"[{ts_file}] Warning #{warning_count}: {', '.join(alerts)}\n"
                with open(log_file, "a") as f:
                    f.write(log_entry)
                if warning_count >= warning_limit:
                    click.echo(f"Warning limit reached ({warning_count}). Exiting.")
                    break

            frame = results.plot()
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            sz, _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            x, y = frame.shape[1] - sz[0] - 10, 30
            cv2.putText(frame, ts, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)

            yy = y + sz[1] + 10 if y < 20 + len(alerts) * 25 else 20
            for a in alerts:
                cv2.putText(frame, a, (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                yy += 25

            cv2.imshow("Exam Cheating Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 

    cap.release()
    cv2.destroyAllWindows()
    click.echo("Monitoring ended.")

    # Final Summary 
    click.echo(f"Final Warning Count: {warning_count}")
    if warning_count > 0:
        click.echo(f"Details saved to {log_file}")

@click.command()
def stop():
    click.echo("Stop command triggered.")

@click.command()
def log():
    click.echo("Log command placeholder.")

cli.add_command(start)
cli.add_command(stop)
cli.add_command(log)

if __name__ == '__main__':
    cli()
