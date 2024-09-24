import os
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
import subprocess
from skimage.exposure import match_histograms
from skimage import exposure
from concurrent.futures import ThreadPoolExecutor, as_completed

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.1, min_tracking_confidence=0.1, refine_landmarks=True)

def get_landmarks(image):
    """
    Get the landmarks of the face in the image using mediapipe

    :param image: The image to get the landmarks from
    :return: The landmarks of the face in the image
    """
    results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    ih, iw = image.shape[:2]
    landmarks_array = np.array([(int(lm.x * iw), int(lm.y * ih)) for lm in landmarks])
    return landmarks_array

def get_transform_params(lm):
    """
    Calculates a quadrilateral around the face using key landmarks. Used for facial alignment
    to normalize position and orientation of the face in an image.

    :param lm: The landmarks of the face in the image
    :return: The quadrilateral around the face
    """
    left_eye = np.mean(lm[[33, 133]], axis=0)
    right_eye = np.mean(lm[[362, 263]], axis=0)
    eye_avg = (left_eye + right_eye) * 0.5
    eye_to_eye = right_eye - left_eye
    mouth_avg = (lm[61] + lm[291]) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.5, np.hypot(*eye_to_mouth) * 2)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    return np.stack([c - x - y, c - x + y, c + x + y, c + x - y])

def align_face(img, quad):
    """
    Aligns the face in the image using the quadrilateral

    :param img: The image to align the face in
    :param quad: The quadrilateral around the face
    :return: The aligned image and the transformation matrix
    """
    dst = np.array([(0, 0), (0, 511), (511, 511), (511, 0)], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (512, 512), flags=cv2.INTER_LINEAR), M

def unalign_face(aligned_img, M, original_shape):
    """
    Unaligns the face in the image using the transformation matrix

    :param aligned_img: The aligned image
    :param M: The transformation matrix
    :param original_shape: The shape of the original image
    :return: The unaligned image
    """
    h, w = original_shape[:2]
    return cv2.warpPerspective(aligned_img, np.linalg.inv(M), (w, h), flags=cv2.INTER_LINEAR)

def process_image(in_path, out_path):
    """
    Processes the image to align the face

    :param in_path: The path to the image to process
    :param out_path: The path to save the aligned image
    """
    img = cv2.imread(in_path)
    lm = get_landmarks(img)
    if lm is not None:
        quad = get_transform_params(lm)
        aligned_img, M = align_face(img, quad)
        cv2.imwrite(out_path, aligned_img)
        return M, img.shape
    return None, None
    
def process_video(in_path, out_path):
    """
    Processes the video to align the face

    :param in_path: The path to the video to process
    :param out_path: The path to save the aligned video
    """
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (512, 512))

    smooth_frames = 5
    transform_info = {'fps': fps, 'transforms': [], 'frame_count': total_frames}

    quad_history = deque(maxlen=smooth_frames)
    last_valid_quad = None
    no_face_count = 0
    face_areas = []
    landmark_buffer = deque(maxlen=5)

    initial_frames = []
    initial_landmarks = []
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        initial_frames.append(frame)
        lm = get_landmarks(frame)
        if lm is not None:
            initial_landmarks.append(lm)
            if len(initial_landmarks) >= 3:
                break

    if initial_landmarks:
        avg_landmarks = np.mean(initial_landmarks, axis=0)
        for i, frame in enumerate(initial_frames):
            quad = get_transform_params(avg_landmarks)
            quad_history.append(quad)
            last_valid_quad = np.mean(quad_history, axis=0)
            aligned_frame, M = align_face(frame, last_valid_quad)
            out.write(aligned_frame)
            transform_info['transforms'].append((M, frame.shape, i, avg_landmarks))
            face_hull = cv2.convexHull(avg_landmarks.astype(np.int32))
            face_area = cv2.contourArea(face_hull)
            face_areas.append(face_area / (width * height))

    landmark_distance_threshold = 75
    last_valid_landmarks = avg_landmarks if initial_landmarks else None

    frames_without_landmarks = 0

    for frame_num in range(len(initial_frames), total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        lm = get_landmarks(frame)
        if lm is not None:
            if last_valid_landmarks is not None:
                distance = np.mean(np.linalg.norm(lm - last_valid_landmarks, axis=1))
                if distance > landmark_distance_threshold:
                    quad = get_transform_params(lm)
                    last_valid_quad = quad
                    last_valid_landmarks = lm
                else:
                    quad = get_transform_params(lm)
                    last_valid_quad = 0.8 * last_valid_quad + 0.2 * quad if last_valid_quad is not None else quad
                    last_valid_landmarks = 0.8 * last_valid_landmarks + 0.2 * lm
            else:
                quad = get_transform_params(lm)
                last_valid_quad = quad
                last_valid_landmarks = lm
            
            quad_history.append(last_valid_quad)
            no_face_count = 0
            
            face_hull = cv2.convexHull(lm.astype(np.int32))
            face_area = cv2.contourArea(face_hull)
            face_areas.append(face_area / (width * height))
            
            landmark_buffer.append(lm)
        else:
            no_face_count += 1
            frames_without_landmarks += 1
            if no_face_count < 5 and landmark_buffer:
                lm = landmark_buffer[-1]
                quad = get_transform_params(lm)
                quad_history.append(quad)
                last_valid_quad = np.mean(quad_history, axis=0)
            else:
                lm = None

        if no_face_count < 15 and last_valid_quad is not None:
            aligned_frame, M = align_face(frame, last_valid_quad)
            out.write(aligned_frame)
            transform_info['transforms'].append((M, frame.shape, frame_num, last_valid_landmarks))
        else:
            aspect_ratio = width / height
            if aspect_ratio > 1:
                new_width, new_height = 512, int(512 / aspect_ratio)
            else:
                new_height, new_width = 512, int(512 * aspect_ratio)
            
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            canvas = np.zeros((512, 512, 3), dtype=np.uint8)
            
            pad_x, pad_y = (512 - new_width) // 2, (512 - new_height) // 2
            
            canvas[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized_frame
            
            out.write(canvas)
            transform_info['transforms'].append((None, frame.shape, frame_num, None))

    cap.release()
    out.release()

    if frames_without_landmarks / total_frames > 0.4:
        raise ValueError("More than 40% of frames don't have any faces.")

    if face_areas:
        avg_face_area = np.mean(face_areas)
        print(f"Average face area: {avg_face_area:.2%} of frame size")
        transform_info['avg_face_size'] = avg_face_area * 100
    else:
        print("No faces detected in the video")
        transform_info['avg_face_size'] = 0

    return transform_info

def align_image(input_path, output_path):
    return process_image(input_path, output_path) if os.path.exists(input_path) and input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) else (None, None)

def align_video(input_path, output_path):
    return process_video(input_path, output_path) if os.path.exists(input_path) and input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) else None

def align_media(input_path, output_path):
    if not os.path.exists(input_path):
        return None

    if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        return process_image(input_path, output_path)
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return process_video(input_path, output_path)
    else:
        return None

def unalign_media(aligned_path, source_path, transform_info, output_path):
    if not os.path.exists(aligned_path) or not os.path.exists(source_path):
        return

    if aligned_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        unalign_image(aligned_path, source_path, transform_info['transforms'][0], transform_info['transforms'][1], output_path)
    elif aligned_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        unalign_video(aligned_path, source_path, transform_info, output_path)

def unalign_image(aligned_path, source_path, M, original_shape, output_path):
    aligned_img = cv2.imread(aligned_path)
    source_img = cv2.imread(source_path)
    if M is not None:
        unaligned_img = unalign_face(aligned_img, M, original_shape)
        mask = np.all(unaligned_img != 0, axis=2).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        kernel = np.ones((20, 20), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        result = unaligned_img * mask[..., np.newaxis] + source_img * (1 - mask[..., np.newaxis])
        cv2.imwrite(output_path, result.astype(np.uint8))
    else:
        cv2.imwrite(output_path, source_img)

def create_face_mask(landmarks, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if landmarks is not None and len(landmarks) > 0:
        face_landmarks = landmarks[0:468].astype(np.int32)
        hull = cv2.convexHull(face_landmarks)
        cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
    return mask

def process_frame(args):
    i, aligned_frame, source_frame, M, original_shape, landmarks = args
    if M is not None and landmarks is not None:
        unaligned_frame = unalign_face(aligned_frame, M, original_shape)
        unaligned_frame = cv2.resize(unaligned_frame, (source_frame.shape[1], source_frame.shape[0]))
        
        face_mask = create_face_mask(landmarks, source_frame.shape)
        face_mask = face_mask.astype(np.float32) / 255.0
        
        unaligned_frame = match_histograms(unaligned_frame, source_frame)
        
        mask_3ch = np.repeat(face_mask[:, :, np.newaxis], 3, axis=2)
        
        result = source_frame * (1 - mask_3ch) + unaligned_frame * mask_3ch
        
        return i, result.astype(np.uint8)
    else:
        return i, source_frame

def match_histograms(source, reference):
    if source.ndim == 3 and reference.ndim == 3:
        multi_mask = np.all(source != 0, axis=2)
        result = source.copy()
        for i in range(3):
            source_channel = source[:,:,i][multi_mask]
            reference_channel = reference[:,:,i][multi_mask]
            if len(source_channel) > 0 and len(reference_channel) > 0:
                matched_channel = exposure.match_histograms(source_channel, reference_channel)
                result[:,:,i][multi_mask] = matched_channel
        return result
    else:
        return exposure.match_histograms(source, reference)

def unalign_video(aligned_path, source_path, transform_info, output_path):
    aligned_cap = cv2.VideoCapture(aligned_path)
    source_cap = cv2.VideoCapture(source_path)
    
    aligned_fps = aligned_cap.get(cv2.CAP_PROP_FPS)
    source_fps = source_cap.get(cv2.CAP_PROP_FPS)
    
    if abs(aligned_fps - source_fps) > 0.01:
        temp_source_path = 'temp_adjusted_source.mp4'
        subprocess.run(['ffmpeg', '-i', source_path, '-filter:v', f'fps={aligned_fps}', '-c:a', 'copy', temp_source_path])
        source_cap.release()
        source_cap = cv2.VideoCapture(temp_source_path)
        source_fps = aligned_fps
    
    fps = source_fps
    total_frames = transform_info['frame_count']

    _, first_frame = source_cap.read()
    height, width = first_frame.shape[:2]
    source_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    batch_size = 100  # Process frames in batches
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i in range(0, total_frames, batch_size):
            batch_end = min(i + batch_size, total_frames)
            futures = []
            
            for j in range(i, batch_end):
                ret_aligned, aligned_frame = aligned_cap.read()
                ret_source, source_frame = source_cap.read()
                if not ret_aligned or not ret_source:
                    break

                M, original_shape, frame_num, landmarks = transform_info['transforms'][j]
                future = executor.submit(process_frame, (frame_num, aligned_frame, source_frame, M, original_shape, landmarks))
                futures.append(future)
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

            results.sort(key=lambda x: x[0])

            for _, result in results:
                out.write(result)

    aligned_cap.release()
    source_cap.release()
    out.release()

    if abs(aligned_fps - source_fps) > 0.01:
        os.remove(temp_source_path)
