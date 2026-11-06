import os
import argparse
import dlib
import numpy as np
import cv2
import PIL.Image
from tqdm import tqdm

# Load face detector and landmark predictor
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(image):
    """ Detects faces and extracts 68 facial landmarks. """
    faces = detector(image, 1)
    return [np.array([(p.x, p.y) for p in predictor(image, face).parts()]) for face in faces]

def shrink(image, quad, qsize, output_size):
    """ Resizes image to speed up processing while preserving detail. """
    shrink_factor = int(np.floor(qsize / output_size * 0.5))
    if shrink_factor > 1:
        rsize = (int(image.size[0] / shrink_factor), int(image.size[1] / shrink_factor))
        image = image.resize(rsize, PIL.Image.Resampling.LANCZOS)
        quad /= shrink_factor
        qsize /= shrink_factor
    return image, quad, qsize

def transform(image, quad, transform_size, output_size):
    """ Applies perspective transformation to align the face. """
    image = image.transform((transform_size, transform_size), PIL.Image.QUAD,
                            (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        image = image.resize((output_size, output_size), PIL.Image.Resampling.LANCZOS)
    return image

def align_face(image_path, output_path, landmarks, output_size=256, transform_size=4096, padding_factor=0.2):
    """ Aligns face using eye and mouth landmarks while preserving full face visibility. """
    img = PIL.Image.open(image_path).convert('RGB')
    left_eye, right_eye = np.mean(landmarks[36:42], axis=0), np.mean(landmarks[42:48], axis=0)
    mouth = np.mean(landmarks[48:60], axis=0)

    eye_avg = (left_eye + right_eye) / 2
    eye_to_eye = right_eye - left_eye
    eye_to_mouth = mouth - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * (2 + padding_factor), np.hypot(*eye_to_mouth) * (1.8 + padding_factor))
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img, quad, qsize = shrink(img, quad, qsize, output_size)
    img = transform(img, quad, transform_size, output_size)
    
    img.save(output_path, 'PNG')

def process_image(image_path, output_dir, output_size):
    """ Detects, aligns, and saves all faces in an image. """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    landmarks_list = get_landmarks(gray)

    os.makedirs(output_dir, exist_ok=True)
    
    for i, landmarks in enumerate(landmarks_list, start=1):
        output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{i}.png"
        output_path = os.path.join(output_dir, output_filename)
        align_face(image_path, output_path, landmarks, output_size)

def batch_process(input_dir, output_dir, output_size):
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    for img_name in tqdm(files, desc="Processing Images"):
        process_image(os.path.join(input_dir, img_name), output_dir, output_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align faces in images with transformations.")
    parser.add_argument("-i", "--input", type=str, default="imgs", help="Path to input images directory")
    parser.add_argument("-o", "--output", type=str, default="alignedimgs", help="Path to output directory for aligned faces")
    parser.add_argument("-s", "--size", type=int, default=256, help="Output image size (default: 256)")

    args = parser.parse_args()
    batch_process(args.input, args.output, args.size)
    print(f"Aligned faces saved in {args.output}")
