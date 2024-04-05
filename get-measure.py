import numpy as np
import cv2 as cv
import glob
import os
import argparse
import sys
import textwrap
import json
import platform
from numpy.typing import NDArray
from typing import List, Tuple

# Variables globales para el dibujo
points = []  # Almacenará los puntos seleccionados
drawing_completed = False  # Indica si se completó el dibujo

def parser_data_user_arguments() -> argparse.ArgumentParser:
    """
    Parse user arguments.

    Returns:
        argparse.ArgumentParser: Object containing parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_index', 
                        type=int, 
                        required=True,
                        help="Index for desired camera ")
    parser.add_argument('--Z',
                        type=float,
                        required=True,
                        help="Scene depth")
    parser.add_argument('--cal_file',
                        type=str,
                        required=True,
                        help='JSON file containing calibration parameters')
    args = parser.parse_args()
    return args

def initialize_camera(args):
    """
    Initialize the camera.

    Args:
        args: Parsed command line arguments.

    Returns:
        cv.VideoCapture: Initialized camera object.
    """
    cap = cv.VideoCapture(args.cam_index)
    if not cap.isOpened():
        print("Error al inicializar la cámara")
        return None
    return cap

def mouse_callback(event, x, y, flags, param):
    """
    Callback function for mouse events.

    Args:
        event: Type of mouse event.
        x: X coordinate of the mouse event.
        y: Y coordinate of the mouse event.
        flags: Flags indicating the mouse event details.
        param: Additional parameters.

    Returns:
        None
    """
    global points, drawing_completed

    if event == cv.EVENT_LBUTTONDOWN:
        if flags & cv.EVENT_FLAG_CTRLKEY:
            if points:
                points.clear()
        elif flags & cv.EVENT_FLAG_ALTKEY:
            if points:
                points.pop()
        else:
            points.append((x, y))  # Agregar el punto seleccionado
    elif event == cv.EVENT_MBUTTONDOWN:
        points.append(points[0])  # Cerrar la figura
        drawing_completed = True

def load_calibration_parameters_from_json_file(args:argparse.ArgumentParser) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load calibration parameters from a JSON file.

    Args:
        args: Parsed command line arguments.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing camera matrix and distortion coefficients.
    """
    json_filename = args.cal_file
    if not os.path.isfile(json_filename):
        print(f"The file {json_filename} does not exist!")
        sys.exit(-1)
        
    with open(json_filename) as f:
        json_data = json.load(f)
        camera_matrix = np.array(json_data['camera_matrix'])
        distortion_coefficients = np.array(json_data['distortion_coefficients'])
    return camera_matrix, distortion_coefficients

def compute_perimeter(points: List[Tuple[int, int]], Z: float, mtx: np.ndarray, height:int, width:int) -> Tuple[List[float], float]:
    """
    Compute the perimeter of a closed figure defined by a set of points in the image.

    Args:
        points: List of points defining the figure.
        Z: Scene depth.
        mtx: Camera matrix.
        height: Height of the image.
        width: Width of the image.

    Returns:
        Tuple[List[float], float]: Tuple containing a list of distances between consecutive points and the total perimeter.
    """
    distances = []
    total_perimeter = 0.0

    Cx = mtx[0,2]*width / 4080
    Cy = mtx[1,2]*height / 3060
    fx = mtx[0,0]*width /4080
    fy = mtx[1,1]*height / 3060
    
    for i in range(1, len(points)):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        
        # Convertir las coordenadas de píxeles a coordenadas 3D
        X1 = (x1 - Cx) * Z / fx
        Y1 = (y1 - Cy) * Z / fy
        X2 = (x2 - Cx) * Z / fx
        Y2 = (y2 - Cy) * Z / fy
        
        # Calcular la distancia entre los puntos en 3D
        dist_3d = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2 )
        distances.append(dist_3d)
    
        
        total_perimeter += dist_3d
    return distances, total_perimeter

def compute_line_segments(points: List[Tuple[int, int]]) -> List[float]:
    """
    Compute the lengths of line segments defined by a set of points in the image.

    Args:
        points: List of points defining the line segments.

    Returns:
        List[float]: List containing lengths of the line segments.
    """
    line_lengths = []
    for i in range(1, len(points)):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        line_lengths.append(length)
    return line_lengths

def undistort_images(
        list_of_distorted_images: List[str], 
        mtx: NDArray, 
        dist: NDArray, 
        path_to_saving_undistorted_images: str
        ) -> None:
    """
    Undistort a list of distorted images using camera calibration parameters and save 
    the undistorted images.

    Args:
        list_of_distorted_images: List of paths to distorted images.
        mtx: Camera matrix.
        dist: Distortion coefficients.
        path_to_saving_undistorted_images: Path to save undistorted images.

    Returns:
        None: The function does not return any value.
    """

    # Loop through distorted images
    for fname in list_of_distorted_images:
        print("Undistorting: {}".format(fname))

        # read current distorted image
        img = cv.imread(fname)

        # Get size
        h, w = img.shape[:2]

        # Get optimal new camera matrix
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

        # Undistort image
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)

        # Crop image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        # Save undistorted image
        undistorted_img_path = os.path.join(path_to_saving_undistorted_images, os.path.basename(fname))
        cv.imwrite(undistorted_img_path, dst)

def run_pipeline():
    global drawing_completed  # Declarar como global
    args = parser_data_user_arguments()
    # Inicializar la cámara
    cap = initialize_camera(args)
    if cap is None:
        print("La cámara no está disponible")
        sys.exit(-1)

    cv.namedWindow('image')
    cv.setMouseCallback('image', mouse_callback)

    # Cargar los parámetros de calibración
    mtx, dist = load_calibration_parameters_from_json_file(args)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el fotograma")
            break
        
        height,width = frame.shape[:2]

        if drawing_completed:
            # Calcular distancias y perímetro total
            distances, total_perimeter = compute_perimeter(points, args.Z, mtx, height, width)

            # Construir el texto para mostrar en la terminal
            text = "Distancias (selección de puntos):\n"
            for i, dist in enumerate(distances, start=1):
                if i < len(points):
                    text += f"Punto {i}-{i+1}: {dist:.2f} cm\n"
                else:
                    text += f"Punto {i}-{1}: {dist:.2f} cm\n"
            
            text += "\nMedidas ordenadas de mayor a menor:\n"
            sorted_distances = sorted(distances, reverse=True)
            for i, dist in enumerate(sorted_distances, start=1):
                index = distances.index(dist)
                if index == len(points) - 1:
                    text += f"Punto {len(points)}-{1}: {dist:.2f} cm\n"
                else:
                    text += f"Punto {index+1}-{index+2}: {dist:.2f} cm\n"
            
            text += f"\nPerímetro total: {total_perimeter:.2f} cm"

            # Imprimir el texto en la terminal
            print(text)
            drawing_completed = False  # Reiniciar la variable para la próxima figura

        # Dibujar las líneas entre los puntos seleccionados
        for i in range(1, len(points)):
            cv.line(frame, points[i-1], points[i], (0, 255, 0), 1)

        # Dibujar los puntos seleccionados
        for point in points:
            cv.circle(frame, point, 3, (0, 255, 0), -1)

        cv.imshow('image', frame)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    run_pipeline()