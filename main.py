import cv2
import math, random
from itertools import cycle
from datetime import timedelta, datetime
import numpy as np
import playsound
import dlib
from imutils import face_utils

class FlappyBird:
    def __init__(self) -> None:
        self.numbers: list = [cv2.imread(f'./sprites/{i}.png', -1) for i in range(10)]
        self.players: list = self.get_players()
        self.message: np.ndarray = cv2.resize(cv2.imread('./sprites/message.png', -1), (135, 180), interpolation = cv2.INTER_AREA)
        self.gameover: np.ndarray = cv2.resize(cv2.imread('./sprites/gameover.png', -1), (180, 36), interpolation = cv2.INTER_AREA)

        self.player_cycle: cycle = cycle([0, 1, 2, 1])

        self.window: str = 'FlapPy Bird OpenCV by @victor-hugo-dc'
        self.capture: cv2.VideoCapture = cv2.VideoCapture(0)
        self.width: int = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        pipe_dimensions: tuple = (self.width // 9, self.height // 2)
        self.lpipe: np.ndarray = cv2.resize(cv2.imread('./sprites/pipe.png', -1), pipe_dimensions, interpolation = cv2.INTER_AREA)
        self.upipe: np.ndarray = cv2.rotate(self.lpipe, cv2.ROTATE_180)
        self.pipes: list = []
        self.pipe_time_buffer: timedelta = timedelta(seconds = 3)
        self.pipe_vel_x: int = -8

        self.player_height, self.player_width, _ = self.players[0].shape
        self.message_height, self.message_width, _ = self.message.shape
        self.gameover_height, self.gameover_width, _ = self.gameover.shape
        self.pipe_height, self.pipe_width, _ = self.upipe.shape

        base: np.ndarray = cv2.imread('./sprites/base.png', -1)
        base: np.ndarray = cv2.hconcat([base, base])
        self.base: np.ndarray = cv2.cvtColor(base, cv2.COLOR_RGB2RGBA).copy()
        self.base_offset: int = self.height - int(self.base.shape[0] // 1.5)
        
        self.score: int = 0

        self.min_pitch_angle: float = 3.0
        self.max_pitch_angle: float = 12.0
        self.nod_allowed: bool = True
        self.nod: bool = False
        self.crashed = False
        self.pipe_time: datetime = datetime.now()
        self.time_buffer: timedelta = timedelta(seconds=1)
        
        self.forehead: list = [self.width // 2, self.height // 3]
        self.nose: list = [self.width // 2, self.height // 2]

        self.die: str = "./audio/die.wav"
        self.hit: str = "./audio/hit.wav"
        self.point: str = "./audio/point.wav"
        self.swoosh: str = "./audio/swoosh.wav"
        self.wing: str = "./audio/wing.wav"

        self.detector: _dlib_pybind11.fhog_object_detector = dlib.get_frontal_face_detector()
        self.predictor: _dlib_pybind11.shape_predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
        self.model_points: np.ndarray = np.array([
            (0.0, 0.0, 0.0),            # Nose tip
            (0.0, -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),    # Left eye left corner
            (225.0, 170.0, -135.0),     # Right eye right corner
            (-150.0, -150.0, -125.0),   # Left Mouth corner
            (150.0, -150.0, -125.0)     # Right mouth corner
        ])
        self.camera_matrix: np.ndarray = np.array([
            [self.width, 0, self.width / 2],
            [0, self.width, self.height / 2],
            [0, 0, 1]
        ], dtype = "double")
        self.dist_coeffs: np.ndarray = np.zeros((4,1))

    def update_frame(self) -> None:
        flag, self.frame = self.capture.read()
        if not flag:
            raise Exception
        self.frame = cv2.flip(self.frame, 1)
        self.update_facial_landmarks()
    
    def update_facial_landmarks(self) -> None:
        """
        Applies the facial landmark recognition model on the frame, then overlays
        key points on the frame. Also determines the pitch angle of the head, which is used
        to determine whether the player has nodded upwards. Updates key variables related to the face.
        """
        gray: np.ndarray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        rect: list = self.detector(gray, 0)

        if not rect:
            return
    
        shape: _dlib_pybind11.full_object_detection = self.predictor(gray, rect[0])
        shape: np.ndarray = face_utils.shape_to_np(shape)

        for (x, y) in shape:
            cv2.circle(self.frame, (x, y), 2, (0, 255, 0), -1)

        image_points: np.ndarray = np.array([
            shape[30],  # Nose tip
            shape[8],   # Chin
            shape[36],  # Left eye left corner
            shape[45],  # Right eye right corne
            shape[48],  # Left Mouth corner
            shape[54]   # Right mouth corner
        ], dtype="double")

        (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, image_points, self.camera_matrix, self.dist_coeffs, flags = cv2.SOLVEPNP_UPNP)
        rotation_matrix: np.ndarray = cv2.Rodrigues(rotation_vector)[0]
        pose_matrix: np.ndarray = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
        pitch, _, _ = [math.radians(_) for _ in euler_angles]
        pitch: float = math.degrees(math.asin(math.sin(pitch)))

        if not self.nod_allowed and pitch <= self.min_pitch_angle:
            self.nod_allowed = True

        if self.nod_allowed and pitch >= self.max_pitch_angle:
            self.nod_allowed = False
            self.nod = True

        self.forehead = shape[27]
        self.nose = shape[30]
        self.x = self.nose[0]

    def reset_variables(self) -> None:
        """
        Resets the variables used in the program.
        """
        self.pipes *= 0
        self.nod = False
        self.score = 0
        self.pipe_time = datetime.now()
        self.pipe_vel_x = -8
    
    def add_random_pipe(self) -> None:
        """
        Adds a pair of pipes to the pipe list.
        """
        pipe_separation: int = int(self.base_offset // 2.5)
        min_pipe_y: int = - int(self.pipe_height * 0.8)
        max_pipe_y: int = self.base_offset - pipe_separation - int(self.pipe_height * 1.2)

        upipe: int = random.randint(min_pipe_y, max_pipe_y)
        lpipe: int = upipe + pipe_separation + self.pipe_height
        npipe: dict = {
            'x': self.width,
            'y': [
                upipe,
                lpipe
            ],
            'passed': False
        }
        self.pipes.append(npipe)

    def get_players(self) -> list:
        """
        Loads the images of each state of the flappy bird sprite, resizes them, and returns them in a list.
        """
        dimensions: tuple = (45, 60)
        bird_1: np.ndarray = cv2.imread(f'./sprites/bird-1.png', -1)
        bird_2: np.ndarray = cv2.imread(f'./sprites/bird-2.png', -1)
        bird_3: np.ndarray = cv2.imread(f'./sprites/bird-3.png', -1)

        bird_1: np.ndarray = cv2.resize(bird_1, dimensions, interpolation = cv2.INTER_AREA)
        bird_2: np.ndarray = cv2.resize(bird_2, dimensions, interpolation = cv2.INTER_AREA)
        bird_3: np.ndarray = cv2.resize(bird_3, dimensions, interpolation = cv2.INTER_AREA)

        return [bird_1, bird_2, bird_3]

    def introduction(self) -> bool:
        """
        Continuously overlays the introduction message until the player nods.
        :return: Whether the program should continue or not.
        :rtype: bool.
        """
        self.crashed = False
        while True:
            if cv2.waitKey(20) == 27:
                break

            self.update_frame()
            x: int = self.nose[0] - (self.message_width // 2)
            y: int = self.nose[1] - (self.message_height // 2)
            self.overlay(self.message, x, y)
            cv2.imshow(self.window, self.frame)

            if self.nod:
                self.x, self.y = self.nose
                self.nod = False
                self.nod_allowed = False
                playsound.playsound(self.wing, False)
                return True

        return False

    def game(self, continue_: bool) -> bool:
        """
        Main game sequence.
        :param continue_: Whether the program should continue or not.
        :type continue_: bool.
        :return: Whether the program should continue or not.
        :rtype: bool.
        """
        player_index: int = 0
        loop_iter: int = 0
        player_vel_y: int = -9
        player_max_vel_y: int = 10
        player_acc_y: int = 1
        player_rot: int = 45
        player_vel_rot: int = 3
        player_rot_threshold: int = 20
        player_flap_acc: int = -9
        player_flapped: bool = False
        
        while continue_:
            if cv2.waitKey(20) == 27: # exit on ESC
                break

            self.update_frame()
            if self.check_crash():
                self.pipe_vel_x = 0
                return True

            if self.nod:
                self.nod = False
                player_vel_y = player_flap_acc
                player_flapped = True
                playsound.playsound(self.wing, False)

            self.display_pipes()
            
            if (loop_iter + 1) % 3 == 0:
                player_index = next(self.player_cycle)

            loop_iter = (loop_iter + 1) % 30

            if player_rot > -90:
                player_rot -= player_vel_rot
            
            if player_vel_y < player_max_vel_y and not player_flapped:
                player_vel_y += player_acc_y

            if player_flapped:
                player_flapped = False
                player_rot = 45
            
            self.y += player_vel_y
            self.display_score(*self.forehead)
            
            visible_rot = player_rot_threshold
            if player_rot <= player_rot_threshold:
                visible_rot = player_rot
            
            player: np.ndarray = self.rotate_image(self.players[player_index], visible_rot)
            x: int = self.nose[0] - (self.player_width // 2)
            y: int = self.y - (self.player_height // 2)
            self.overlay(player, x, y)
            cv2.imshow(self.window, self.frame)

        return False

    def show_gameover(self, continue_: bool) -> bool:
        """
        Continuously overlays the gameover message until the player nods.
        :param continue_: Whether the program should continue or not.
        :type continue_: bool.
        :return: Whether the program should continue or not.
        :rtype: bool.
        """
        while continue_:
            if cv2.waitKey(20) == 27:
                break

            self.update_frame()
            x: int = self.forehead[0] - (self.gameover_width // 2)
            y: int = self.forehead[1] - (self.gameover_height // 2)
            self.overlay(self.gameover, x, y)
            self.display_score(*self.nose)
            self.display_pipes()
            cv2.imshow(self.window, self.frame)

            if self.nod:
                self.nod = False
                return True
        
        return False

    def display_pipes(self) -> None:
        """
        Handles adding/deleting pipes from the pipes list, displaying those pipes on the frame,
        and updating the score should the player pass a pipe successfully.
        """
        if datetime.now() - self.pipe_time >= self.pipe_time_buffer:
            self.add_random_pipe()
            self.pipe_time = datetime.now()
        
        buffer: int = 4
        for i, pipe in enumerate(self.pipes):
            midpoint: int = pipe['x'] + (self.pipe_width // 2)
            if not self.crashed and midpoint - buffer <= self.x <= midpoint + buffer and not pipe['passed']:
                self.score += 1
                self.pipes[i]['passed'] = True
                playsound.playsound(self.point, False)
            
            self.overlay(self.upipe, pipe['x'], pipe['y'][0])
            self.overlay(self.lpipe, pipe['x'], pipe['y'][1])

            self.pipes[i]['x'] += self.pipe_vel_x
        
        if self.pipes and self.pipes[0]['x'] < -self.pipe_width:
            self.pipes.pop(0)
        
        self.overlay(self.base, 0, self.base_offset)

    def overlay(self, image: np.ndarray, x: int, y: int) -> None:
        """
        Overlays the given image onto the frame at the coordinates (x, y).
        :param image: Image to be overlayed onto the frame.
        :type image: NumPy array (np.ndarray).
        :param x: The x-coordinate of the image.
        :type x: Int.
        :param y: The y-coordinate of the image.
        :type y: Int.
        """
        y1, y2 = max(0, y), min(self.frame.shape[0], y + image.shape[0])
        x1, x2 = max(0, x), min(self.frame.shape[1], x + image.shape[1])
        y1o, y2o = max(0, -y), min(image.shape[0], self.frame.shape[0] - y)
        x1o, x2o = max(0, -x), min(image.shape[1], self.frame.shape[1] - x)

        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        channels: int = self.frame.shape[2]
        alpha: float = image[y1o:y2o, x1o:x2o, 3] / 255.0
        alpha_inv: float = 1.0 - alpha

        for c in range(channels):
            self.frame[y1:y2, x1:x2, c] = (alpha * image[y1o:y2o, x1o:x2o, c] + alpha_inv * self.frame[y1:y2, x1:x2, c])

    def rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """
        @author: Alex Rodrigues, StackOverflow
        Rotates the image by a specified angle.
        :param image: Image to be rotated.
        :type image: NumPy array (np.ndarray).
        :param angle: The angle to rotate the image by.
        :type angle: Int.
        :return: Rotated image.
        :rtype: NumPy array (np.ndarray).
        """
        center: tuple = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat: np.ndarray = cv2.getRotationMatrix2D(center, angle, 1.0)
        result: np.ndarray = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def display_score(self, x: int, y: int) -> None:
        """
        Overlays the score of the player at the given coordinates.
        :param x: The x-coordinate of the image.
        :type x: Int.
        :param y: The y-coordinate of the image.
        :type y: Int.
        """
        points: list = [int(i) for i in str(self.score)]
        total_width: int = np.sum([self.numbers[i].shape[1] for i in points])
        x_offset: int = x - (total_width // 2)
        y_offset: int = y - (self.numbers[0].shape[0] // 2)
        for i in points:
            self.overlay(self.numbers[i], x_offset, y_offset)
            x_offset += self.numbers[i].shape[1]

    def check_crash(self) -> bool:
        """
        Checks whether the player has crashed into any of the borders or pipes.
        :return: Whether the player has crashed or not.
        :rtype: Bool.
        """
        height: int = self.player_height // 2
        width: int = self.player_width // 2
        if self.y - height <= 0 or self.y + height >= self.base_offset:
            # TODO: implement pixel-perfect collisions using get_hitmask
            playsound.playsound(self.hit, False)
            playsound.playsound(self.die, False)
            self.crashed = True
            return True

        for pipe in self.pipes:
            front: bool = pipe['x'] <= self.x + width <= pipe['x'] + self.pipe_width
            back: bool = pipe['x'] <= self.x - width <= pipe['x'] + self.pipe_width
            if front or back:
                upper: bool = 0 <= self.y - height <= pipe['y'][0] + self.pipe_height
                lower: bool = pipe['y'][1] <= self.y + height <= self.base_offset
                if upper or lower:
                    # TODO: implement pixel-perfect collisions using get_hitmask
                    playsound.playsound(self.hit, False)
                    playsound.playsound(self.die, False)
                    self.crashed = True
                    return True

        return False

    def play(self) -> None:
        """
        Full game sequence.
        """
        while True:
            intro: bool = self.introduction()
            game: bool = self.game(intro)
            if not self.show_gameover(game):
                break
            self.reset_variables()
        
        self.quit()
        print("Game Over.")

    '''
    TODO: use hitmask to do pixel-perfect collisions.
    def get_hitmask(self, image) -> list:
        mask: list = []
        for x in range(image.shape[1]):
            mask.append([])
            for y in range(image.shape[0]):
                alpha: float = image[y, x, 3] / 255.0
                mask[x].append(bool(alpha))
        return mask
    '''

    def quit(self) -> None:
        """
        Stops the video capture and closes all the OpenCV windows.
        """
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    FlappyBird().play()