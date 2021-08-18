
import cv2
import dlib
import numpy as np
import math, random
from itertools import cycle
from imutils import face_utils
from playsound import playsound
from datetime import timedelta, datetime

class FacePyBird(object):
    def __init__(self, highscore_file: bool) -> None:
        self.numbers: list[np.ndarray] = [cv2.imread(f'./sprites/{i}.png', -1) for i in range(10)]
        self.players: list[np.ndarray] = self.get_players()
        self.message: np.ndarray = cv2.resize(cv2.imread('./sprites/message.png', -1), (135, 180), interpolation = cv2.INTER_AREA)
        self.gameover_orig: np.ndarray = cv2.imread('./sprites/scoreboard.png', -1) #30, 110
        self.gameover: np.ndarray = self.gameover_orig.copy()
        self.bronze: np.ndarray = cv2.resize(cv2.imread('./sprites/Bronze.png', -1), (50, 50), interpolation = cv2.INTER_AREA)
        self.silver: np.ndarray = cv2.resize(cv2.imread('./sprites/Silver.png', -1), (50, 50), interpolation = cv2.INTER_AREA)
        self.gold: np.ndarray = cv2.resize(cv2.imread('./sprites/Gold.png', -1), (50, 50), interpolation = cv2.INTER_AREA)
        self.platinum: np.ndarray = cv2.resize(cv2.imread('./sprites/Platinum.png', -1), (50, 50), interpolation = cv2.INTER_AREA)
        self.new: np.ndarray = cv2.resize(cv2.cvtColor(cv2.imread('./sprites/new.png', -1), cv2.COLOR_RGB2RGBA).copy(), (0, 0), fx = 0.6, fy = 0.6)
        
        self.player_cycle: cycle = cycle([0, 1, 2, 1])
        self.player_index: int = 0
        self.player_vel_y: int = -9
        self.player_max_vel_y: int = 10
        self.player_acc_y: int = 1
        self.player_rot: int = 45
        self.player_vel_rot: int = 3
        self.player_rot_threshold: int = 20
        self.visible_rot: int = 20
        self.falling: bool = True
        self.loop_iter: int = 0
        self.player_flap_acc: int = -9
        self.player_flapped: bool = False

        self.pipes: list[dict] = []
        self.pipe_time_buffer: timedelta = timedelta(seconds = 3)
        self.pipe_vel_x: int = -8

        self.player_height, self.player_width, _ = self.players[0].shape
        self.message_height, self.message_width, _ = self.message.shape
        self.gameover_height, self.gameover_width, _ = self.gameover.shape

        base: np.ndarray = cv2.imread('./sprites/base_full.png', -1)
        self.base: np.ndarray = cv2.cvtColor(base, cv2.COLOR_RGB2RGBA).copy()
        self.orig_base: np.ndarray = self.base.copy()
        self.base_x_offset: int = 10
        self.base_x: int = 0
        
        self.score: int = 0
        self.highscore_file: bool = highscore_file
        if self.highscore_file:
            score: _io.TextIOWrapper = open('./resources/highscore.txt', 'r')
            self.highscore: int = int(score.read())
            score.close()
        else:
            self.highscore: int = 0

        self.min_pitch_angle: float = 3.0
        self.max_pitch_angle: float = 12.0
        self.nod_allowed: bool = True
        self.nod: bool = False
        self.nod_time: datetime = datetime.now()
        self.crashed = False
        self.pipe_time: datetime = datetime.now()
        self.time_buffer: timedelta = timedelta(seconds=0.2)
        
        self.die: str = "./audio/die.wav"
        self.hit: str = "./audio/hit.wav"
        self.point: str = "./audio/point.wav"
        self.swoosh: str = "./audio/swoosh.wav"
        self.wing: str = "./audio/wing.wav"

        self.detector: _dlib_pybind11.fhog_object_detector = dlib.get_frontal_face_detector()
        self.predictor: _dlib_pybind11.shape_predictor = dlib.shape_predictor("./resources/shape_predictor_68_face_landmarks.dat")
        self.model_points: np.ndarray = np.array([
            (0.0, 0.0, 0.0),            # Nose tip
            (0.0, -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),    # Left eye left corner
            (225.0, 170.0, -135.0),     # Right eye right corner
            (-150.0, -150.0, -125.0),   # Left Mouth corner
            (150.0, -150.0, -125.0)     # Right mouth corner
        ])
        self.dist_coeffs: np.ndarray = np.zeros((4,1))

        self._introduction: bool = True
        self._maingame_seq: bool = False
        self._endgame_seq: bool = False

        self.init = False
    
    def reset_variables(self) -> None:
        """
        Resets the variables used in the program.
        """
        self.pipes *= 0
        self.nod = False
        self.score = 0
        self.pipe_time = datetime.now()
        self.pipe_vel_x = -8
        self.base = self.orig_base.copy()
        self.player_index = 0
        self.player_vel_y = -9
        self.player_max_vel_y = 10
        self.player_acc_y = 1
        self.player_rot = 45
        self.player_vel_rot = 3
        self.player_rot_threshold = 20
        self.visible_rot = 20
        self.falling = True
        self.gameover = self.gameover_orig.copy()
        self.crashed = False
        self._introduction: bool = True
        self._maingame_seq: bool = False
        self._endgame_seq: bool = False
        self.loop_iter: int = 0
        self.player_flap_acc: int = -9
        self.player_flapped: bool = False

    def init_dependent_vars(self):
        """
        Initialize the variables dependent on frame height and width.
        """
        pipe_dimensions: tuple = (int(self.width // 9), int(self.height // 2))
        self.lpipe: np.ndarray = cv2.resize(cv2.imread('./sprites/pipe.png', -1), pipe_dimensions, interpolation = cv2.INTER_AREA)
        self.upipe: np.ndarray = cv2.rotate(self.lpipe, cv2.ROTATE_180)
        self.pipe_height, self.pipe_width, _ = self.upipe.shape
        self.base_offset: int = self.height - int(self.base.shape[0] // 1.5)
        self.forehead: list = [self.width // 2, self.height // 3]
        self.nose: list = [self.width // 2, self.height // 2]
        self.camera_matrix: np.ndarray = np.array([
            [self.width, 0, self.width / 2],
            [0, self.width, self.height / 2],
            [0, 0, 1]
        ], dtype = "double")
    
    def play(self, soundfile: str) -> None:
        """
        Play the sound from the given soundfile.
        :param soundfile: Path to the soundfile to play.
        :type soundfile: String.
        """
        try:
            playsound(soundfile, False)
        except:
            pass

    def get_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Pass in an unprocessed frame, return a processed one
        :param frame: Image frame received from OpenCV VideoCapture object.
        :type frame: NumPy array (np.ndarray).
        :return: Processed frame.
        :rtype: NumPy array (np.ndarray).
        """
        frame = cv2.flip(frame, 1)
        if not self.init:
            self.height, self.width, _ = frame.shape
            self.init_dependent_vars()
            self.init = True
        
        self.update_facial_landmarks(frame, True)
        if self._introduction:
            self.introduction(frame)
        elif self._maingame_seq:
            self.main_game(frame)
        elif self._endgame_seq:
            self.show_gameover(frame)
        return frame
    
    def get_players(self) -> list[np.ndarray]:
        """
        Loads the images of each state of the flappy bird sprite, resizes them, and returns them in a list.
        """
        dimensions: tuple[int] = (45, 60)
        birds: list[np.ndarray] = [cv2.imread(f'./sprites/bird-{i}.png', -1) for i in range(1, 4)]
        return list(map(lambda x: cv2.resize(x, dimensions, interpolation = cv2.INTER_AREA), birds))
    
    def current_player(self) -> np.ndarray:
        """
        Rotates the Flappy Bird image by the visible rotation angle.
        :return: Rotated player image.
        :rtype: NumPy array (np.ndarray).
        """
        player: np.ndarray = self.players[self.player_index]
        center: tuple = tuple(np.array(player.shape[1::-1]) / 2)
        rot_mat: np.ndarray = cv2.getRotationMatrix2D(center, self.visible_rot, 1.0)
        player = cv2.warpAffine(player, rot_mat, player.shape[1::-1], flags=cv2.INTER_LINEAR)
        return player
    
    def update_facial_landmarks(self, frame: np.ndarray, show_points: bool) -> None:
        """
        Applies the facial landmark recognition model on the frame, then overlays
        key points on the frame. Also determines the pitch angle of the head, which is used
        to determine whether the player has nodded upwards. Updates key variables related to the face.
        :param frame: Image frame received from OpenCV VideoCapture object.
        :type frame: NumPy array (np.ndarray).
        :param show_points: Whether to show facial landmarks or not.
        :type show_points: Bool.
        """
        gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rect: list = self.detector(gray, 0)

        if not rect:
            return
        
        shape: _dlib_pybind11.full_object_detection = self.predictor(gray, rect[0])
        shape: np.ndarray = face_utils.shape_to_np(shape)

        if show_points:
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
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
        pitch, _, _ = [math.radians(angle) for angle in euler_angles]
        pitch: float = math.degrees(math.asin(math.sin(pitch)))

        reset_possible_angle: bool = not self.nod_allowed and pitch <= self.min_pitch_angle
        reset_possible_time: bool = datetime.now() - self.nod_time >= self.time_buffer
        
        if reset_possible_angle and reset_possible_time:
            self.nod_allowed = True
            self.nod_time = datetime.now()
        
        nod_occurred: bool = self.nod_allowed and pitch >= self.max_pitch_angle
        if nod_occurred:
            self.nod_allowed = False
            self.nod = True
        
        self.forehead = shape[27]
        self.nose = shape[30]
        if self.falling:
            self.x = self.nose[0]
    
    def overlay(self, frame: np.ndarray, image: np.ndarray, x: int, y: int) -> None:
        """
        Overlays the given image onto the frame at the coordinates (x, y).
        :param frame: Image frame received from OpenCV VideoCapture object.
        :type frame: NumPy array (np.ndarray).
        :param image: Image to be overlayed onto the frame.
        :type image: NumPy array (np.ndarray).
        :param x: The x-coordinate of the image.
        :type x: Int.
        :param y: The y-coordinate of the image.
        :type y: Int.
        """
        y1, y2 = max(0, y), min(frame.shape[0], y + image.shape[0])
        x1, x2 = max(0, x), min(frame.shape[1], x + image.shape[1])
        y1o, y2o = max(0, -y), min(image.shape[0], frame.shape[0] - y)
        x1o, x2o = max(0, -x), min(image.shape[1], frame.shape[1] - x)

        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        channels: int = frame.shape[2]
        alpha: float = image[y1o:y2o, x1o:x2o, 3] / 255.0
        alpha_inv: float = 1.0 - alpha

        for c in range(channels):
            frame[y1:y2, x1:x2, c] = (alpha * image[y1o:y2o, x1o:x2o, c] + alpha_inv * frame[y1:y2, x1:x2, c])

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

    def display_pipes(self, frame: np.ndarray) -> None:
        """
        Handles adding/deleting pipes from the pipes list, displaying those pipes on the frame,
        and updating the score should the player pass a pipe successfully.
        :param frame: Image frame received from OpenCV VideoCapture object.
        :type frame: NumPy array (np.ndarray).
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
                self.play(self.point)
            
            self.overlay(frame, self.upipe, pipe['x'], pipe['y'][0])
            self.overlay(frame, self.lpipe, pipe['x'], pipe['y'][1])

            self.pipes[i]['x'] += self.pipe_vel_x
        
        if self.pipes and self.pipes[0]['x'] < -self.pipe_width:
            self.pipes.pop(0)
        
        if self.base_x + self.base.shape[1] <= self.width + self.base_x_offset:
            self.base = self.base[:,-self.base_x:,:]
            self.base = cv2.hconcat([self.base, self.orig_base])
            self.base_x = 0
        
        self.overlay(frame, self.base, self.base_x, self.base_offset)
        self.base_x += self.pipe_vel_x
    
    def display_score(self, frame:np.ndarray, x: int, y: int) -> None:
        """
        Overlays the score of the player at the given coordinates.
        :param frame: Image frame received from OpenCV VideoCapture object.
        :type frame: NumPy array (np.ndarray).
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
            self.overlay(frame, self.numbers[i], x_offset, y_offset)
            x_offset += self.numbers[i].shape[1]
    
    def finalize_score(self) -> None:
        """
        Update gameover message to display correct score/highscore. 
        Save highscore in highscore file.
        """
        if self.score >= 40:
            self.overlay(self.gameover, self.platinum, 30, 110)
        elif self.score >= 30:
            self.overlay(self.gameover, self.gold, 30, 110)
        elif self.score >= 20:
            self.overlay(self.gameover, self.silver, 30, 110)
        elif self.score >= 10:
            self.overlay(self.gameover, self.bronze, 30, 110)
        
        points: list[int] = [int(i) for i in str(self.score)]
        numbers: list[np.ndarray] = [cv2.resize(self.numbers[i], (0, 0), fx = 0.6, fy = 0.6) for i in range(10)]
        total_width: int = np.sum([numbers[i].shape[1] for i in points])
        x: int = 210 - total_width
        y: int = 103
        for i in points:
            self.overlay(self.gameover, numbers[i], x, y)
            x += numbers[i].shape[1]

        if self.score > self.highscore:
            self.highscore = self.score
            self.overlay(self.gameover, self.new, 175 - self.new.shape[1], 130)
            if self.highscore_file:
                f: _io.TextIOWrapper = open('./resources/highscore.txt', 'w')
                f.write(str(self.highscore))
                f.close()
        
        points: list = [int(i) for i in str(self.highscore)]
        total_width: int = np.sum([numbers[i].shape[1] for i in points])
        x: int = 210 - total_width
        y: int = 145
        for i in points:
            self.overlay(self.gameover, numbers[i], x, y)
            x += numbers[i].shape[1]
        
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
            self.play(self.hit)
            self.play(self.die)
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
                    self.play(self.hit)
                    self.play(self.die)
                    self.crashed = True
                    return True

        return False
    
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

    def introduction(self, frame: np.ndarray) -> None:
        """
        Continuously overlays the introduction message until the player nods.
        :param frame: Image frame received from OpenCV VideoCapture object.
        :type frame: NumPy array (np.ndarray).
        """
        x: int = self.nose[0] - (self.message_width // 2)
        y: int = self.nose[1] - (self.message_height // 2)
        self.overlay(frame, self.message, x, y) 

        if self.nod:
            self.x, self.y = self.nose
            self.nod = False
            self.nod_allowed = False
            self._introduction = False
            self._maingame_seq = True
            self.play(self.wing)

    def main_game(self, frame: np.ndarray) -> None:
        """
        Main game sequence.
        :param frame: Image frame received from OpenCV VideoCapture object.
        :type frame: NumPy array (np.ndarray).
        """
        if self.check_crash():
            self.pipe_vel_x = 0
            self._maingame_seq = False
            self._endgame_seq = True
            self.finalize_score()
            return
        
        if self.nod:
            self.nod = False
            self.player_vel_y = self.player_flap_acc
            self.player_flapped = True
            self.play(self.wing)
        
        self.display_pipes(frame)

        if (self.loop_iter + 1) % 3 == 0:
            self.player_index = next(self.player_cycle)
        
        self.loop_iter = (self.loop_iter + 1) % 30

        if self.player_rot > -90:
            self.player_rot -= self.player_vel_rot
            
        if self.player_vel_y < self.player_max_vel_y and not self.player_flapped:
            self.player_vel_y += self.player_acc_y

        if self.player_flapped:
            self.player_flapped = False
            self.player_rot = 45
            
        self.y += self.player_vel_y
        self.display_score(frame, *self.forehead)
            
        self.visible_rot = self.player_rot_threshold
        if self.player_rot <= self.player_rot_threshold:
            self.visible_rot = self.player_rot
            
        x: int = self.nose[0] - (self.player_width // 2)
        y: int = self.y - (self.player_height // 2)
        self.overlay(frame, self.current_player(), x, y)
    
    def show_gameover(self, frame: np.ndarray) -> None:
        """
        Continuously overlays the gameover message until the player nods.
        :param frame: Image frame received from OpenCV VideoCapture object.
        :type frame: NumPy array (np.ndarray).
        """
        x: int = self.forehead[0] - (self.gameover_width // 2)
        y: int = self.forehead[1] - (self.gameover_height // 2)
        self.display_pipes(frame)
        self.overlay(frame, self.gameover, x, y)
            
        if self.falling:
            if self.player_rot > -90:
                self.player_rot -= self.player_vel_rot
                
            if self.player_vel_y < self.player_max_vel_y:
                self.player_vel_y += self.player_acc_y
                
            self.y += self.player_vel_y
                
            self.visible_rot = self.player_rot_threshold
            if self.player_rot <= self.player_rot_threshold:
                self.visible_rot = self.player_rot
            
        x: int = self.x - (self.player_width // 2)
        y: int = self.y - (self.player_height // 2)

        if self.y + (self.player_height // 2) >= self.base_offset:
            y = self.base_offset - self.player_height
            self.falling = False
            
        self.overlay(frame, self.current_player(), x, y)

        if self.nod:
            self.nod = False
            self.reset_variables()