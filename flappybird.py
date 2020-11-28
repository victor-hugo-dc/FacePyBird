import cv2
import math, random
from itertools import cycle
from datetime import timedelta, datetime
import numpy as np
from scipy.signal import savgol_filter
from scipy import stats
from facialrecog import MarkDetector, draw_annotation_box
import playsound
import threading

class FlappyBird:
    def __init__(self) -> None:
        np.seterr(divide='ignore')
        # loading and resizing our images
        self.number = [cv2.imread(f'sprites\\{i}.png', -1) for i in range(10)] # list containing the images of the numbers for scoring
        self.player = self.get_players() # gets a list of resized images of each state of the bird's wing
        self.message = cv2.resize(cv2.imread('sprites\\message.png', -1), (135, 180), interpolation = cv2.INTER_AREA) # the introductory message
        self.gameover = cv2.resize(cv2.imread('sprites\\gameover.png', -1), (180, 36), interpolation = cv2.INTER_AREA) # the gameover text message

        # movement utilized during introduction
        self.player_cycle = cycle([0, 1, 2, 1]) # player index cycle
        self.player_shm_vals = {'val': 0, 'dir': 1} # simple harmonic motion for the player bird

        # OpenCV frame initialization
        self.window = 'Flappy Bird OpenCV by @victor-hugo-dc' # window name
        self.vc = cv2.VideoCapture(0) # video capture object
        self.rval, self.frame = self.vc.read() # get the frame from the video capture object
        self.frame = cv2.flip(self.frame, 1) # flip the frame
        
        # pipe-related variables
        pipe_dims = (int(self.frame.shape[1] // 9), int(self.frame.shape[0] // 2)) # dimensions for the pipes
        self.pipe_bottom = cv2.resize(cv2.imread('sprites\\pipe.png', -1), pipe_dims, interpolation = cv2.INTER_AREA) # read the pipe and resize the image
        self.pipe_top = cv2.rotate(self.pipe_bottom, cv2.ROTATE_180) # rotate the pipe by 180 degrees
        self.pipes = [] # list that will hold onto our pipe objects
        self.pipe_time_buffer = timedelta(seconds=5) # the time in between pipes, 5-7 seconds range is optimal
        self.pipe_vel_x = -4 # velocity of the pipes

        # measurements
        self.height, self.width = self.frame.shape[:2] # get the dimensions of the frame
        self.player_height, self.player_width = self.player[0].shape[:2] # get the dimensions of the player
        self.message_height, self.message_width = self.message.shape[:2] # get the dimensions of the introduction message
        self.gameover_height, self.gameover_width = self.gameover.shape[:2] # get the dimensions of the gameover message
        self.pipe_height, self.pipe_width = self.pipe_top.shape[:2] # get the dimensions of the pipe image

        # the ground a.k.a the "base"
        # **it could be possible to instead resize the base.png to best fit the screen but in my
        #   attempts it has come out looking strange every time, therefore this was my solution:
        base_img = cv2.imread('sprites\\base.png', -1) # read in the base image
        base_img = cv2.hconcat([base_img, base_img]) # concatenate the base image with a copy horizontally
        self.base = cv2.cvtColor(base_img, cv2.COLOR_RGB2RGBA).copy() # sets the base equal to a copy that turns the base into a four-dimensional image: (r, g, b, a)
        self.base_offset = self.height - int(self.base.shape[0] // 1.5) # we calculate a distance in which to draw the base, I got this on trial and error, may be corrected 
        
        self.score = 0 # self explanatory

        # face detection variables
        self.angle_one, self.angle_two = [], [] # create two arrays which will each store angles created from the player's nose to a calculated line
        self.mark_detector = MarkDetector() # create the mark detector object
        self.nod = False # boolean, True if the most recently added angle is an outlier in the list of angles (the player's head has nodded)
        self.nod_time = self.pipe_time = datetime.now() # time variables, used in preventing multiple self.nod True values back-to-back and knowing when to add a new pipe 
        self.time_buffer = timedelta(seconds=1) # time buffer for the self.nod_time variable
        self.model_points = np.array([  # used in facial landmark detection
            (0.0, 0.0, 0.0),            # Nose tip
            (0.0, -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),    # Left eye left corner
            (225.0, 170.0, -135.0),     # Right eye right corner
            (-150.0, -150.0, -125.0),   # Left Mouth corner
            (150.0, -150.0, -125.0)     # Right mouth corner
        ])
        self.camera_matrix = np.array([ # used in facial landmark detection
            [self.width, 0, self.width / 2],
            [0, self.width, self.height / 2],
            [0, 0, 1]
        ], dtype = "double")
        # position variables, initialize to None
        self.forehead, self.nose, self.prev_forehead, self.prev_nose = None, None, None, None

        # paths to audio files
        self.die = "audio\\die.wav"
        self.hit = "audio\\hit.wav"
        self.point = "audio\\point.wav"
        self.swoosh = "audio\\swoosh.wav"
        self.wing = "audio\\wing.wav"

    def reset_variables(self) -> None:
        self.pipes *= 0 # clear pipes list
        self.nod = False # ensure that the node variable is False
        self.score = 0 # reset score
        self.angle_one *= 0 # clear angles lists
        self.angle_two *= 0 
        self.nod_time = self.pipe_time = datetime.now() # reset time starts

    def get_random_pipe(self) -> dict: # returns a dict containing the information for a new random pipe

        pipe_sep = int(self.base_offset / 2.5) # the size of the separation between the pipes
        min_pipe_y = - int(self.pipe_height * 0.8) # the minimum y-value at which we will draw the top pipe on
        max_pipe_y = self.base_offset - pipe_sep - int(self.pipe_height * 1.2) # the maximum y-value at which we will draw the top pipe on

        top_pipe = random.randint(min_pipe_y, max_pipe_y) # the y-position for the top pipe
        bottom_pipe = top_pipe + pipe_sep + self.pipe_height # the y-position for the bottom pipe

        return {
            'x': self.width, # when a pipe is initially added, it will be at the edge of the screen
            'y': [
                top_pipe, # The y-value for when the top pipe begins
                bottom_pipe # The y-value for when the bottom pipe begins
            ],
            'passed': False # the boolean value that lets us know if we have already passed the pipe
        }

    def get_players(self) -> list:
        # I kept getting an error when I tried to read and resize a bird image within a list comprehension so I tried it like this and it worked
        bird_dim = (45, 60) # Dimensions for Flappy Bird
        # read in the flappy bird images, bird images are named in order
        bird_one = cv2.imread(f'sprites\\bird-1.png', -1) 
        bird_two = cv2.imread(f'sprites\\bird-2.png', -1)
        bird_three = cv2.imread(f'sprites\\bird-3.png', -1)

        # resize the bird images to the proper dimensions
        bird_one = cv2.resize(bird_one, bird_dim, interpolation = cv2.INTER_AREA)
        bird_two = cv2.resize(bird_two, bird_dim, interpolation = cv2.INTER_AREA)
        bird_three = cv2.resize(bird_three, bird_dim, interpolation = cv2.INTER_AREA)

        return [bird_one, bird_two, bird_three] # return the list of the birds

    def get_facial_landmarks(self) -> list:
        faceboxes = self.mark_detector.extract_cnn_facebox(self.frame) # get the faceboxes fom the mark detector with the frame as input for the convolutional neural network
        for facebox in faceboxes: # loop through the faceboxes 
            face_img = self.frame[facebox[1] : facebox[3], facebox[0] : facebox[2]] # using the values from the neural network, crops face_img to only include the face from frame
            face_img = cv2.resize(face_img, (128, 128)) # resize our face image to a 128 x 128 pixel image
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) # OpenCV reads images as BGR if read through imread(), unsure if this line should stay or be commented out***

            marks = self.mark_detector.detect_marks([face_img]) # returns landmarks from face_img to marks (numpy array)
            marks *= (facebox[2] - facebox[0]) 
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]
            shape = marks.astype(np.uint)
            self.mark_detector.draw_marks(self.frame, marks, color=(0, 255, 0)) # draw the marks on the frame
            image_points = np.array([
                shape[30],  # Nose tip
                shape[8],   # Chin
                shape[36],  # Left eye left corner
                shape[45],  # Right eye right corne
                shape[48],  # Left Mouth corner
                shape[54]   # Right mouth corner
            ], dtype="double")
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose
            (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, self.camera_matrix, dist_coeffs)
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = draw_annotation_box(self.frame, rotation_vector, translation_vector, self.camera_matrix)
            # Find the angles created by the above lines
            try:
                m = (p2[1] - p1[1])/(p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
            except:
                ang1 = 90

            try:
                m = (x2[1] - x1[1])/(x2[0] - x1[0])
                ang2 = int(math.degrees(math.atan(-1/m)))
            except:
                ang2 = 90

            # append the angles to the lists
            self.angle_one.append(ang1)
            self.angle_two.append(ang2)

            try: # if len(self.angle_one) >= 5 and len(self.angle_two) >= 5
                self.angle_one = savgol_filter(self.angle_one, 5, 2, mode='nearest').tolist() # Savitzky-Golay filter applied to x for smoothing, np array
                self.angle_two = savgol_filter(self.angle_two, 5, 2, mode='nearest').tolist()
            except:
                pass

            # if the difference between most recently added angle and the mean of the angles is greater than three times the standard deviation of the angles
            # and it has been at least time_buffer seconds since the last nod.
            if abs(self.angle_one[-1] - np.mean(self.angle_one)) > 3 * np.std(self.angle_one) and datetime.now() - self.nod_time >= self.time_buffer:
                self.nod_time = datetime.now() # reset the nod time
                return [shape[27], shape[30], True]
            
            self.angle_one, self.angle_two = self.angle_one[-30:], self.angle_two[-30:]

            return [shape[27], shape[30], False] # middle of the forehead, tip of the nose 
    

    def update_face_variables(self) -> None:
        try:
            self.forehead, self.nose, self.nod = self.get_facial_landmarks() # update the current position variables and the nod boolean
        except: # in the event the above line fails, it will be an error in which a NoneType cannot be unpacked
            if self.prev_forehead is not None and self.prev_nose is not None: # if our previous variables are not None, set the current position variables to the previous variables
                self.forehead, self.nose, self.nod = self.prev_forehead, self.prev_nose, False # and set the nod variable to False
            else: # if any of our previous variables are None, we can simply set the current position variables to positions along the center of the frame
                self.forehead = [int(self.width // 2), int(self.height // 3)] # the forehead and the nose then will be set to equivalent x-values
                self.nose = [int(self.width // 2), int(self.height // 2)] # and the y-value of self.forehead will be drawn slightly higher than self.nose
                self.nod = False # set to False to prevent any error

        if self.forehead is not None: # if the current forehead is not None,
            self.prev_forehead = self.forehead # set the previous forehead position to the current position
            
        if self.nose is not None: # follow the same logic as above on the nose variable
            self.prev_nose = self.nose
        

    def introduction(self) -> bool: # continuously displays the welcome message on the player's face, returns True when head nods
        while self.rval:
            self.update_face_variables()
            self.overlay(self.message, self.nose[0] - (self.message_width // 2), self.nose[1] - (self.message_height // 2))
            cv2.imshow(self.window, self.frame)
            if cv2.waitKey(20) == 27: # exit on ESC
                break
            self.rval, self.frame = self.vc.read()
            self.frame = cv2.flip(self.frame, 1)

            if self.nod:
                self.player_x, self.player_y = self.nose[:2]
                playsound.playsound(self.wing, False)
                return True # continue

        return False # Empty dict signifies the end of the program, other methods will check for this

    def game(self, continue_: bool) -> bool:

        if not continue_: # either rval was false or client escaped
            return continue_
        
        player_index = loop_iter = 0 # still unsure ab player_index and loop_iter
        player_vel_y = -9   # player's velocity along Y, default same as playerFlapped
        player_max_vel_y = 10   # max vel along Y, max descend speed
        player_acc_y = 1   # players downward accleration
        player_rot = 45   # player's rotation
        player_vel_rot = 3   # angular speed
        player_rot_threshold = 20   # rotation threshold
        player_flap_acc = -9   # players speed on flapping
        player_flapped = False # True when player flaps
        
        while self.rval:
            
            if cv2.waitKey(20) == 27: # exit on ESC
                break

            self.update_face_variables() # update the mark detector and the nod boolean
            if self.nod and self.player_y > -2 * self.player_height: # if the player nodded and is in the bounds of the screen
                player_vel_y = player_flap_acc # set the player y velocity equal to the flap acceleration (so we can start going in the other direction)
                player_flapped = True # set this boolean to True
                playsound.playsound(self.wing, False) # play the wing flapping noise

            if self.check_crash(self.nose[0], player_index): # check if we have crashed or not, if not, we continue
                return True

            self.display_pipes() # display the pipes and update score
            self.overlay(self.base, 0, self.base_offset) # draw the base after the pipes

            
            if (loop_iter + 1) % 3 == 0: # player index change
                player_index = next(self.player_cycle) # changes how the bird looks like while flying

            loop_iter = (loop_iter + 1) % 30

            # player rotation
            if player_rot > -90: # if the angle is greater than straight down
                player_rot -= player_vel_rot # decrease the angle
            
            # player movement
            if player_vel_y < player_max_vel_y and not player_flapped: # if the y-velocity has not yet peaked and the bird did not flap,
                player_vel_y += player_acc_y # increase the velocity in the y direction by the acceleration in the y direction

            if player_flapped: # if the player flapped
                player_flapped = False # reset the variable
                player_rot = 45 # set the player angle to look upward (flying)
            
            #self.player_height = self.player[player_index].shape[0]
            self.player_y += player_vel_y # increase the distance in the y direction by the velocity in the y direction
            self.display_score(self.forehead[0], self.forehead[1]) # show the score on the forehead
            
            visible_rot = player_rot_threshold # the visible rotation of the bird is set to the rotation threshold
            if player_rot <= player_rot_threshold: # if the player's rotation is less than the rotation threshold, 
                visible_rot = player_rot # set the visible rotation to the player rotation
            
            flappy_bird = self.rotate_image(self.player[player_index], visible_rot) # rotate the image by visible_rot degrees
            self.overlay(flappy_bird, self.nose[0] - (self.player_width // 2), self.player_y - (self.player_height // 2)) # draw the rotated bird
            cv2.imshow(self.window, self.frame) # show the frame
            self.rval, self.frame = self.vc.read() # get a new frame from the video capture object
            self.frame = cv2.flip(self.frame, 1) # flip the frame

        return False

    def show_gameover(self, continue_: bool) -> bool:
        if not continue_: # if the boolean value passed in indicated to not continue, the user has pressed ESC, thus quit and return False
            self.quit()
            return False

        while self.rval: # while a frame was read

            if cv2.waitKey(20) == 27: # exit on ESC
                break

            self.update_face_variables() # update the forehead, nose, and nod variables
            self.overlay(self.gameover, self.forehead[0] - (self.gameover_width // 2), self.forehead[1] - (self.gameover_height // 2)) # draw the gameover message on the forehead
            self.display_score(self.nose[0], self.nose[1]) # display the score on the nose
            cv2.imshow(self.window, self.frame) # show the frame
            self.rval, self.frame = self.vc.read() # get a new frame from the video capture object
            self.frame = cv2.flip(self.frame, 1) # flip the frame to correct it

            if self.nod: # if the player nodded, then we return True so the main game loop can return to the introduction
                return True
        
        # in the event that the while loop ends then either the user input the ESC key or self.rval was False, in each case, we end the game
        self.quit()
        return False

    def display_pipes(self) -> None: # displays the pipes on the screen, handles when pipes are off of the screen, and updates the score
        if datetime.now() - self.pipe_time >= self.pipe_time_buffer: # if the time since we've last placed a pipe is greater than the pipe buffer time
            self.pipes.append(self.get_random_pipe()) # we append a new random pipe to our pipes list
            self.pipe_time = datetime.now() # reset the pipe time
        
        #player_mid_pos = self.player_x + (self.player_width / 2) 
        for pipe in self.pipes: # for each pipe dictionary in our pipes list
            pipe_mid_pos = pipe['x'] + (self.pipe_width / 2) # the mid position of the pipe
            if pipe_mid_pos <= self.player_x < pipe_mid_pos + 4 and not pipe['passed']: # check if the player is within range of the middle of the pipes and hasn't already passed this pipe
                self.score += 1 # increase the score
                pipe['passed'] = True # set 'passed' to True, to avoid scoring twice on the same pipe
                playsound.playsound(self.point, False) # play point sound
            
            # draw the pipes
            self.overlay(self.pipe_top, pipe['x'], pipe['y'][0])
            self.overlay(self.pipe_bottom, pipe['x'], pipe['y'][1])

            pipe['x'] += self.pipe_vel_x # change the x-position of the pipes
        
        if self.pipes and self.pipes[0]['x'] < -self.pipe_width: # if the first pipe in the list has an x-value such that it doesn't show within the frame, 
            self.pipes.pop(0) # delete that pipe

    def overlay(self, image: np.ndarray, x: int, y: int) -> None:
        # The following prevents use from accessing pixels that are out of bounds
        y1, y2 = max(0, y), min(self.frame.shape[0], y + image.shape[0])
        x1, x2 = max(0, x), min(self.frame.shape[1], x + image.shape[1])
        y1o, y2o = max(0, -y), min(image.shape[0], self.frame.shape[0] - y)
        x1o, x2o = max(0, -x), min(image.shape[1], self.frame.shape[1] - x)

        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o: # exit if there is nothing to do
            return

        channels = self.frame.shape[2] # gets the channels of the alpha
        alpha = image[y1o:y2o, x1o:x2o, 3] / 255.0
        alpha_inv = 1.0 - alpha

        for c in range(channels):
            self.frame[y1:y2, x1:x2, c] = (alpha * image[y1o:y2o, x1o:x2o, c] + alpha_inv * self.frame[y1:y2, x1:x2, c])

    # The following function was written by Alex Rodrigues on StackOverflow
    # rotating an image by a specified angle using numpy
    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def update_player_motion(self) -> None: # updates the simple harmonic motion dictionary of the player
        
        if abs(self.player_shm_vals['val']) == 8: # oscillates the value of player_shm['val'] between 8 and -8
            self.player_shm_vals['dir'] *= -1 # switches the direction

        self.player_shm_vals['val'] += self.player_shm_vals['dir'] # val is moving in the direction dir
    
    def display_score(self, x: int, y: int) -> None: # display the score on the frame
        
        points = [int(i) for i in str(self.score)] # turn the score from an integer to a list 
        total_width = sum([self.number[i].shape[1] for i in points]) # get the total width of the score
        x_offset = x - (total_width // 2) # calculate where to begin the score image 
        y_offset = y - (self.number[0].shape[0] // 2) # calculate the y-offset to properly draw the score
        for i in points: # for every integer in the list points
            self.overlay(self.number[i], x_offset, y_offset) # overlay the image self.number[i] on self.frame at the given x and y offsets
            x_offset += self.number[i].shape[1] # move the x offset by the width of the image recently overlayed

    def check_crash(self, xpos: int, player_index: int) -> bool: # returns True if player collides with the base or pipes

        # if the bird has hit the ground
        if self.player_y + (self.player_width // 2) > self.base_offset:
            # TODO: implement pixel-perfect collisions using get_hitmask
            # play ending sounds then return True
            playsound.playsound(self.hit, False)
            playsound.playsound(self.die, False)
            return True

        for pipe in self.pipes: # loop through every pipe to see if the player has collided with it
            if pipe['x'] <= xpos + (self.player_width // 2) <= pipe['x'] + self.pipe_width: # if the bird has an x-value between or near a pipe, check it, otherwise we're wasting time
                if 0 <= self.player_y - (self.player_width // 2) <= min(pipe['y']) or max(pipe['y']) <= self.player_y + (self.player_width // 2) <= self.height: # check if the y-value is in the range of the pipes
                    # TODO: implement pixel-perfect collisions using get_hitmask
                    # play ending sounds then return True
                    playsound.playsound(self.hit, False)
                    playsound.playsound(self.die, False)
                    return True

        # the player collides with no pipes, is not colliding with the ground
        return False

    def play(self) -> None:
        # Repetitive playing loop, allows for a restart
        # after the player has crashed by clearing the variables
        while True: # The loop will continue forever unless ESC key is pressed
            intro = self.introduction() # draws the introductory message
            game = self.game(intro) # main game loop
            if not self.show_gameover(game): # self.show_gameover(game): True = replay, False = quit
                break
            self.reset_variables() # reset the dynamic variables

        print("Game Over.")

    '''
    TODO: use hitmask to do pixel-perfect collisions.
    def get_hitmask(self, image) -> list:
        mask = []
        for x in range(image.shape[1]):
            mask.append([])
            for y in range(image.shape[0]):
                alpha = image[y, x, 3] / 255.0
                mask[x].append(bool(alpha))
        return mask
    '''

    def quit(self) -> None:
        self.vc.release() # end the video capture
        cv2.destroyAllWindows() # close all of the opencv windows

if __name__ == '__main__':
    FlappyBird().play()