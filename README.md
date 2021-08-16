# FlapPy Bird with OpenCV
Flappy Bird implementation in Python that replaces arrow key inputs with head nods.
![Example](example.gif)

## Usage
To run this project,
1. Install Python 3.7 or above from [here](https://www.python.org/download/releases/)
2. Clone the repository:
    ```
    git clone https://github.com/victor-hugo-dc/FacePyBird.git
    ```
    or download as a zip file and extract.
3. To install all dependencies, run:
    ```
    pip install -r requirements.txt
    ```
4. In the root directory, run:
    ```
    python3 main.py
    ```
5. Use <kbd>Esc</kbd> to close the game.

## Change Log
#### 2021-08-16
- Made the class `FacePyBird` modular, so it can take in frames from external OpenCV captures.
#### 2021-04-10
- Implemented high score functionality with external file.
- High score message appears with correct medal.
#### 2021-08-09
- Corrected bug in pitch angle estimation.
- Added falling animation for once the bird has crashed.
- Implemented a continuously moving base.

## Resources
[FlapPyBird, a Flappy Bird implementation in Python](https://github.com/sourabhv/FlapPyBird)\
[Head Pose Estimation using OpenCV](https://github.com/jerryhouuu/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV)