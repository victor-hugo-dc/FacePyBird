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

## Future Improvements
- [ ] Fix bugs in pitch angle estimation, sometimes theres a double jump because of it.
- [ ] Falling animation for once the bird has crashed.
- [ ] Make the base move continuously.

## Resources
[FlapPyBird, a Flappy Bird implementation in Python](https://github.com/sourabhv/FlapPyBird)\
[Head Pose Estimation using OpenCV](https://github.com/jerryhouuu/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV)