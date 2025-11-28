# Setup
## Hardware
- Make sure camera module is connect to the pi
- Open the terminal and run `cam --list` to see if the camera is recognized
- Run the command `rpicam-hello` to get a preview window of the camera feed

## Software
- Project assumes you are working from Raspberry Pi OS
- Install the following packages
    - `sudo apt install python3-picamera2`
    - `sudo apt install python3-opencv`
- Add the ultralytics module in the following way
    - Enter the project directory
    - Run `pip -m venv .venv --system-site-packages`
    - then enter the virtual environment with `source .venv/bin/activate`
    - Next run `pip install ultralytics`

# Running the program
- after setting up camera and getting all dependencies, run the following command after entering the virtual environment `python3 main.py`
- a window will pop up showing a video feed from the camera
    - when a face is detected, an image of a cat head is placed in front of the face
- to close the program simply press any key on the keyboard
