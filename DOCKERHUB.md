# sign-language-to-text

Fingerspelling translator for ASL and RSL. The camera recognizes hand shapes and
builds text; the reverse mode plays typed text back as an animated hand. There is
also a practice mode that shows a letter and checks that you repeat it.

Built with OpenCV, MediaPipe Hand Landmarker and NumPy.

Available alphabets:

- ASL (American Sign Language): signs to text, text to signs
- RSL (Russian Sign Language): signs to text, text to signs

## Examples

![Fig. 1: Example 1](https://mathematiclove.github.io/my-cv/content/projects/SIGN_TO_TEXT/EXAMPLE_1.png) ![Fig. 2: Example 2](https://mathematiclove.github.io/my-cv/content/projects/SIGN_TO_TEXT/EXAMPLE_3.png)

## Tags

- latest: built from the default branch
- sha-COMMIT: one tag per commit, using the full commit sha
- BRANCH: latest build of that branch
- vX.Y.Z: release tags

Each tag is a multi-arch manifest for linux/amd64 and linux/arm64.

## Pull

    docker pull USERNAME/sign-language-to-text

Replace USERNAME with the Docker Hub account this image is published under.

## Run the tests

This needs neither a camera nor a display, so it works on any host:

    docker run --rm USERNAME/sign-language-to-text python -m pytest

## Run the app

The app opens an OpenCV window and reads a webcam, so the container needs both
forwarded from the host. Docker can do that on Linux with X11:

    xhost +local:docker
    docker run --rm \
      -e DISPLAY=$DISPLAY \
      -e QT_X11_NO_MITSHM=1 \
      -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
      --device /dev/video0:/dev/video0 \
      USERNAME/sign-language-to-text

On macOS and Windows, Docker does not pass a webcam through to the container, so
use the image for tests only and run the app natively on the host.

Options:

    docker run --rm ... USERNAME/sign-language-to-text \
      python -m app.main --mode recognize --lang asl

    docker run --rm ... USERNAME/sign-language-to-text \
      python -m app.main --mode spell --lang rsl --text privet

    docker run --rm ... USERNAME/sign-language-to-text \
      python -m app.main --mode practice --lang rsl

Russian letters are typed in latin and transliterated, so "privet" becomes the
Cyrillic word.

## What is inside the image

Base is python:3.11-slim. On top of it:

- system packages libgl1, libglib2.0-0, libsm6, libxext6, libxrender1, needed by
  OpenCV, and fonts-dejavu-core, which is the font used to draw Cyrillic labels
- the Python dependencies from requirements-dev.txt: opencv-python, mediapipe,
  numpy, Pillow, pytest
- the MediaPipe hand_landmarker.task model, baked in at
  /app/models/hand_landmarker.task, so the container never downloads anything at
  run time
- the project sources: app, cli, tests, pytest.ini, under /app

Environment variables set in the image:

- SLT_FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf, the font used for
  on-screen text
- SLT_HAND_MODEL=/app/models/hand_landmarker.task, where the model is read from
- SLT_NO_DOWNLOAD=1, which forbids downloading the model at run time

Default command is: python -m app.main

## Source

https://github.com/MathematicLove/sign-language-to-text
