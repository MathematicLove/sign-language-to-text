# ASL, RSL to text (text to ASL, RSL)

Fingerspelling translator for ASL and RSL. The camera recognizes hand shapes and
builds text; the reverse mode plays typed text back as an animated hand. There is
also a practice mode that shows a letter and checks that you repeat it.

Built with OpenCV, MediaPipe Hand Landmarker and NumPy.

Available text:

- ASL (Americal Sign Language) / Sign to text, text to sing
- РЖЯ (Русский Язык Жестов) / Жесты в текст, текст в жесты

## Examples

<p>
  <img src="https://mathematiclove.github.io/my-cv/content/projects/SIGN_TO_TEXT/EXAMPLE_1.png" alt="Fig. 1: Example 1" width="45%">
  <img src="https://mathematiclove.github.io/my-cv/content/projects/SIGN_TO_TEXT/EXAMPLE_3.png" alt="Fig. 2: Example 2" width="45%">
</p>

## Install

    conda env create -f environment.yml
    conda activate slt

Or with pip:

    pip install -r requirements.txt

## Run

    python -m app.main

Options:

    python -m app.main --mode recognize --lang asl
    python -m app.main --mode spell --lang rsl --text privet
    python -m app.main --mode practice --lang rsl
    python -m app.main --camera 1 --width 1180 --height 760

Russian letters are typed in latin and transliterated: privet becomes the RSL
word.

## Tests

    python -m pytest

## Docker

A prebuilt image is published on Docker Hub on every commit, see DOCKERHUB.md:

    docker pull flugmaschine/sign-language-to-text

To build and run locally, tests run anywhere:

    docker compose run --rm tests

The app needs a camera and a window, which Docker only forwards on Linux with
X11:

    xhost +local:docker
    docker compose up app

**by Salimli Ayzek (Салимли Айзек): https://mathematiclove.github.io**