import requests
import os


def get_yellowhammer_intervals(wav_bytes :bytes) -> list[tuple[float, float]]:
    

    url = os.environ.get("BIRDNET_URL", "http://localhost:8000/process")

    response = requests.post(url, files={"file": ("audio.wav", wav_bytes, "audio/wav")})

    response.raise_for_status()

    json = response.json()
    # print(json['segments'])
    return json['segments']




# if __name__ == '__main__':
#     with open('.tstdata/F002413.wav', 'rb') as f:
#         get_yellowhammer_intervals()
