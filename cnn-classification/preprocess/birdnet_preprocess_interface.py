import requests
import os
from typing import *
import logging
import time

logger = logging.getLogger("birdnet-interface")

def get_yellowhammer_intervals(wav_bytes :bytes) -> List[Tuple[float, float]]:
    

    url = os.environ.get("BIRDNET_URL", "http://localhost:8000/process")

    logger.info(f"sending request to birdnet at {url}")
    s_t = time.time()
    response = requests.post(url, files={"file": ("audio.wav", wav_bytes, "audio/wav")})
    d_t = time.time() - s_t
    logger.info(f"request took {d_t} seconds")

    response.raise_for_status()


    json = response.json()
    logger.debug(f"birdnet responded with segments: {json}")
    return json['segments']




# if __name__ == '__main__':
#     with open('.tstdata/F002413.wav', 'rb') as f:
#         get_yellowhammer_intervals()
