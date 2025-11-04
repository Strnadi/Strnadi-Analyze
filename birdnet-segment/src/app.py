from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile, wave, shutil
import io, zipfile, os

from segment import get_yellowhammers

app = FastAPI()

@app.post("/process")
async def process(file: UploadFile):
    # Save uploaded WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)


    segments = get_yellowhammers(tmp_path)
    

    return JSONResponse(content={"segments": segments})


