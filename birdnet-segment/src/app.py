from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile, shutil
import logging


from segment import get_yellowhammers

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("app")
app = FastAPI()

@app.post("/process")
async def process(file: UploadFile):
    # Save uploaded WAV
    logger.info("got file")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)


    segments = get_yellowhammers(tmp_path)
    
    logger.info("file processed, responding")
    logger.debug(f"segments: {segments}")
    return JSONResponse(content={"segments": segments})


