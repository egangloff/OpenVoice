from openvoice.api import BaseSpeakerTTS, ToneColorConverter
import torch, os
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
en_ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'

tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')

converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

@app.post("/tts")
async def tts_endpoint(text: str):
    src_path = "tmp.wav"
    out_path = "output.wav"
    tts.tts(text, src_path, speaker='default', language='English')
    os.rename(src_path, out_path)
    return FileResponse(out_path, media_type="audio/wav")
