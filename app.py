from boss_env import load_aws_secrets
load_aws_secrets()
from fastapi import FastAPI, Request
import uvicorn
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from pydantic import ValidationError
import traceback
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, date as dt_date
from contextlib import asynccontextmanager
from datetime import datetime
import pytz
from typing import List


from MEMORY_SYSTEM.DATABASE.CONNECT.connect import db_manager
from MEMORY_SYSTEM.EXTRACTOR.LAYER_2_Postgres.create_layer_2_tables import ensure_layer_2_table_exists
from model import bedrock_llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await ensure_layer_2_table_exists()
    except Exception as e:
        raise

    yield 


    try:
        print("Completed")
    except Exception as e:
        raise



# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)



FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "https://orbitaim-lime.vercel.app",
    "http://127.0.0.1:5050",
    "https://orbit.orbitaim.io"
]



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.get('/')
def home():
    return "Hello, reniforcemnet learnings"


@app.post('/model')
async def newsreports(request: Request):
    try:
        data = await request.json()
        user_id = data.get("user_id",None)
        system_prompt = data.get("system_prompt",None)
        user_prompt = data.get("user_prompt", None)  
        context = None
        result = await bedrock_llm(user_id, system_prompt,user_prompt, context)
        return result
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": tb.splitlines()
            }
        )



if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=6929, reload=True)
