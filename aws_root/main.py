from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
import json
app = FastAPI()
origins = ['*']
app.add_middleware(CORSMiddleware,
                   allow_origins = origins,
                   allow_credentials = True,
                   allow_methods = origins,
                   allow_headers  = origins
                   )
handler = Mangum(app)

@app.get("/")
async def read_root():
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda, This is our first deployed  APP!')
    }

@app.get("/home")
async def read_home():
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Home Page!')
    }
