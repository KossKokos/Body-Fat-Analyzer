import uvicorn
from fastapi import FastAPI

from src.routes import router

app = FastAPI()
app.include_router(router, prefix='/api')

@app.get("/")
async def root():
    return {"message": "Hello world"}

async def main():  
    uvicorn.run('main:app', host='0.0.0.0', port=80, reload=True)    

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

a = [39, 'male', 74, 1.72, 180, 143, 62, 1.25, 1280, 'cardio', 26, 2.62, ]