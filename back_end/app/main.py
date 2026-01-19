# import uvicorn
# from fastapi import FastAPI

# from src.routes import router

# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings, logger
from app.api.v1.api import api_router
from app.services.model_service import ModelService


def create_application() -> FastAPI:
    # Setup logging
    logger.setup_logging(
        log_level=settings.LOG_LEVEL,
        log_json=settings.LOG_JSON,
        log_file=settings.LOG_FILE if settings.LOG_TO_FILE else None
    )
    
    app_logger = logger.logger.with_context(app="fat_percentage_predictor")
    app_logger.info("Starting application", version="1.0.0")
    
    # Create FastAPI app
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs" if settings.DOCS else None,
        redoc_url="/redoc" if settings.DOCS else None,
    )
    
    # Setup CORS
    if settings.BACKEND_CORS_ORIGINS:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Include routers
    application.include_router(api_router, prefix=settings.API_V1_STR)
    
    # Startup event - load models
    @application.on_event("startup")
    async def startup_event():
        app_logger.info("Loading ML models")
        try:
            ModelService.load_models()
            app_logger.info("ML models loaded successfully")
        except Exception as e:
            app_logger.exception("Failed to load ML models", error=str(e))
            raise
    
    # Shutdown event
    @application.on_event("shutdown")
    async def shutdown_event():
        app_logger.info("Shutting down application")
    
    # Health check endpoint
    @application.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    
    return application


app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )

# app = FastAPI()
# app.include_router(router, prefix='/api')

# @app.get("/")
# async def root():
#     return {"message": "Hello world"}

# async def main():  
#     uvicorn.run('main:app', host='0.0.0.0', port=80, reload=True)    

# if __name__ == '__main__':
#     import asyncio
#     asyncio.run(main())

# a = [39, 'male', 74, 1.72, 180, 143, 62, 1.25, 1280, 'cardio', 26, 2.62, ]