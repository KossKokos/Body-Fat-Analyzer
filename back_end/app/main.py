from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from config import logger
from services.model_service import ModelService
from services.prediction_service import PredictionService
from api.endpoints.predictions import router as prediction_router
from api.endpoints.health import router as health_router
from api.endpoints.feedback import router as feedback_router
from api.security import (
    APIKeyMiddleware, 
    SecurityHeadersMiddleware,
    RateLimitMiddleware
)


class Application():
    
    def _setup_logger(self):
        # Setup logging
        logger.setup_logging(
            log_level=settings.LOG_LEVEL,
            log_json=settings.LOG_JSON,
            log_file=settings.LOG_FILE if settings.LOG_TO_FILE else None
        )
        
        self.app_logger = logger.logger.with_context(app=settings.PROJECT_NAME)
        self.app_logger.info("Starting application", version=settings.PROJECT_VERSION)

    def _setup_middlewares(self):
        # Setup CORS
        self.app_logger.info("Setting up middlewares")

        self.application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=settings.ALLOW_METHODS,  
            allow_headers=settings.ALLOW_HEADERS,  
            expose_headers=settings.EXPOSE_HEADERS,
            max_age=settings.MAX_AGE,
        )

        # Add security middlewares (order matters)
        self.application.add_middleware(SecurityHeadersMiddleware)
        self.application.add_middleware(APIKeyMiddleware)

        if settings.ENVIRONMENT == "production":
            self.application.add_middleware(RateLimitMiddleware, calls_per_minute=settings.CALLS_PER_MINUTE)

        self.app_logger.info("Middlewares set up successfully")

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        # --- STARTUP ---
        self.app_logger.info("Loading ML models")

        model_service = ModelService()
        model_service.load_models()

        app.state.model_service = model_service
        app.state.prediction_service = PredictionService(model_service)

        self.app_logger.info("ML models loaded successfully")

        yield

        # --- SHUTDOWN ---
        self.app_logger.info("Shutting down application")
        
    def _create_application(self) -> FastAPI:

        self.app_logger.info("Application start up")
        # Create FastAPI app
        self.application = FastAPI(
            title=settings.PROJECT_NAME,
            version=settings.PROJECT_VERSION,
            openapi_url="/api/openapi.json",
            docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
            redoc_url="/redoc" if settings.DOCS else None,
            lifespan=self._lifespan,
        )

        self.app_logger.info("Application started successfully")
        return self.application
    
    def init_app(self):
        self._setup_logger()
        self._create_application()
        self._setup_middlewares()


application = Application()
application.init_app()
app = application.application

# Include routes 
app.include_router(router=prediction_router, prefix=settings.API_PREFIX)
app.include_router(router=feedback_router, prefix=settings.API_PREFIX)
app.include_router(router=health_router, prefix=settings.API_PREFIX)

def main():
    import os 
    import uvicorn
    
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    uvicorn.run(
        settings.APP_MAIN,
        host=settings.APP_HOST,
        port=int(settings.APP_PORT), 
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )

if __name__ == "__main__":
    main()