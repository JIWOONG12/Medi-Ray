from fastapi import FastAPI
from app.routers import dicom_upload_con_test, vit_cam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback


app = FastAPI()

# 허용할 출처 설정
origins = [
    "http://localhost:8089",  # 클라이언트가 실행되는 출처 추가
    "https://localhost:8443", # 필요 시 https 출처도 추가
]

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 허용할 출처 목록
    allow_credentials=True,
    allow_methods=["*"],  # 허용할 HTTP 메서드 (GET, POST 등)
    allow_headers=["*"],  # 허용할 HTTP 헤더
    expose_headers=["Access-Control-Allow-Origin"],  # CORS 관련 헤더 노출
)

# 라우터 등록
app.include_router(dicom_upload_con_test.router, prefix="/dicom", tags=["Medical Analysis"])
app.include_router(vit_cam.router, prefix="/dicom", tags=["Grad Cam"])

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    # 콘솔에 에러 로그 출력
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"message": f"Internal Server Error: {str(exc)}"},
    )

# uvicorn으로 이 모듈을 직접 실행할 때 서버를 구동하기 위한 코드
if __name__ == "__main__":
    import uvicorn
    # FastAPI 애플리케이션을 uvicorn으로 실행
    uvicorn.run(app, host="192.168.0.2", port=8000, ssl_certfile="cert.pem", ssl_keyfile="key.pem")