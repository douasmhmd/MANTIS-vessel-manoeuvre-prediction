@echo off
REM MANTIS — démarrage local sans Docker (3 fenêtres)
cd /d %~dp0
if exist .env for /f "usebackq tokens=1,2 delims==" %%a in (".env") do set %%a=%%b
start "MANTIS API"      cmd /k "uvicorn api.main:app --port 8000"
start "MANTIS Operator" cmd /k "streamlit run ui\operator.py --server.port 8501"
start "MANTIS Admin"    cmd /k "streamlit run ui\admin.py --server.port 8502"
start "MLflow UI"       cmd /k "mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000"
echo API http://127.0.0.1:8000/docs ^| Operateur http://localhost:8501 ^| Admin http://localhost:8502 ^| MLflow http://localhost:5000
