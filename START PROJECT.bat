@echo off

start cmd /k "cd /d C:\Users\Administrator\Federated-Meta-Learning-AI-Assistant\backend && ..\venv\Scripts\activate && uvicorn main:app --reload --port 8000"

timeout /t 3 >nul

start cmd /k "cd /d C:\Users\Administrator\Federated-Meta-Learning-AI-Assistant\frontend && ..\venv\Scripts\activate && streamlit run app.py"

timeout /t 5 >nul

start http://localhost:8501