#Start-Job -Name backend -ScriptBlock {
#    & C:\Users\hmaro\AppData\Roaming\Python\Scripts\uv.exe run uvicorn backend.src.api.main:app --reload --host 192.168.1.156 --port 8000 |
#        ForEach-Object { "[BACKEND] $_" }
#}
#
#Start-Job -Name frontend -ScriptBlock {
#    & C:\Users\hmaro\AppData\Roaming\Python\Scripts\uv.exe run streamlit run app_moved.py --server.port 8501 |
#        ForEach-Object { "[FRONTEND] $_" }
#}
#
#Write-Host "Backend + Frontend lancés. Logs combinés :`n"
#
#while ($true) {
#    Receive-Job -Name backend -Keep
#    Receive-Job -Name frontend -Keep
#    Start-Sleep -Milliseconds 2000
#}
Start-Process powershell.exe -ArgumentList '-NoExit', '-Command', 'uv run uvicorn backend.src.api.main:app --reload --host localhost --port 8000'
Start-Process powershell.exe -ArgumentList '-NoExit', '-Command', 'uv run streamlit run app.py --server.port 8501'
