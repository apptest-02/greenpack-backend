import threading  
import webbrowser  
import time  
import os  
import sys  
from fastapi.staticfiles import StaticFiles  
from app.main import app  
  
if getattr(sys, 'frozen', False):  
    base_path = sys._MEIPASS  
else:  
    base_path = os.path.dirname(os.path.abspath(__file__))  
  
frontend_path = os.path.join(base_path, "frontend")  
if os.path.exists(frontend_path):  
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")  
    print(f"Frontend served from: {frontend_path}")  
  
def run():  
    import uvicorn  
    uvicorn.run(app, host="127.0.0.1", port=8000)  
  
if __name__ == "__main__":  
    threading.Thread(target=run, daemon=True).start()  
    time.sleep(2)  
    webbrowser.open("http://127.0.0.1:8000")  
    print("Greenpack Pro running at http://127.0.0.1:8000")  
    print("Press Enter to exit...")  
    input() 
