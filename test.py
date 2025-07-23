from functions.andorsdk.andorsdk import *

if __name__ == "__main__":
    camera = AndorSDK2Handler(driver_path="C:/helmi/研究/raman-app/drivers")
    camera.initialize_camera()