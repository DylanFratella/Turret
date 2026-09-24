# Turret
Ensure your python version is 3.11.9 for this to work.
Download CMAKE for Windows x64 titled “cmake-4.3.0-windows-x86_64.msi” at https://cmake.org/download/ 
Make a new folder on your computer and put both python scripts inside of it.
Download 7-zip for windows x64 at “https://www.7-zip.org/download.html”
Download lib-usb at https://github.com/libusb/libusb/releases. Once downloaded, open the 7-zip app then find libusb-1.0.29.7z on it. Open the file, then find the MinGW64 file, then dll. Then find the file “libusb-1.0.dll” and put it inside the folder with the python script.
Then run the command “python -m pip install mediapipe opencv-python pyusb tensorflow scikit-learn pillow” inside the VS-Code terminal and wait for everything to be installed.
Then make 2 folders inside the folder, one to store people’s images and another to store sound effects. 
Change the directory inside the python scripts to include your personal directory, for example my folder is “C:\Users\csstudent\turretStuff”, but yours will be different.
