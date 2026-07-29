SMART VISION - FIRST START

1. Extract this folder to a short path, for example:
   C:\MLProject\smart_vision_final_project_laptop_cpu

2. Double-click:
   install_windows.bat

3. The virtual environment will be created here:
   %USERPROFILE%\venvs\ml-yolo-cpu

4. After installation, double-click:
   check_environment.bat

5. Copy model files and datasets to the paths configured in src\config.py.

6. Double-click:
   check_project.bat

7. Start Jupyter with:
   start_notebook.bat

8. In Jupyter select this kernel:
   Python (ML YOLO CPU)

Important:
- This laptop has no NVIDIA CUDA GPU. CUDA available: False is expected.
- Do not place the project inside the virtual environment folder.
- Keep the project path short to avoid Windows long-path errors.
