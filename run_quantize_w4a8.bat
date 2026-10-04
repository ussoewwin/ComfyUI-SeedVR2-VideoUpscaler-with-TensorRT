@echo off
setlocal
echo ========================================================
echo Starting SeedVR2 7B W4A8 Quantization (ConvRot 256)
echo ========================================================

set TARGET=D:\USERFILES\ComfyUI\ComfyUI\models\SEEDVR2\seedvr2_7b_w4a8.safetensors
if exist "%TARGET%" (
    echo Existing target file found: %TARGET%
    attrib -r "%TARGET%"
    del /f /q "%TARGET%"
    if exist "%TARGET%" (
        echo WARNING: Failed to delete %TARGET%
    ) else (
        echo Successfully removed old %TARGET%
    )
)

"D:\USERFILES\ComfyUI\python_embeded\python.exe" "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT\tools\quantize_seedvr2_w4a8.py" --input "D:\USERFILES\ComfyUI\ComfyUI\models\SEEDVR2\seedvr2_7b_fp16.safetensors" --output "%TARGET%" --reference "D:\USERFILES\ComfyUI\ComfyUI\models\SEEDVR2\seedvr2_7b_nvfp4.safetensors" --group-size 16 --convrot-groupsize 256 --device cuda
set EXIT_CODE=%ERRORLEVEL%
echo ========================================================
echo Quantization finished with exit code %EXIT_CODE%
echo ========================================================
exit /b %EXIT_CODE%
