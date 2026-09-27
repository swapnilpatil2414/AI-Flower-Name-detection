@echo off
echo ======================================================================
echo Opening AI Annotated Flower Detection Video in Default Media Player...
echo ======================================================================
if exist "output_videos\annotated_flower_detection.mp4" (
    start "" "output_videos\annotated_flower_detection.mp4"
) else (
    echo Annotated video not found, generating now...
    python detect_flower.py
    start "" "output_videos\annotated_flower_detection.mp4"
)

