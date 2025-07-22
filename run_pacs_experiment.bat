@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM Define variables
set "DATA_PATH=C:\Internship\Contrastive Learning\Datasets\PACS"
set "BATCH_SIZE=16"
set "EPOCHS=200"
set "EVAL_EPOCHS=100"
set "ARCH=resnet50"
set "OUT_DIM=128"

@REM REM Create directories for experiment results
@REM if not exist "experiment_results" mkdir "experiment_results"
@REM if not exist "evaluation_results" mkdir "evaluation_results"

@REM REM Run experiments directly without function calls
@REM echo ===== Running experiment: Art_Cartoon_Sketch-to-Photo =====
@REM python run.py -data "%DATA_PATH%" --dataset-name pacs --source-domains Art Cartoon Sketch --target-domain Photo --experiment-name Art_Cartoon_Sketch-to-Photo --arch %ARCH% --batch-size %BATCH_SIZE% --epochs %EPOCHS% --out_dim %OUT_DIM% --fp16-precision
@REM if errorlevel 1 (
@REM     echo Error during training for Art_Cartoon_Sketch-to-Photo
@REM     exit /b 1
@REM )
@REM python eval_pacs.py --epochs "%EVAL_EPOCHS%" -data "%DATA_PATH%" --target-domain "Photo" --checkpoint "experiment_results\Art_Cartoon_Sketch-to-Photo\model.pth.tar" --arch "%ARCH%" --batch-size "%BATCH_SIZE%"

@REM echo ===== Running experiment: Photo_Cartoon_Sketch-to-Art =====
@REM python run.py -data "%DATA_PATH%" --dataset-name pacs --source-domains Photo Cartoon Sketch --target-domain Art --experiment-name Photo_Cartoon_Sketch-to-Art --arch %ARCH% --batch-size %BATCH_SIZE% --epochs %EPOCHS% --out_dim %OUT_DIM% --fp16-precision
@REM if errorlevel 1 (
@REM     echo Error during training for Photo_Cartoon_Sketch-to-Art
@REM     exit /b 1
@REM )
@REM python eval_pacs.py --epochs "%EVAL_EPOCHS%" -data "%DATA_PATH%" --target-domain "Art" --checkpoint "experiment_results\Photo_Cartoon_Sketch-to-Art\model.pth.tar" --arch "%ARCH%" --batch-size "%BATCH_SIZE%"

@REM echo ===== Running experiment: Photo_Art_Sketch-to-Cartoon =====
@REM python run.py -data "%DATA_PATH%" --dataset-name pacs --source-domains Photo Art Sketch --target-domain Cartoon --experiment-name Photo_Art_Sketch-to-Cartoon --arch %ARCH% --batch-size %BATCH_SIZE% --epochs %EPOCHS% --out_dim %OUT_DIM% --fp16-precision
@REM if errorlevel 1 (
@REM     echo Error during training for Photo_Art_Sketch-to-Cartoon
@REM     exit /b 1
@REM )
@REM python eval_pacs.py --epochs "%EVAL_EPOCHS%" -data "%DATA_PATH%" --target-domain "Cartoon" --checkpoint "experiment_results\Photo_Art_Sketch-to-Cartoon\model.pth.tar" --arch "%ARCH%" --batch-size "%BATCH_SIZE%"

echo ===== Running experiment: Photo_Art_Cartoon-to-Sketch =====
python run.py -data "%DATA_PATH%" --dataset-name pacs --source-domains Photo Art Cartoon --target-domain Sketch --experiment-name Photo_Art_Cartoon-to-Sketch --arch %ARCH% --batch-size %BATCH_SIZE% --epochs %EPOCHS% --out_dim %OUT_DIM% --fp16-precision
if errorlevel 1 (
    echo Error during training for Photo_Art_Cartoon-to-Sketch
    exit /b 1
)
python eval_pacs.py --epochs "%EVAL_EPOCHS%" -data "%DATA_PATH%" --target-domain "Sketch" --checkpoint "experiment_results\Photo_Art_Cartoon-to-Sketch\model.pth.tar" --arch "%ARCH%" --batch-size "%BATCH_SIZE%"

echo All experiments completed!
ENDLOCAL
pause
