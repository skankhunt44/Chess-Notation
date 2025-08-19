"""
training_pipeline.py - Handle AI model training workflow
========================================================
Manages preprocessing and training commands for the chess square classifier.
Commands are run from the chess_cnn directory with correct paths.
"""

import subprocess
import threading
import json
import os
import glob
from pathlib import Path
from fastapi import Response
from fastapi.responses import JSONResponse, StreamingResponse

class TrainingPipeline:
    def __init__(self, chess_cnn_dir="chess_cnn"):
        self.chess_cnn_dir = Path(chess_cnn_dir)
        self.data_dir = Path("data")
        self.assets_dir = Path("assets")
        
        # Ensure directories exist
        (self.data_dir / "crops_my_cam").mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def get_current_session(self):
        """Get the most recent session directory"""
        session_pattern = self.data_dir / "session_*"
        session_dirs = list(session_pattern.parent.glob(session_pattern.name))
        
        if not session_dirs:
            return None
        
        # Sort by modification time, get most recent
        session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return session_dirs[0]

    def stream_command_output(self, cmd, cwd=None):
        """Stream command output as JSON lines"""
        def generate():
            try:
                cmd_str = " ".join(cmd)
                yield f'{json.dumps({"log": f"Running: {cmd_str}", "status": "Starting command..."})}\n'
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    cwd=cwd
                )
                
                # Stream output line by line
                for line in iter(process.stdout.readline, ''):
                    if line.strip():
                        yield f'{json.dumps({"log": line.strip()})}\n'
                
                # Wait for process to complete
                return_code = process.wait()
                
                if return_code == 0:
                    yield f'{json.dumps({"status": "Command completed successfully", "complete": True})}\n'
                else:
                    yield f'{json.dumps({"status": f"Command failed with exit code {return_code}", "error": True})}\n'
                    
            except FileNotFoundError as e:
                yield f'{json.dumps({"status": f"Command not found: {str(e)}", "error": True})}\n'
            except Exception as e:
                yield f'{json.dumps({"status": f"Error: {str(e)}", "error": True})}\n'
        
        return StreamingResponse(generate(), media_type='application/json')

    def preprocess_data(self):
        """Preprocess training data from current session"""
        try:
            current_session = self.get_current_session()
            if not current_session:
                return JSONResponse(
                    {"error": "No session data found. Collect some training samples first."}, 
                    status_code=400
                )
            
            samples_dir = current_session / "samples"
            if not samples_dir.exists() or not any(samples_dir.iterdir()):
                return JSONResponse(
                    {"error": f"No samples found in {samples_dir}"}, 
                    status_code=400
                )
            
            # Build the preprocessing command (run from chess_cnn directory)
            src_dir = f"../data/{current_session.name}/samples"  # Relative to chess_cnn
            dst_dir = "../data/crops_my_cam"  # Relative to chess_cnn
            
            cmd = [
                'python', 'chess_square_classifier2.py', 'preprocess',
                '--src-dir', src_dir,
                '--dst-dir', dst_dir,
                '--workers', '2'
            ]
            
            print(f"Preprocessing from {current_session}")
            print(f"Running command in {self.chess_cnn_dir}: {' '.join(cmd)}")
            
            return self.stream_command_output(cmd, cwd=str(self.chess_cnn_dir))
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    def train_model(self):
        """Train the AI model with preprocessed data"""
        try:
            # Check if preprocessed data exists
            crops_dir = self.data_dir / "crops_my_cam"
            if not crops_dir.exists() or not any(crops_dir.iterdir()):
                return JSONResponse({
                    "error": "No preprocessed data found. Run preprocessing first."
                }, status_code=400)
            
            # Check if we have the model to resume from
            model_path = self.assets_dir / "model.pt"
            if not model_path.exists():
                return JSONResponse({
                    "error": f"Base model not found at {model_path}. Please ensure model.pt exists."
                }, status_code=400)
            
            # Build the training command (run from chess_cnn directory)
            cmd = [
                'python', 'chess_square_classifier2.py', 'train',
                '--data-dir', '../data/crops_my_cam',  # Relative to chess_cnn
                '--preprocessed',
                '--epochs', '6',
                '--batch-size', '256',
                '--lr', '2e-4',
                '--resume', '../assets/model.pt',  # Relative to chess_cnn
                '--model-out', '../assets/model.pt'  # Relative to chess_cnn
            ]
            
            print(f"Running training command in {self.chess_cnn_dir}: {' '.join(cmd)}")
            
            return self.stream_command_output(cmd, cwd=str(self.chess_cnn_dir))
            
        except Exception as e:
            print(f"Training error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    def get_status(self):
        """Get training pipeline status"""
        try:
            current_session = self.get_current_session()
            crops_dir = self.data_dir / "crops_my_cam"
            model_path = self.assets_dir / "model.pt"
            
            status = {
                "has_session_data": current_session is not None,
                "session_path": str(current_session) if current_session else None,
                "has_preprocessed_data": crops_dir.exists() and any(crops_dir.iterdir()),
                "has_model": model_path.exists(),
                "can_preprocess": current_session is not None,
                "can_train": crops_dir.exists() and any(crops_dir.iterdir()) and model_path.exists()
            }
            
            # Count samples in current session
            if current_session:
                samples_dir = current_session / "samples"
                if samples_dir.exists():
                    sample_files = [f for f in samples_dir.iterdir() 
                                  if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
                    label_files = [f for f in samples_dir.iterdir() 
                                 if f.name.startswith('labels_') and f.suffix == '.json']
                    status["sample_count"] = len(sample_files)
                    status["labeled_count"] = len(label_files)
            
            # Count preprocessed crops
            if status["has_preprocessed_data"]:
                try:
                    crop_count = sum(len(list(subdir.iterdir())) 
                                   for subdir in crops_dir.iterdir() 
                                   if subdir.is_dir())
                    status["crop_count"] = crop_count
                except:
                    status["crop_count"] = 0
            
            return JSONResponse(status)
            
        except Exception as e:
            print(f"Status check error: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    def cleanup_old_crops(self):
        """Clean up old preprocessed crops to start fresh"""
        try:
            crops_dir = self.data_dir / "crops_my_cam"
            if crops_dir.exists():
                import shutil
                shutil.rmtree(crops_dir)
                crops_dir.mkdir(parents=True, exist_ok=True)
                return JSONResponse({"message": "Old preprocessed data cleaned up"})
            return JSONResponse({"message": "No old data to clean"})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

# Global instance
training_pipeline = TrainingPipeline()