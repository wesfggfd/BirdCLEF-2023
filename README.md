**Dataset_V_1:https://huggingface.co/datasets/wesfggfd/BirdCLEF-2023**

**Pretrained_Model_V_1:https://huggingface.co/wesfggfd/Pretrained-Birds-Classifier**

- [wesfggf/BirdCLEF-2023/]
- [└── 📁 input/]
-   [├── 📁 birdclef-2023-v1/]             
-    [│   ├── 📁 train_audio/]
-    [│     └──  📄 train_metadata.csv]        
-    [│   └── 📁 test_soundscapes/]                
-    [│]
-    [├── 📁 pretrained_models/]             
-    [│   └── 📄 v1_bird_vocalization_classifier/]       
-    [│       └── 📁 assets/]       
-    [│       └── 📁 variables/]             






**V1 Best Score:0.61058**

**V2 Best Score:0.61600**

# BirdCLEF 2023 Audio Classification Pipeline

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)

This repository implements an audio classification pipeline for the [BirdCLEF 2023 competition](https://www.kaggle.com/competitions/birdclef-2023), using a pre-trained bird vocalization classifier from TensorFlow Hub. The solution processes 5-second audio frames, generates species predictions, and formats outputs for competition submission.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Pipeline Architecture](#pipeline-architecture)
- [Implementation Details](#implementation-details)
  - [1. Audio Preprocessing](#1-audio-preprocessing)
  - [2. Class Alignment](#2-class-alignment)
  - [3. Model Inference](#3-model-inference)
  - [4. Submission Generation](#4-submission-generation)
- [Usage](#usage)
- [Key Considerations](#key-considerations)
- [License](#license)

## Features
- **Audio Resampling**: Converts audio to 32kHz mono format
- **Frame-based Processing**: 5-second non-overlapping window analysis
- **Competition-Class Mapping**: Aligns model outputs with 264 target species
- **Probabilistic Predictions**: Softmax-normalized confidence scores
- **Kaggle Integration**: Direct compatibility with competition I/O formats

## Installation
```bash
# Core dependencies
pip install tensorflow tensorflow-hub tensorflow-io librosa pandas numpy

# For Jupyter integration (optional)
pip install ipython matplotlib
```
## Pipeline Architecture

- **graph TD**
    - **A[Raw Audio] --> B[Resample to 32kHz]**
    - **B --> C[Frame into 5s Windows]**
    - **C --> D[Model Inference]**
    - **D --> E[Class Probability Mapping]**
    - **E --> F[CSV Submission]**

## Implementation Details

- **1.Audio Preprocessing**

- **Resampling Function**
```python
def ensure_sample_rate(waveform: np.ndarray, 
                      original_rate: int, 
                      target_rate: int = 32000) -> tuple[int, np.ndarray]:
    """Resamples audio to target rate using TensorFlow I/O.
    
    Args:
        waveform: Raw audio samples
        original_rate: Input sample rate (Hz)
        target_rate: Output sample rate (32000 Hz for model)
    
    Returns:
        Tuple of (target_rate, resampled_audio)
    """
    if original_rate != target_rate:
        waveform = tfio.audio.resample(waveform, original_rate, target_rate)
    return target_rate, waveform.numpy()
```
- **Framing Function**
```python
def frame_audio(audio: np.ndarray, 
               window_sec: float = 5.0, 
               hop_sec: float = 5.0, 
               sample_rate: int = 32000) -> np.ndarray:
    """Segments audio into fixed-length windows.
    
    Args:
        audio: 1D array of audio samples
        window_sec: Window length in seconds
        hop_sec: Hop size between windows
        sample_rate: Samples per second
    
    Returns:
        Framed audio as 2D array [num_frames, frame_samples]
    """
    frame_samples = int(window_sec * sample_rate)
    hop_samples = int(hop_sec * sample_rate)
    return tf.signal.frame(audio, frame_samples, hop_samples, pad_end=True).numpy()
```

- **2.Class Alignment**

- **Class Mapping Strategy**
```python
# Load competition classes
train_metadata = pd.read_csv("/kaggle/input/birdclef-2023/train_metadata.csv")
competition_classes = sorted(train_metadata.primary_label.unique())  # 264 species

# Load model classes from label.csv
with open("label.csv") as f:
    model_classes = [row[0] for row in csv.reader(f)][1:]  # 1,000+ species

# Create index mapping
class_map = []
for bird in competition_classes:
    try:
        class_map.append(model_classes.index(bird))
    except ValueError:
        class_map.append(0)  # Default to background class
        print(f"Unmapped species: {bird}")
```

- **3.Model Inference**

- **Model Loading**
```python
model = hub.load("https://kaggle.com/models/google/bird-vocalization-classifier/versions/1")
```

- **Batch Inference Function**
```python
def process_audio_file(filename: str, 
                      submission_df: pd.DataFrame, 
                      max_seconds: int = 15) -> None:
    """Processes a single audio file and updates submission DataFrame.
    
    Args:
        filename: Path to .ogg audio file
        submission_df: Submission template DataFrame
        max_seconds: Maximum duration to process (competition requirement)
    """
    # Load and preprocess
    audio, orig_sr = librosa.load(filename)
    target_sr, resampled = ensure_sample_rate(audio, orig_sr)
    frames = frame_audio(resampled)
    
    # Process frames
    max_frames = max_seconds // 5
    batch_logits = model.infer_tf(frames[:max_frames])
    batch_probs = tf.nn.softmax(batch_logits).numpy()
    
    # Map to competition classes
    competition_probs = batch_probs[:, class_map]
    
    # Update submission DataFrame
    file_id = Path(filename).stem
    for i in range(competition_probs.shape[0]):
        row_id = f"{file_id}_{5*(i+1)}"
        submission_df.loc[submission_df.row_id == row_id, competition_classes] = competition_probs[i]
```

- **4.Submission Generation**

- **Main Execution Flow**
```python
if __name__ == "__main__":
    # Initialize submission template
    submission = pd.read_csv("/kaggle/input/birdclef-2023/sample_submission.csv")
    
    # Process all test files
    test_files = glob.glob("/kaggle/input/birdclef-2023/test_soundscapes/*.ogg")
    for file in tqdm(test_files, desc="Processing audio files"):
        process_audio_file(file, submission)
    
    # Save results
    submission.to_csv("submission.csv", index=False)
    print(f"Submission saved with {len(submission)} rows")
```

##Usage
- **1.Kaggle Environment Setup**
```python
# Verify TensorFlow Hub model
import tensorflow_hub as hub
print(hub.resolve("https://kaggle.com/models/google/bird-vocalization-classifier/versions/1"))
```

- **2.Full Pipeline Execution**
```bash
python main.py \
    --test_dir "/kaggle/input/birdclef-2023/test_soundscapes" \
    --output_file "submission.csv"
```

- **3.Output Validation**
```python
# Inspect submission
sub = pd.read_csv("submission.csv")
print(f"Columns: {sub.columns.tolist()}")
print(f"Row count: {len(sub)}")
print(f"Sample prediction:\n{sub.iloc[0]}")
```

##License

- **Apache License 2.0 - Compatible with Kaggle competition rules and TensorFlow Hub model terms.**
```txt
This comprehensive README provides technical depth while maintaining readability, with complete code integration and architecture diagrams. It serves both as documentation and an implementation guide.
```

 薛源宝宝快乐

<!-- 爱心设计 -->
<div align="center">
  
✨ **I LOVE U**   
❤️❤️❤️❤️❤️  
❤️        ❤️  
❤️        ❤️  
❤️  超级无敌可爱薛宝天天开心 YEAH ❤️  
❤️        ❤️  
❤️        ❤️  
❤️❤️❤️❤️❤️  

</div>

