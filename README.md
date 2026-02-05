# Buildathon 2026 - Broadcast AI Alert System (REFER BRANCH Agent-Upgrade1.0 for the latest iteration)

## Overview

This project is an intelligent broadcast alert system that analyzes messages in multiple Indian languages (Tamil, Kannada, Hindi, Punjabi, Telugu, Malayalam, Bengali, Assamese, Gujarati, and English) to detect and prioritize alerts across different regional areas in India.

The system combines:
- **Machine Learning Classification**: Character-level TF-IDF features with Random Forest classifier for language and region detection
- **AI-Powered Analysis**: Integration with Ollama (Mistral/Qwen2.5 models) for intelligent alert analysis
- **Interactive Visualization**: Real-time map-based visualization of alert distribution across regions
- **Multi-threaded Processing**: Efficient queue-based message processing and dispatch

## Project Structure

```
Buildathon_2026/
├── Demo.py                      # Main application with GUI and agent-based alert system
├── Demo_development.py          # Development version with matplotlib visualization
├── model.py                     # ML model training script
├── atsc3_comparative_data.csv   # Comparative data for ATSC3 analysis
├── language_model_rf.pkl        # Trained Random Forest classifier (generated)
├── vectorizer_rf.pkl            # TF-IDF vectorizer (generated)
└── README.md                    # This file
```

## Features

### 1. Machine Learning Model (`model.py`)
- **Training**: Character-level TF-IDF vectorization + Random Forest classification
- **Multi-language Support**: Handles 10 Indian languages with regional mapping
- **Optimization**: Grid search for hyperparameter tuning
- **Serialization**: Saves trained models using joblib for production use

### 2. Main Application (`Demo.py`)
- **Agent-Based Architecture**: Queue-based message processing with configurable dispatch intervals
- **Ollama Integration**: Real-time analysis using local LLM (Mistral/Qwen2.5)
- **GUI Interface**: Tkinter-based interface for message input and analysis
- **Alert Prioritization**: Automatic priority classification (P1-P4 levels)
- **Multi-threading**: Async processing for smooth user experience

### 3. Development Version (`Demo_development.py`)
- **Interactive Visualization**: Matplotlib-based map showing alert distribution
- **Region-wise Tracking**: Real-time updates of alerts per Indian region
- **Fallback Logic**: Keyword-based classification if ML model unavailable
- **Testing Environment**: Ideal for prototyping and visualization

## Setup & Installation

### Requirements
- Python 3.8+
- Required Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `requests`, `joblib`
- **Ollama**: Local LLM server running on `http://localhost:11434`

### Installation Steps

1. **Clone/Navigate to project directory**
```bash
cd Buildathon_2026
```

2. **Install dependencies**
```bash
pip install numpy pandas scikit-learn matplotlib requests joblib
```

3. **Setup Ollama (Required for Demo.py)**
   - Download and install [Ollama](https://ollama.ai)
   - Pull the Mistral model:
   ```bash
   ollama pull mistral
   # OR for lighter model:
   ollama pull qwen2.5:1.5b
   ```
   - Start Ollama service (runs on `http://localhost:11434` by default)

4. **Train the ML Model**
```bash
python model.py
```
This generates:
- `language_model_rf.pkl` - Trained classifier
- `vectorizer_rf.pkl` - Feature vectorizer

## Usage

### Running the Main Application
```bash
python Demo.py
```
This launches a Tkinter GUI where you can:
- Input messages in any supported language
- View AI-powered analysis and alert classification
- Monitor real-time alert dispatch queue

### Running the Development/Visualization Version
```bash
python Demo_development.py
```
This shows:
- Interactive map of Indian regions
- Visual distribution of alerts
- Simulated message processing with ML predictions

## Key Components

### Alert Prioritization System
- **P1**: Critical/Emergency alerts
- **P2**: High-priority alerts
- **P3**: Medium-priority alerts
- **P4**: Low-priority informational alerts

### Supported Languages & Regions
| Language | Region |
|----------|--------|
| Tamil | Tamil Nadu |
| Kannada | Karnataka |
| Telugu | Andhra Pradesh |
| Malayalam | Kerala |
| Hindi | Delhi |
| Punjabi | Punjab |
| Bengali | West Bengal |
| Assamese | Assam |
| Gujarati | Gujarat |
| English | All Regions |

### LLM Integration
The system uses Ollama's Mistral or Qwen2.5 model to:
- Analyze message content and context
- Determine alert priority (P1-P4)
- Extract regional information
- Validate language detection

## Performance Metrics

The trained Random Forest model achieves:
- High accuracy in language detection across 10 languages
- Fast inference time suitable for real-time processing
- Robust character-level feature extraction for regional variations

## Technical Details

### Message Processing Pipeline
1. User input message
2. Queue-based storage (AGENT_INPUT_QUEUE)
3. ML-based language/region classification
4. LLM analysis via Ollama for priority determination
5. Alert dispatch based on priority and region
6. Visualization/Logging of results

### Agent-Based System
- **Dispatch Interval**: Configurable queue processing interval (default: 5 seconds)
- **Threading**: Separate thread for agent processing to prevent UI blocking
- **Queue Management**: Deque-based FIFO processing for message handling

## Configuration

Edit the following in `Demo.py` for customization:
```python
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"  # or "mistral"
DISPATCH_INTERVAL = 5  # seconds between queue processing
```

## Troubleshooting

### Model Files Not Found
- Run `python model.py` to generate model files
- Ensure `Datasets` folder exists with language subdirectories

### Ollama Connection Error
- Verify Ollama service is running: `ollama serve`
- Check URL: `http://localhost:11434/api/generate`
- Ensure model is pulled: `ollama list`

### GUI Issues
- Use Python 3.8+ for Tkinter compatibility
- On Linux, may need: `sudo apt-get install python3-tk`

## Future Enhancements

- Multi-model ensemble for improved accuracy
- Real-time streaming alert notifications
- Advanced analytics and reporting

## License

This project was developed for Buildathon 2026.

