# 🚦 AI-Powered Adaptive Traffic Light System

An intelligent traffic management system that uses computer vision to dynamically adjust traffic light timings based on real-time vehicle detection and counting.

## 📋 Overview

This project implements an adaptive traffic light control system that analyzes video footage to detect vehicles, track their movement, and automatically adjust green light duration based on traffic density. The system aims to reduce wait times and improve traffic flow efficiency.

## ✨ Features

- **Real-time Vehicle Detection**: Uses background subtraction (MOG2) to identify moving vehicles
- **Intelligent Tracking**: Assigns unique IDs to vehicles and tracks their movement across frames
- **Adaptive Timing**: Dynamically calculates green light duration based on waiting vehicle count
- **Line Crossing Detection**: Accurately counts vehicles crossing a designated line
- **Visual Feedback**: Live display showing:
  - Current traffic light state (Red/Yellow/Green)
  - Timer for current state
  - Total vehicles crossed
  - Vehicles waiting for green light
  - Next green light duration

## 🛠️ Technologies Used

- **Python 3.x**
- **OpenCV (cv2)**: Computer vision and video processing
- **NumPy**: Numerical computations
- **MOG2 Background Subtractor**: Motion detection algorithm

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-traffic-light-system.git
cd ai-traffic-light-system
```

2. **Install required dependencies**
```bash
pip install opencv-python numpy
```

## 🚀 Usage

1. **Prepare your video file**
   - Place your traffic video file in the project directory
   - Update the `video_file_name` variable in `main()`:
   ```python
   video_file_name = "your_video.mp4"
   ```

2. **Run the application**
```bash
python traffic_light_system.py
```

3. **Controls**
   - Press `q` to quit the application
   - The video will automatically loop when it reaches the end

## 🧮 Algorithm

### Traffic Light Timing Formula

The green light duration is calculated using:

```
T(green) = max(10, min(60, 10 + (N × 5)))
```

Where:
- `N` = Number of vehicles waiting
- Minimum duration: 10 seconds
- Maximum duration: 60 seconds
- Base time: 10 seconds
- Additional time per vehicle: 5 seconds

### Detection Pipeline

1. **Background Subtraction**: MOG2 algorithm separates moving objects from static background
2. **Shadow Removal**: Thresholding eliminates shadow artifacts
3. **Noise Reduction**: Morphological operations (closing + opening) with 5×5 elliptical kernel
4. **Contour Detection**: Identifies vehicle blobs in the processed frame
5. **Area Filtering**: Removes noise by filtering contours below 500 pixels²
6. **Centroid Calculation**: Computes center point of each detected vehicle
7. **Vehicle Tracking**: Matches vehicles across frames using proximity-based tracking
8. **Line Crossing**: Detects when vehicles cross the designated line

## ⚙️ Configuration Parameters

You can adjust these parameters in the code to fine-tune performance:

```python
MAX_TRACKING_DISTANCE = 50   # Max pixels a vehicle can move between frames
MIN_VEHICLE_AREA = 500        # Minimum contour area to be considered a vehicle
YELLOW_LIGHT_DURATION = 3     # Fixed yellow light duration (seconds)
RED_LIGHT_DURATION = 30       # Fixed red light duration (seconds)
```

### MOG2 Parameters
```python
history=500           # Number of frames for background model
varThreshold=16       # Threshold for pixel-model matching
detectShadows=True    # Enable shadow detection
```

## 📊 System States

### 🟢 GREEN Light
- Vehicles can pass through the intersection
- Duration is adaptive based on previous waiting count
- Transitions to YELLOW after timer expires

### 🟡 YELLOW Light
- Warning phase before red light
- Fixed duration: 3 seconds
- Transitions to RED after timer expires

### 🔴 RED Light
- Vehicles must stop
- System counts waiting vehicles during this phase
- Fixed duration: 30 seconds
- Calculates next green duration based on waiting count

## 📈 Performance Metrics

The system displays:
- **Total Crossed**: Cumulative count of all vehicles that crossed the line
- **Vehicles Waiting**: Number of vehicles detected during current red phase
- **Next Green**: Calculated duration for the upcoming green light

## 🎯 Use Cases

- Traffic flow simulation and analysis
- Smart city traffic management research
- Computer vision education and demonstrations
- Traffic pattern analysis
- Intersection efficiency optimization

## 🔍 How It Works

1. The system continuously processes video frames
2. Moving vehicles are detected using background subtraction
3. Each vehicle is assigned a unique tracking ID
4. When a vehicle crosses the vertical centerline, it's counted
5. During RED phase, crossing vehicles are counted as "waiting"
6. When RED transitions to GREEN, the next green duration is calculated based on waiting count
7. The cycle repeats, adapting to traffic density

## 📝 Future Enhancements

- [ ] Multi-lane detection and separate counting
- [ ] Deep learning-based vehicle detection (YOLO, SSD)
- [ ] Direction-based traffic flow analysis
- [ ] Emergency vehicle priority handling
- [ ] Integration with real traffic light controllers
- [ ] Historical data logging and analytics
- [ ] Multi-intersection coordination

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

Your Name - [@yourhandle](https://twitter.com/yourhandle)

## 🙏 Acknowledgments

- OpenCV community for excellent documentation
- Computer vision researchers for background subtraction algorithms
- Traffic management professionals for domain insights

---

**Note**: This is a simulation system designed for research and educational purposes. Real-world deployment would require additional safety measures, redundancy systems, and regulatory approval.
