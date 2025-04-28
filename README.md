# Can My ML Models Guess Your Sketches?

This project recreates the concept of **"Quick, Draw!"**, a game developed and released by Google in 2016 where players are asked to sketch prompts (like "cat", "pizza", or "bicycle") in less than 20 seconds, and an AI tries to guess what you are drawing *in real-time*. I trained two machine learning models for this task: a CNN model and an RNN model. All model training Jupyter notebooks can be found under the "models" directory.

You can view and play my deployed version of the game here: [Live Website](https://ml-sketch-guesser.up.railway.app/)  

A gameplay sample video: [View Sample Video](https://drive.google.com/file/d/10mYG6Lfbgmhn0CiVP5ODB-WTAUAIwITz/view?usp=drive_link)

---

## Datasets

Two types of datasets from the **QuickDraw** project were used:

- **CNN Model**:  
  - Inputs: 28x28 grayscale images (bitmap format)  
  - Each image represents a preprocessed centered drawing
  - Stored as `.npy` files

- **RNN (LSTM) Model**:  
  - Inputs: Sequences of pen movements (vector format), each stroke recorded as (∆x, ∆y, pen state)
  - Stored in `.npz` files

**Categories Used (20 classes):**
- cat
- tree
- fish
- clock
- castle
- crown
- lollipop
- moon
- watermelon
- tornado
- apple
- bowtie
- bicycle
- diamond
- flower
- butterfly
- eye
- lightning
- cloud
- pizza

**Splits:**
- 80% training
- 10% development
- 10% test

**Total Data Points:**  
10,000 data points per class -> 200,000 sketches

---

## Model Architectures

### CNN Model (Image-based)

- **Architecture:**

![Model Architecture](/CNN_Model_Architecture.png)

- **Loss Function:** Cross Entropy Loss
- **Final Performance:**  
  - Training Accuracy: 91.94%  
  - Validation Accuracy: 91.81%
  - Test Accuracy: 91.83% 

---

### RNN Model (Stroke-sequence based, LSTM)

- **Architecture:**

![Model Architecture](/RNN_Model_Architecture.png)

- **Loss Function:** Cross Entropy Loss
- **Final Performance:**  
  - Train Accuracy: 95.12%  
  - Validation Accuracy: 94.58%
  - Test Accuracy: 94.51%
 
  ---
## How to Run the Project Locally

If you'd like to run the project locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/Sohaila-Abdulsattar-Mohammed/ML-Sketch-Guesser.git
cd ML-Sketch-Guesser
```

### 2. Create a Virtual Environment (Recommended)
It is recommended to use a virtual environment to manage dependencies:
```bash
python -m venv venv
source venv/bin/activate       # On macOS/Linux
# OR
venv\Scripts\activate          # On Windows
```

### 3. Install Required Packages
Install all necessary Python libraries listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Run the Backend Server
Start the Flask server by running:
```bash
python app.py
```
The server will start at:
```
http://localhost:5000
```

### 5. Open the Frontend
Once the server is running:
- Open your browser.
- Navigate to:
```
http://localhost:5000
```
You should now see the game interface!

---

## How the Game Works
- You choose a machine learning model to test.
- A target word is shown.
- You draw the word in the canvas.
- The system predicts what you are drawing in real time; the bar chart on the left side of the screen shows the model's predictions as they update.
- If the model successfully and confidently identifies your drawing before time runs out, you win!

---

## Resources

- QuickDraw Dataset: [Google QuickDraw Dataset](https://quickdraw.withgoogle.com/data)
- Sketch-RNN Paper: ["A Neural Representation of Sketch Drawings"](https://arxiv.org/abs/1704.03477)

---

**Feel free to reach out if you have any questions!**
