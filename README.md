# mini-research

A work-in-progress repository for exploring model interpretability through logit lens analysis and related visualizations.

## Setup

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   (Recommended to use a virtual environment)

2. **Download Dataset**
   ```bash
   python dataset.py
   ```
   After running, copy the printed dataset path from the console output.

3. **Configure Paths**  
   Paste the copied dataset path into both:
   - `loggit_lens.py`: 
     ```python
     if __name__ == '__main__':
         dataset_path = "PASTE_HERE"  # ← Replace this
     ```
   - `analyze.py`:
     ```python
     if __name__ == '__main__':
         dataset_path = "PASTE_HERE"  # ← Replace this
     ```

## Usage

- **Logit Lens Visualization**  
  ```bash
  python loggit_lens.py
  ```
  Generates logit lens analysis plots.

- **Research Analysis Visualization**  
  ```bash
  python analyze.py
  ```
  Produces plots from the mini-research investigation.

## Notes

⚠️
This repository may contain rough edges. For the most complete/stable version of this code, please see the Kaggle Notebook:  
[https://www.kaggle.com/code/finnympargarut/mech-interpretability](https://www.kaggle.com/code/finnympargarut/mech-interpretability)
