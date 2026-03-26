#  Voice-Based Technical Evaluator

An AI-powered oral assessment system that evaluates spoken answers by converting speech to text and comparing it with reference answers using semantic similarity.

---

##  Overview

The Voice-Based Technical Evaluator automates oral assessments by allowing users to record their answers instead of typing. The system transcribes speech, analyzes the meaning of the response, and provides a score along with meaningful feedback.

This project demonstrates the integration of speech recognition, natural language processing, and web development to build a real-world AI application.

---

##  Features

-  Record spoken answers directly from the browser
-  Automatic speech-to-text transcription
-  Semantic similarity-based evaluation
-  Instant scoring system
-  Automated feedback generation
-  Audio playback and transcript viewing
-  Simple and interactive web interface

---

##  How It Works

1. **Voice Input**
   - User records their answer using the web interface.

2. **Speech-to-Text**
   - Audio is converted into text using Whisper ASR.

3. **Text Preprocessing**
   - Removes filler words, punctuation, and noise from the transcript.

4. **Semantic Encoding**
   - Both user answer and reference answer are converted into embeddings using SBERT.

5. **Similarity Calculation**
   - Cosine similarity is computed between the two embeddings.

6. **Scoring & Feedback**
   - Based on similarity score, marks and feedback are generated.

---

##  Tech Stack

- **Backend:** Python, Flask  
- **Speech Recognition:** Whisper ASR  
- **NLP:** Sentence Transformers (SBERT)  
- **Data Processing:** NumPy, Pandas  
- **Frontend:** HTML, CSS, JavaScript  

---

##  Scoring Logic

| Similarity Score | Evaluation        |
|-----------------|------------------|
| 0.85 - 1.00     | Excellent        |
| 0.70 - 0.84     | Good             |
| 0.50 - 0.69     | Average          |
| < 0.50          | Needs Improvement|

---

##  Key Concepts Used

- Automatic Speech Recognition (ASR)
- Sentence Embeddings
- Semantic Similarity
- Cosine Similarity
- Text Preprocessing


