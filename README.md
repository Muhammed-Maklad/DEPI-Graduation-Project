![Chatbot Logo](https://github.com/user-attachments/assets/ea8f09f9-7a93-4aab-b74b-61d846971cbe)

## Project Description
This project is a powerful web application designed to analyze sentiment in multilingual text and facilitate business intelligence through a chatbot. Using Streamlit as the front-end, it integrates state-of-the-art machine learning models, including Whisper for audio transcription, a fine-tuned DistilBERT for sentiment classification, and advanced language models for generating intelligent responses. The application can transcribe spoken input, classify sentiment in customer feedback, and answer business-related queries based on product reviews extracted via web scraping from Amazon.

## Installation Instructions
To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Muhammed-Maklad/DEPI-Graduation-Project
   cd DEPI-Graduation-Project
   ```

2. **Set Up a Python Environment**:
   Ensure you have Python 3.8 or higher installed. Set up a virtual environment and activate it:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Required Dependencies**:
   Install the necessary packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure additional dependencies like `pandas`, `torch`, `transformers`, `streamlit`, `beautifulsoup4`, `deep-translator`, `sounddevice`, and `FAISS` are included.

4. **Download and Place Necessary Models**:
   - Place the fine-tuned DistilBERT model in the specified directory (`FinetunedModel`).
   - Install Whisper and ensure CUDA is configured if using GPU acceleration.

## Usage Instructions
1. **Run the Streamlit App**:
   ```bash
   streamlit run chat.py
   ```

2. **Select an Option**:
   - Choose "Sentiment Analysis" to input `text` or `record audio` for sentiment classification.
   - Choose "Chat Bot" to upload a `CSV` of product reviews or provide a `URL` to scrape Amazon reviews. The chatbot then answers queries based on extracted data.

3. **Recording Audio**:
   - Click the microphone button to record audio input for transcription and sentiment analysis.

4. **Scraping Amazon Reviews**:
   - The `scraper.py` script collects reviews from an Amazon product URL, classifies them, and saves them as `reviews.csv`. The reviews are processed and used to provide insights within the chatbot.

## Code Functionality Overview
- **`chat.py`**:
   - Contains the main logic for running the Streamlit app.
   - Integrates Whisper for transcription and DistilBERT for sentiment analysis.
   - Sets up a Retrieval-Augmented Generation (RAG) system using FAISS and embeddings for chatbot responses.
- **`scraper.py`**:
   - Scrapes Amazon reviews and processes details such as reviewer name, rating, review title, date, and full content.
   - Classifies sentiment based on star ratings and saves results to a CSV file.
 
## Demo
Alternatively, watch a short video walkthrough of the app's functionality:
<video controls>
  <source src="https://github.com/user-attachments/assets/264e4a43-4d19-4de1-b379-a903ab7cc896" type="video/mp4">
  Your browser does not support the video tag.
</video>




## Contributing Guidelines
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

Please ensure that your code follows the project’s style and includes necessary documentation.

## License Information
This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.
