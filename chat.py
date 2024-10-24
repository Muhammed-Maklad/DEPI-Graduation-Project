import streamlit as st
import torch
import sounddevice as sd
import numpy as np
import soundfile as sf
import gc  # Garbage collector for memory cleanup
import os
import scraper
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from deep_translator import GoogleTranslator
from langchain.document_loaders import CSVLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
import whisper

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Multilingual Sentiment & BI Chatbot", layout="centered")

# Display logo
st.image("Logo.png", use_column_width=True)

# Initialize translator for Arabic text
translator = GoogleTranslator(source='ar', target='en')

# Initialize session state variables for conditional loading
if "choice_made" not in st.session_state:
    st.session_state.choice_made = False
    st.session_state.choice = None
    st.session_state.bert_loaded = False
    st.session_state.whisper_loaded = False
    st.session_state.chatbot_loaded = False

# If the choice is not yet made, allow the user to choose
if not st.session_state.choice_made:
    option = st.radio(
        "Select an option:",
        ("Sentiment Analysis", "Chat Bot")
    )
    if st.button("Confirm Choice"):
        st.session_state.choice_made = True
        st.session_state.choice = option
        st.rerun()  # Refresh the page to lock the choice

# Function to load BERT and Whisper models for Sentiment Analysis


def load_sentiment_analysis_components():
    # Load Whisper for transcription (CPU)
    if not st.session_state.whisper_loaded:

        st.session_state.whisper_model = whisper.load_model(
            "medium", device='cuda')
        st.session_state.whisper_loaded = True

    # Load fine-tuned BERT model for sentiment classification (GPU)
    if not st.session_state.bert_loaded:
        st.session_state.device = torch.device('cuda')
        st.session_state.tokenizer_bert, st.session_state.model_bert = load_finetuned_model()
        st.session_state.bert_loaded = True

# Function to load components for the BI Chat Bot


def load_chatbot_components():
    if not st.session_state.chatbot_loaded:
        # Initialize LLM and embeddings
        st.session_state.llm = OllamaLLM(model="llama3.1:8b-instruct-q8_0")
        st.session_state.embedding_model = OllamaEmbeddings(
            model="llama3.1:8b-instruct-q8_0")
        st.session_state.chatbot_loaded = True

# Load fine-tuned DistilBERT model for sentiment classification (on GPU)


def load_finetuned_model():
    model_directory = r"FinetunedModel"
    tokenizer = DistilBertTokenizer.from_pretrained(
        "D:\PythonProjects\DEPI Grad Project\FinetunedModel")
    model = DistilBertForSequenceClassification.from_pretrained(
        "D:\PythonProjects\DEPI Grad Project\FinetunedModel").to(st.session_state.device)
    return tokenizer, model

# Function to record audio for voice input


def record_audio(duration=5, fs=16000):
    st.write("Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs,
                   channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to finish
    return np.squeeze(audio)

# Function to transcribe audio using Whisper


def transcribe_audio(audio):
    st.write("Transcribing audio...")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(
        st.session_state.whisper_model.device)
    result = st.session_state.whisper_model.decode(mel)
    del audio, mel
    gc.collect()
    return result.text

# Function to classify sentiment using the fine-tuned DistilBERT model


def classify_text(transcription):
    # Translate to English if Arabic is detected
    if any('\u0600' <= char <= '\u06FF' for char in transcription):
        transcription = translator.translate(transcription)

    inputs = st.session_state.tokenizer_bert(
        transcription, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(st.session_state.device)
              for key, value in inputs.items()}

    with torch.no_grad():
        outputs = st.session_state.model_bert(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

    del inputs, transcription, probabilities, logits, outputs
    gc.collect()

    labels = ['negative', 'positive']
    return labels[predicted_class.item()]

# Function to set up RAG system with a CSV file


def setup_rag_system(csv_file_path):
    st.write('Received a CSV file. Now loading it to memory...')
    loader = CSVLoader(file_path=csv_file_path, encoding='utf')
    documents = loader.load()

    # Free the memory for the loader after loading documents
    del loader
    gc.collect()

    # Extract review texts, classify sentiment, and embed them with sentiment context
    texts = []
    metadatas = []

    for doc in documents:
        review_text = doc.page_content
        # Directly use the sentiment from the column
        # Default to neutral if missing
        sentiment = doc.metadata.get('Sentiment', 'neutral').lower()
        # Add sentiment context to the text for embedding
        sentiment_text = f"{sentiment}: {review_text}"
        texts.append(sentiment_text)

        # Store metadata with sentiment
        metadata = doc.metadata
        metadata['Sentiment'] = sentiment
        metadatas.append(metadata)

    # Free the memory for documents since we no longer need them
    del documents
    gc.collect()  # Force garbage collection to free memory

    # Generate embeddings and create the FAISS index
    st.write('Transforming the data to enable chatbot responses...')
    vector_store = FAISS.from_texts(
        texts=texts, embedding=st.session_state.embedding_model, metadatas=metadatas)

    # Free the memory for texts and metadata
    del texts, metadatas
    gc.collect()  # Force garbage collection to free memory

    # Save the vector store to disk and free memory
    vector_store.save_local("faiss_reviews_index")
    st.write('Finished saving the data.')

    return "faiss_reviews_index"  # Return the path to the saved vector store

# Load vector store from disk (when needed)


def load_vector_store(path="faiss_reviews_index"):
    return FAISS.load_local(path, st.session_state.embedding_model, allow_dangerous_deserialization=True)

# Function to generate a response based on RAG system with conversation history


def generate_response(query, vector_store, chat_history, top_k=5):
    # Determine if the query asks for a specific sentiment
    if "negative" in query.lower():
        sentiment_filter = 'negative'
    elif "positive" in query.lower():
        sentiment_filter = 'positive'
    elif "neutral" in query.lower():
        sentiment_filter = 'neutral'
    else:
        sentiment_filter = None  # No specific sentiment filter

    # Perform similarity search, then filter based on sentiment context
    all_docs = vector_store.similarity_search(
        query, k=top_k*2)  # Retrieve more docs initially

    if sentiment_filter:
        # Filter documents based on the sentiment metadata
        relevant_docs = [
            doc for doc in all_docs
            if doc.metadata.get('sentiment') == sentiment_filter
        ]
        relevant_docs = relevant_docs[:top_k]  # Limit to top_k after filtering
    else:
        # No filter, retrieve documents normally
        relevant_docs = all_docs[:top_k]

    relevant_reviews = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Combine previous history into a context, limited to the last 10 messages
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]  # Keep only the last 10 messages

    # Construct a prompt with the chat history, the relevant reviews, and the current query
    context = "\n\n".join(chat_history)
    prompt = f'''
    You are an expert Business Intelligence analyst with deep expertise in analyzing product feedback, customer satisfaction, and market trends. 
    Your client relies on you to provide detailed and actionable insights based on the reviews you've gathered. This feedback includes positive 
    and negative experiences, pain points, and areas of improvement. Your goal is to synthesize this data into valuable recommendations that 
    can help your client make informed strategic decisions.

    The context below contains both past conversations and relevant reviews extracted from customer feedback. These reviews highlight specific 
    product experiences and provide a foundation for understanding customer needs, desires, and frustrations. Your analysis should focus on key 
    patterns, emerging themes, and any noticeable trends.
    
    Negative reviews are below 3 stars and Neutral reviews are 3 stars while Positive reviews are 3 to 5 starts. Take this into account

    ### Key Objectives:
    1. **Comprehensive Analysis**: Carefully consider the user's question and address all parts of it. Break down complex questions into sub-parts 
    and ensure that each is answered in a clear and concise manner.
    2. **Actionable Insights**: Provide actionable insights based on the reviews and context. Your recommendations should help your client optimize 
    product features, improve customer satisfaction, and identify any potential risks or opportunities.
    3. **Data-Driven Reasoning**: Base your analysis on the data provided, backing up your conclusions with specific examples from the relevant reviews. 
    Highlight notable quotes or trends if they add value to your answer.
    4. **Strategic Suggestions**: Offer concrete suggestions on how your client can address customer feedback. Consider practical steps that can 
    enhance product performance, marketing strategy, or customer experience.
    5. **Professional Clarity**: Use clear, professional language that is easy to understand. Avoid jargon unless necessary and ensure that your 
    analysis is well-structured and logically flows from data to conclusion.

    ### Context for Analysis:
    Below is the context, including past conversations and relevant reviews. Use this information to guide your response. 
    Focus on extracting meaningful patterns and providing insights tailored to the user's question.

    #### Relevant Reviews:
    {relevant_reviews}

    #### Conversation Context:
    {context}

    ### User's Question:
    {query}

    ### Instructions:
    1. **Think like a BI Analyst**: Focus on data, trends, and insights. Provide clarity and be straightforward in your suggestions.
    2. **Synthesize Information**: Look at both specific details and the bigger picture. Use the data to highlight what stands out and what matters most.
    3. **Keep it Concise**: Aim for a structured response with key takeaways, avoiding any unnecessary or irrelevant details.
    4. **Professional Tone**: Your client expects a clear, factual, and well-organized analysis. Be confident in your assessment and support your 
    claims with evidence.
    5. **Recommendations**: Provide recommendations that are practical and directly applicable. Consider any potential limitations or additional 
    considerations your client should be aware of.

    Proceed with the analysis using the information above, and focus on delivering a professional and insightful response.
    '''

    # Generate the response from the model
    response = st.session_state.llm.invoke(prompt)

    # Update the chat history, and maintain only the last 10 messages
    chat_history.append(f"User: {query}")
    chat_history.append(f"AI: {response}")

    # Keep only the last 10 entries in chat_history
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    # Update the session state with the trimmed chat history
    st.session_state.chat_history = chat_history

    # Free the memory for the retrieved documents
    del relevant_docs, relevant_reviews
    gc.collect()

    return response


# Once the user has chosen, load the relevant models/components
if st.session_state.choice_made:
    if st.session_state.choice == "Sentiment Analysis":
        load_sentiment_analysis_components()

        # Sentiment Analysis Section
        st.header("Sentiment Analysis")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Function to render chat messages
        def render_messages():
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])

        # Display chat messages from history
        render_messages()

        # Layout with chat input and button side by side
        col1, col2 = st.columns([6, 1])

        with col1:
            prompt = st.text_input(
                "Enter your text or press the mic to record audio:")

        with col2:
            mic_button = st.button("🎤")

        # If user submits text input
        if prompt:
            with col1:
                with st.chat_message("user"):
                    st.markdown(prompt)
            st.session_state.messages.append(
                {"role": "user", "content": prompt})

            response = classify_text(prompt)
            with col1:
                with st.chat_message("assistant"):
                    st.markdown(f"Sentiment: {response}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Sentiment: {response}"})

        # If user presses the mic button to record voice input
        if mic_button:
            audio_data = record_audio(duration=5)
            transcription = transcribe_audio(audio_data)
            del audio_data
            gc.collect()

            with col1:
                with st.chat_message("user"):
                    st.markdown(f"Transcribed: {transcription}")
            st.session_state.messages.append(
                {"role": "user", "content": f"Transcribed: {transcription}"})

            sentiment = classify_text(transcription)
            del transcription
            gc.collect()
            with col1:
                with st.chat_message("assistant"):
                    st.markdown(f"Sentiment: {sentiment}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Sentiment: {sentiment}"})

    elif st.session_state.choice == "Chat Bot":
        load_chatbot_components()

        # Chat Bot Section
        st.header("Business Intelligence Chatbot")

        # Ensure `vector_store_path`, `chat_history`, and message history are in session state
        if "vector_store_path" not in st.session_state:
            st.session_state.vector_store_path = None
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.link_uploaded = False

        if not st.session_state.vector_store_path and not st.session_state.link_uploaded:
            data_source_choice = st.radio(
                "Choose your data source:", ("Upload CSV", "Enter a URL"))

            # If user chooses to upload a CSV file
            if data_source_choice == "Upload CSV":
                uploaded_file = st.file_uploader(
                    "Upload a CSV file", type="csv")
                if uploaded_file is not None:
                    # Save the uploaded file locally
                    file_path = f"uploaded_data.csv"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Set up the RAG system with the saved file path
                    st.session_state.vector_store_path = setup_rag_system(
                        file_path)
                    st.success(
                        "File uploaded, saved locally, and data processed successfully!")

                    # Free the memory for the uploaded CSV file
                    del uploaded_file
                    gc.collect()

            # If user chooses to enter a URL
            elif data_source_choice == "Enter a URL":
                url_input = st.text_input("Enter a URL to scrape data:")
                if url_input:
                    st.write('Searching the amazon page')
                    csv_file_path = scraper.scrape_amazon_reviews(url_input)
                    st.write('Found the amazon paged and saved the reviews.')
                    if csv_file_path:
                        # Use the scraped CSV to set up the RAG system
                        st.session_state.vector_store_path = setup_rag_system(
                            csv_file_path)
                        st.session_state.link_uploaded = True
                        st.success(
                            "Done! You can now ask your chat bot questions about the product")

        # Function to render chat messages
        def render_chatbot_messages():
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])

        # Check if the vector store path is set, if not, show a message to upload a file or provide a URL
        if st.session_state.vector_store_path:
            # Load vector store from disk (reduces in-memory usage)
            vector_store = load_vector_store(
                st.session_state.vector_store_path)

            # Display chat messages from history
            render_chatbot_messages()

            # Chat interface for RAG-based responses
            user_query = st.text_input("Ask your question:")
            if user_query:
                with st.chat_message("user"):
                    st.markdown(user_query)
                st.session_state.messages.append(
                    {"role": "user", "content": user_query})

                response = generate_response(
                    user_query, vector_store, st.session_state.chat_history, top_k=20)
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})

                # Free the vector store after processing (reduce memory footprint)
                del vector_store
                gc.collect()
        else:
            st.info(
                "Please upload a CSV file or provide a URL to initialize the chat bot.")
