import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import concurrent.futures
import time

# Step 1: Initialize the summarizer once (cached for performance)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-small")

summarizer = load_summarizer()

# Step 2: Function to fetch the transcript of a YouTube video (cached)
@st.cache_data
def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join([entry['text'] for entry in transcript])
        return full_transcript
    except Exception as e:
        return f"Error retrieving transcript: {e}"

# Step 3: Summarize the transcript with chunking for longer texts (cached)
@st.cache_data
def summarize_transcript(transcript):
    try:
        # Check the transcript length and split it into chunks if necessary
        if len(transcript) > 1000:
            # Split into chunks of 1000 characters
            chunks = [transcript[i:i + 1000] for i in range(0, len(transcript), 1000)]
            summaries = []
            
            # Process chunks in parallel for faster execution
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_summaries = [executor.submit(summarizer, chunk, max_length=150, min_length=40, do_sample=False) for chunk in chunks]
                for future in concurrent.futures.as_completed(future_summaries):
                    summaries.append(future.result()[0]['summary_text'])
            
            # Combine all chunk summaries
            return " ".join(summaries)
        else:
            summary = summarizer(transcript, max_length=150, min_length=40, do_sample=False)
            return summary[0]['summary_text']
    except Exception as e:
        return f"Error summarizing transcript: {e}"

# Step 4: Extract video ID from YouTube URL
def extract_video_id(video_url):
    if "v=" in video_url:
        return video_url.split("v=")[1].split("&")[0]
    return video_url.split("/")[-1]

# Step 5: Main function to fetch and summarize the transcript
def get_summary_from_youtube(video_url):
    video_id = extract_video_id(video_url)

    transcript = get_youtube_transcript(video_id)
    
    # Fallback for missing or unavailable transcripts
    if "Error" in transcript:
        return transcript

    summary = summarize_transcript(transcript)
    return summary

# Step 6: Streamlit UI
def main():
    st.title("YouTube Transcript Summarizer")

    # Input field for the YouTube video URL
    video_url = st.text_input("Enter YouTube Video URL:")

    # Button to generate summary
    if st.button("Summarize"):
        if video_url:
            with st.spinner("Fetching and summarizing..."):
                summary = get_summary_from_youtube(video_url)
                st.subheader("Summary:")
                st.write(summary)
        else:
            st.warning("Please enter a valid YouTube video URL.")

# Run the app
if __name__ == "__main__":
    main()
