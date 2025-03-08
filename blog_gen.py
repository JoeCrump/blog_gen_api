from fastapi import FastAPI, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List
import os
from dotenv import load_dotenv


load_dotenv()
app = FastAPI()

# Initialize Gemini Model
GEMINI_MODEL = "gemini-2.0-flash"


def get_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")  # Fetch from Heroku Config Vars
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing API Key: Set GOOGLE_API_KEY in Heroku Config Vars.")

    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=GEMINI_MODEL,
        temperature=0.7,
        max_tokens=None,
        timeout=30,
        max_retries=2,
    )

# Extract video ID from YouTube URL
def extract_video_id(link: str) -> str:
    """Extract video ID from YouTube URL."""
    if "v=" in link:
        return link.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in link:
        return link.split("youtu.be/")[-1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL")

# Fetch transcript from YouTube
def get_transcript(video_url: str) -> str:
    try:
        video_id = extract_video_id(video_url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript])
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching transcript: {str(e)}")

# Process individual transcript
async def process_transcript(video_url: str):
    transcript_text = get_transcript(video_url)
    gemini = get_gemini()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this transcript into key points in English, regardless of its original language."),
        ("human", transcript_text)
    ])
    response = gemini.invoke(prompt.format())
    return response.content

# Merge multiple outlines
async def merge_outlines(outlines: List[str]):
    gemini = get_gemini()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Merge these outlines into a structured format in English."),
        ("human", "\n".join(outlines))
    ])
    response = gemini.invoke(prompt.format())
    return response.content

# Generate full blog post with HTML formatting
async def generate_blog_post(merged_outline: str):
    gemini = get_gemini()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional blog writer. Your task is to write a detailed, engaging, and well-structured blog post based on the provided outline. Follow these guidelines strictly:

- **Word Count:** The article must be between **1900 and 2300 words**—never below 1900 and never exceeding 2300 words.
- **Formatting:** Ensure the content is formatted in **valid HTML** for direct use in WordPress.
- **Headings & Structure:**  
  - Use **H2 for major sections** and **H3/H4 for subsections** where appropriate.  
  - Maintain a clear hierarchical structure to enhance readability and SEO.  
- **Text Formatting:**  
  - Use **bold (<b>)** for key points and **underline (<u>)** for emphasis.  
  - Utilize **ordered (<ol>) and unordered (<ul>) lists** where necessary.  
  - Ensure **paragraphs are well-spaced and easy to read**.  
- **Quality & Coherence:**  
  - Maintain a **logical flow** between sections.  
  - Avoid redundancy and ensure smooth transitions between ideas.  
  - Keep the writing **engaging, informative, and SEO-friendly** while maintaining a professional tone.

Now, generate a well-structured and high-quality blog post adhering to these guidelines."""),
        ("human", merged_outline)
    ])
    response = gemini.invoke(prompt.format())
    return response.content

@app.post("/process_videos/")
async def process_videos(video_urls: List[str]):
    if not isinstance(video_urls, list):
        raise HTTPException(status_code=400, detail="Input should be a valid list of video URLs.")
    outlines = [await process_transcript(video_url) for video_url in video_urls]
    merged_outline = await merge_outlines(outlines)
    final_blog_post = await generate_blog_post(merged_outline)
    return {"blog_post": final_blog_post}