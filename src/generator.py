# src/generator.py
from gtts import gTTS
import io
from src.model_loader import initialise_llm

def generate_podcast_script(answer):
    llm = initialise_llm()
    
    # We tell the AI to write "natural speech" (um, ah, oh, exactly!)
    prompt = f"""
    Context: {answer}
    Task: Write a natural 2-minute podcast dialogue between Sarah and Maya.
    
    STYLE RULES:
    1. Don't be formal. Use phrases like "That's a great point," "Wait, really?", and "Exactly."
    2. Make sure they react to what the other person just said.
    3. Sarah is the expert, Maya is the curious learner.
    4. End with: "That's all for today, thanks for joining us!"
    
    Format:
    SARAH: [text]
    MAYA: [text]
    """
    response = llm.complete(prompt)
    return response.text

def text_to_speech_bytes(script_text):
    """
    Creates natural flow without robotic 'name announcing'.
    """
    # 1. REMOVE the names completely so they aren't spoken
    clean_text = script_text.replace("SARAH:", "").replace("MAYA:", "")
    
    # 2. Add '...' in place of the speaker change to create a 0.5s breathing pause
    # This helps the brain realize the speaker has switched.
    final_text = clean_text.replace("\n", "... ") 
    
    tts = gTTS(text=final_text, lang='en', slow=False)
    
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    return audio_buffer.getvalue()