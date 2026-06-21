import os
import time
from gtts import gTTS

def clean_filename(name):
    return name.lower().replace(" ", "_")

def translate_grid_sentence(sentence, lang):
    """
    Translates a predicted sentence (either GRID corpus format or conversational phrases)
    into grammatically correct English, Hindi, or Telugu.
    """
    clean_sent = sentence.lower().strip()
    if not clean_sent:
        return ""
        
    # Conversational phrase mappings
    phrases = {
        "thank you": {
            "en": "Thank you",
            "hi": "धन्यवाद",
            "te": "ధన్యవాదాలు"
        },
        "help me": {
            "en": "Help me",
            "hi": "मेरी मदद करें",
            "te": "నాకు సహాయం చేయండి"
        },
        "yes please": {
            "en": "Yes please",
            "hi": "हाँ कृपया",
            "te": "అవును దయచేసి"
        },
        "no thank you": {
            "en": "No thank you",
            "hi": "जी नहीं धन्यवाद",
            "te": "వద్దు ధన్యవాదాలు"
        },
        "water please": {
            "en": "Water please",
            "hi": "कृपया पानी दीजिए",
            "te": "దయచేసి నీరు ఇవ్వండి"
        }
    }
    
    # Check if the sentence matches any conversational phrase
    for phrase, translations in phrases.items():
        if phrase in clean_sent:
            return translations.get(lang, translations["en"])
            
    # Fallback to GRID corpus processing
    words = clean_sent.split()
    if lang == "en":
        # Form a clean English phrase: "Place blue at F 2 now"
        cleaned_words = []
        for w in words:
            if w == "bin":
                cleaned_words.append("place")
            elif w == "f":
                cleaned_words.append("F")
            elif w in ["two", "three", "four", "five"]:
                num_map = {"two": "2", "three": "3", "four": "4", "five": "5"}
                cleaned_words.append(num_map[w])
            else:
                cleaned_words.append(w)
        return " ".join(cleaned_words).capitalize()
        
    # Extract semantic features for GRID translation
    color = "blue"  # default
    number = "two"
    adverb = ""
    
    for w in words:
        if w in ["blue", "red", "green", "white"]:
            color = w
        elif w in ["two", "three", "four", "five"]:
            number = w
        elif w in ["now", "soon", "please", "again"]:
            adverb = w
            
    hi_colors = {"blue": "नीला", "red": "लाल", "green": "हरा", "white": "सफेद"}
    te_colors = {"blue": "నీలిరంగు", "red": "ఎరుపు", "green": "ఆకుపచ్చ", "white": "తెలుపు"}
    
    hi_numbers = {"two": "दो", "three": "तीन", "four": "चार", "five": "पाँच"}
    te_numbers = {"two": "రెండు", "three": "మూడు", "four": "నాలుగు", "five": "ఐదు"}
    
    if lang == "hi":
        # Structure: [adverb] [color] को एफ़ [number] पर रखें
        adv_str = ""
        if adverb == "please": adv_str = "कृपया "
        elif adverb == "now": adv_str = "अभी "
        elif adverb == "soon": adv_str = "जल्द ही "
        elif adverb == "again": adv_str = "दोबारा "
        
        color_hi = hi_colors.get(color, "नीला")
        num_hi = hi_numbers.get(number, "दो")
        
        return f"{adv_str}{color_hi} को एफ़ {num_hi} पर रखें"
        
    elif lang == "te":
        # Structure: [adverb] [color]ను ఎఫ్ [number] వద్ద ఉంచండి
        adv_str = ""
        if adverb == "please": adv_str = "దయచేసి "
        elif adverb == "now": adv_str = "ఇప్పుడు "
        elif adverb == "soon": adv_str = "త్వరలో "
        elif adverb == "again": adv_str = "మళ్ళీ "
        
        color_te = te_colors.get(color, "నీలిరంగు")
        num_te = te_numbers.get(number, "రెండు")
        
        return f"{adv_str}{color_te}ను ఎఫ్ {num_te} వద్ద ఉంచండి"
        
    return sentence.upper()

def pregenerate_audio_assets(target_dir="assets/audio"):
    """
    Pregenerates basic directories. Dynamic TTS runs on demand for arbitrary sentences.
    """
    os.makedirs(target_dir, exist_ok=True)
