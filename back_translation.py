"""
Back-translation module for Enhanced Language Learning Bot.

Back-translation is a technique where:
1. Text is translated from source language to a pivot language
2. Then translated back to the source language
This helps verify if the original meaning is preserved and reveals ambiguities.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Any
from enhanced_translator import translate_text

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Default pivot languages to use for back-translation
DEFAULT_PIVOT_LANGUAGES = ["zh-CN", "ja", "ru", "ar", "de", "fr"]

def get_random_pivot_language(exclude: Optional[str] = None) -> str:
    """
    Get a random pivot language for back-translation.
    
    Args:
        exclude (str, optional): Language code to exclude
        
    Returns:
        str: Random pivot language code
    """
    languages = [lang for lang in DEFAULT_PIVOT_LANGUAGES if lang != exclude]
    return random.choice(languages) if languages else "zh-CN"

def perform_back_translation(
    text: str, 
    source_lang: str, 
    pivot_lang: Optional[str] = None,
    service_chain: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Perform back-translation using specified pivot language.
    
    Args:
        text (str): Original text to back-translate
        source_lang (str): Source language code
        pivot_lang (str, optional): Pivot language code
        service_chain (List[str], optional): Translation service chain to use
        
    Returns:
        Dict: Dictionary with original, pivot, and back-translated text
    """
    if not pivot_lang:
        # Don't use source language as pivot
        pivot_lang = get_random_pivot_language(exclude=source_lang)
    
    result = {
        "original_text": text,
        "original_lang": source_lang,
        "pivot_lang": pivot_lang,
        "pivot_text": None,
        "back_text": None,
        "success": False
    }
    
    try:
        # Step 1: Source → Pivot
        pivot_translation = translate_text(text, source_lang, pivot_lang)
        
        if not pivot_translation:
            logger.error(f"Failed to translate to pivot language {pivot_lang}")
            return result
        
        result["pivot_text"] = pivot_translation
        
        # Step 2: Pivot → Source
        back_translation = translate_text(pivot_translation, pivot_lang, source_lang)
        
        if not back_translation:
            logger.error(f"Failed to translate back from pivot language {pivot_lang}")
            return result
        
        result["back_text"] = back_translation
        result["success"] = True
        
        return result
    
    except Exception as e:
        logger.error(f"Error in back-translation: {e}")
        return result

def perform_multi_pivot_back_translation(
    text: str,
    source_lang: str,
    pivot_langs: Optional[List[str]] = None,
    count: int = 3
) -> List[Dict[str, Any]]:
    """
    Perform back-translation using multiple pivot languages.
    
    Args:
        text (str): Original text to back-translate
        source_lang (str): Source language code
        pivot_langs (List[str], optional): List of pivot language codes
        count (int): Number of different pivot languages to use
        
    Returns:
        List[Dict]: List of back-translation results
    """
    if not pivot_langs:
        # Use default languages excluding source language
        available_langs = [lang for lang in DEFAULT_PIVOT_LANGUAGES if lang != source_lang]
        # Choose up to 'count' languages
        pivot_langs = random.sample(available_langs, min(count, len(available_langs)))
    
    results = []
    
    for pivot_lang in pivot_langs:
        result = perform_back_translation(text, source_lang, pivot_lang)
        results.append(result)
    
    return results

def compare_back_translations(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple back-translations and provide analysis.
    
    Args:
        results (List[Dict]): List of back-translation results
        
    Returns:
        Dict: Analysis of back-translations including commonalities and differences
    """
    if not results or len(results) < 2:
        return {"error": "Need at least 2 back-translations to compare"}
    
    # Extract back-translated texts
    back_texts = [r.get("back_text", "") for r in results if r.get("back_text")]
    
    # Calculate simple similarity measure (future enhancement: use more sophisticated NLP)
    similarity_percentage = calculate_similarity(back_texts)
    
    analysis = {
        "count": len(back_texts),
        "similarity_percentage": similarity_percentage,
        "pivot_languages": [r.get("pivot_lang") for r in results],
    }
    
    # Identify potential meaning shifts (simple naive implementation)
    original = results[0].get("original_text", "")
    words_orig = set(original.lower().split())
    
    # Compare with back-translations
    for i, back_text in enumerate(back_texts):
        if back_text:
            words_back = set(back_text.lower().split())
            # Words in original but not in back-translation
            missing = words_orig - words_back
            # Words in back-translation but not in original
            added = words_back - words_orig
            
            if missing or added:
                if "changes" not in analysis:
                    analysis["changes"] = []
                
                analysis["changes"].append({
                    "pivot_lang": results[i].get("pivot_lang"),
                    "missing_words": list(missing) if missing else None,
                    "added_words": list(added) if added else None
                })
    
    return analysis

def calculate_similarity(texts: List[str]) -> float:
    """
    Calculate similarity between texts. Simple implementation using word overlap.
    
    Args:
        texts (List[str]): List of texts to compare
        
    Returns:
        float: Similarity percentage (0-100)
    """
    if not texts or len(texts) < 2:
        return 0.0
    
    # Convert to sets of words
    word_sets = [set(text.lower().split()) for text in texts]
    
    # Calculate overlap between all pairs
    total_similarity = 0.0
    pair_count = 0
    
    for i in range(len(word_sets)):
        for j in range(i+1, len(word_sets)):
            set1, set2 = word_sets[i], word_sets[j]
            if not set1 or not set2:
                continue
                
            # Jaccard similarity
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            if union > 0:
                similarity = (intersection / union) * 100
                total_similarity += similarity
                pair_count += 1
    
    # Average similarity
    return total_similarity / pair_count if pair_count > 0 else 0.0

def format_back_translation_message(result: Dict[str, Any]) -> str:
    """
    Format back-translation result as a user-friendly message.
    
    Args:
        result (Dict): Back-translation result
        
    Returns:
        str: Formatted message
    """
    if not result.get("success"):
        return "⚠️ Back-translation failed. Please try again."
    
    original_text = result.get("original_text", "")
    pivot_text = result.get("pivot_text", "")
    back_text = result.get("back_text", "")
    original_lang = result.get("original_lang", "unknown")
    pivot_lang = result.get("pivot_lang", "unknown")
    
    message = (
        f"🔄 *Back-Translation*\n\n"
        f"1️⃣ *Original ({original_lang})*:\n{original_text}\n\n"
        f"2️⃣ *Pivot ({pivot_lang})*:\n{pivot_text}\n\n"
        f"3️⃣ *Back to {original_lang}*:\n{back_text}\n\n"
    )
    
    # Simple analysis
    if original_text != back_text:
        message += "ℹ️ *Note*: The back-translation differs from the original, indicating potential ambiguities or translation challenges."
    else:
        message += "✅ *Note*: The back-translation perfectly matches the original!"
    
    return message