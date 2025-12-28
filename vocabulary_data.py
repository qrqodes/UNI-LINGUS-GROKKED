"""
This module provides vocabulary data for the translation bot's word learning game.
Words are organized by difficulty level (beginner, intermediate, advanced).
"""
import json
import re
import logging
import random
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import AI services for vocabulary generation
try:
    from ai_services_simplified import query_claude
except ImportError:
    # Define a fallback function if import fails
    def query_claude(prompt):
        logger.warning("AI services not available for vocabulary generation")
        return None
import random

# Vocabulary data organized by difficulty level
VOCABULARY = {
    'beginner': [
        {
            'word': 'cat',
            'definition': 'A small domesticated carnivorous mammal.',
            'example': 'The cat slept on the windowsill all afternoon.',
            'synonym': 'feline',
            'translations': {'es': 'gato', 'fr': 'chat', 'it': 'gatto', 'pt': 'gato', 'ru': 'кот', 'zh-CN': '猫'}
        },
        {
            'word': 'dog',
            'definition': 'A domesticated carnivorous mammal, typically kept as a pet or for work.',
            'example': 'She takes her dog for a walk every morning.',
            'synonym': 'canine',
            'translations': {'es': 'perro', 'fr': 'chien', 'it': 'cane', 'pt': 'cachorro', 'ru': 'собака', 'zh-CN': '狗'}
        },
        {
            'word': 'house',
            'definition': 'A building for human habitation.',
            'example': 'They bought a new house in the suburbs.',
            'synonym': 'home',
            'translations': {'es': 'casa', 'fr': 'maison', 'it': 'casa', 'pt': 'casa', 'ru': 'дом', 'zh-CN': '房子'}
        },
        {
            'word': 'book',
            'definition': 'A written or printed work consisting of pages bound together.',
            'example': 'I read an interesting book about space exploration.',
            'synonym': 'publication',
            'translations': {'es': 'libro', 'fr': 'livre', 'it': 'libro', 'pt': 'livro', 'ru': 'книга', 'zh-CN': '书'}
        },
        {
            'word': 'car',
            'definition': 'A road vehicle powered by an engine designed to carry a small number of people.',
            'example': 'We drove to the beach in my new car.',
            'synonym': 'automobile',
            'translations': {'es': 'coche', 'fr': 'voiture', 'it': 'auto', 'pt': 'carro', 'ru': 'машина', 'zh-CN': '汽车'}
        },
        {
            'word': 'apple',
            'definition': 'The round fruit of a tree of the rose family with red, yellow, or green skin and sweet flesh.',
            'example': 'She ate an apple for a healthy snack.',
            'synonym': 'fruit',
            'translations': {'es': 'manzana', 'fr': 'pomme', 'it': 'mela', 'pt': 'maçã', 'ru': 'яблоко', 'zh-CN': '苹果'}
        },
        {
            'word': 'water',
            'definition': 'A transparent, colorless, odorless liquid that forms the seas, lakes, rivers, and rain.',
            'example': 'Drink plenty of water on hot days.',
            'synonym': 'H2O',
            'translations': {'es': 'agua', 'fr': 'eau', 'it': 'acqua', 'pt': 'água', 'ru': 'вода', 'zh-CN': '水'}
        },
        {
            'word': 'table',
            'definition': 'A piece of furniture with a flat top and one or more legs.',
            'example': 'We gathered around the table for dinner.',
            'synonym': 'desk',
            'translations': {'es': 'mesa', 'fr': 'table', 'it': 'tavolo', 'pt': 'mesa', 'ru': 'стол', 'zh-CN': '桌子'}
        },
        {
            'word': 'food',
            'definition': 'Any nutritious substance that people eat or drink to maintain life and growth.',
            'example': 'The refrigerator was stocked with food.',
            'synonym': 'nourishment',
            'translations': {'es': 'comida', 'fr': 'nourriture', 'it': 'cibo', 'pt': 'comida', 'ru': 'еда', 'zh-CN': '食物'}
        },
        {
            'word': 'friend',
            'definition': 'A person with whom one has a bond of mutual affection.',
            'example': 'She has been my friend since childhood.',
            'synonym': 'companion',
            'translations': {'es': 'amigo', 'fr': 'ami', 'it': 'amico', 'pt': 'amigo', 'ru': 'друг', 'zh-CN': '朋友'}
        }
    ],
    'intermediate': [
        {
            'word': 'journey',
            'definition': 'An act of traveling from one place to another, especially over a long distance.',
            'example': 'The journey across the desert took several days.',
            'synonym': 'voyage',
            'translations': {'es': 'viaje', 'fr': 'voyage', 'it': 'viaggio', 'pt': 'jornada', 'ru': 'путешествие', 'zh-CN': '旅程'}
        },
        {
            'word': 'accomplish',
            'definition': 'Achieve or complete successfully.',
            'example': 'She accomplished all her goals for the project.',
            'synonym': 'achieve',
            'translations': {'es': 'lograr', 'fr': 'accomplir', 'it': 'realizzare', 'pt': 'realizar', 'ru': 'выполнить', 'zh-CN': '完成'}
        },
        {
            'word': 'determine',
            'definition': 'Cause (something) to occur in a particular way or to have a particular nature.',
            'example': 'The results will determine the next steps we take.',
            'synonym': 'decide',
            'translations': {'es': 'determinar', 'fr': 'déterminer', 'it': 'determinare', 'pt': 'determinar', 'ru': 'определять', 'zh-CN': '决定'}
        },
        {
            'word': 'necessary',
            'definition': 'Required to be done, achieved, or present; needed; essential.',
            'example': 'It\'s necessary to wear protective clothing.',
            'synonym': 'essential',
            'translations': {'es': 'necesario', 'fr': 'nécessaire', 'it': 'necessario', 'pt': 'necessário', 'ru': 'необходимый', 'zh-CN': '必要的'}
        },
        {
            'word': 'solution',
            'definition': 'A means of solving a problem or dealing with a difficult situation.',
            'example': 'There is no simple solution to this problem.',
            'synonym': 'answer',
            'translations': {'es': 'solución', 'fr': 'solution', 'it': 'soluzione', 'pt': 'solução', 'ru': 'решение', 'zh-CN': '解决方案'}
        },
        {
            'word': 'provide',
            'definition': 'Make available for use; supply.',
            'example': 'The hotel provides breakfast for its guests.',
            'synonym': 'supply',
            'translations': {'es': 'proporcionar', 'fr': 'fournir', 'it': 'fornire', 'pt': 'fornecer', 'ru': 'предоставлять', 'zh-CN': '提供'}
        },
        {
            'word': 'consider',
            'definition': 'Think carefully about something, typically before making a decision.',
            'example': 'I\'m considering a career change.',
            'synonym': 'contemplate',
            'translations': {'es': 'considerar', 'fr': 'considérer', 'it': 'considerare', 'pt': 'considerar', 'ru': 'рассматривать', 'zh-CN': '考虑'}
        },
        {
            'word': 'improve',
            'definition': 'Make or become better.',
            'example': 'His health has improved in recent months.',
            'synonym': 'enhance',
            'translations': {'es': 'mejorar', 'fr': 'améliorer', 'it': 'migliorare', 'pt': 'melhorar', 'ru': 'улучшать', 'zh-CN': '改进'}
        },
        {
            'word': 'significant',
            'definition': 'Sufficiently great or important to be worthy of attention; notable.',
            'example': 'There has been a significant increase in crime in the area.',
            'synonym': 'important',
            'translations': {'es': 'significativo', 'fr': 'significatif', 'it': 'significativo', 'pt': 'significativo', 'ru': 'значительный', 'zh-CN': '重要的'}
        },
        {
            'word': 'challenge',
            'definition': 'A task or situation that tests someone\'s abilities.',
            'example': 'The marathon presented a challenge to all the participants.',
            'synonym': 'test',
            'translations': {'es': 'desafío', 'fr': 'défi', 'it': 'sfida', 'pt': 'desafio', 'ru': 'вызов', 'zh-CN': '挑战'}
        }
    ],
    'advanced': [
        {
            'word': 'eloquent',
            'definition': 'Fluent or persuasive in speaking or writing.',
            'example': 'She gave an eloquent speech that moved the audience.',
            'synonym': 'articulate',
            'translations': {'es': 'elocuente', 'fr': 'éloquent', 'it': 'eloquente', 'pt': 'eloquente', 'ru': 'красноречивый', 'zh-CN': '雄辩的'}
        },
        {
            'word': 'meticulous',
            'definition': 'Showing great attention to detail; very careful and precise.',
            'example': 'He was meticulous in his research for the project.',
            'synonym': 'thorough',
            'translations': {'es': 'meticuloso', 'fr': 'méticuleux', 'it': 'meticoloso', 'pt': 'meticuloso', 'ru': 'тщательный', 'zh-CN': '一丝不苟的'}
        },
        {
            'word': 'ephemeral',
            'definition': 'Lasting for a very short time.',
            'example': 'The beauty of cherry blossoms is ephemeral.',
            'synonym': 'transient',
            'translations': {'es': 'efímero', 'fr': 'éphémère', 'it': 'effimero', 'pt': 'efêmero', 'ru': 'мимолетный', 'zh-CN': '短暂的'}
        },
        {
            'word': 'ubiquitous',
            'definition': 'Present, appearing, or found everywhere.',
            'example': 'Mobile phones have become ubiquitous in modern society.',
            'synonym': 'omnipresent',
            'translations': {'es': 'ubicuo', 'fr': 'omniprésent', 'it': 'onnipresente', 'pt': 'ubíquo', 'ru': 'вездесущий', 'zh-CN': '无所不在的'}
        },
        {
            'word': 'profound',
            'definition': 'Very great or intense; having or showing great knowledge or insight.',
            'example': 'The book had a profound influence on her thinking.',
            'synonym': 'deep',
            'translations': {'es': 'profundo', 'fr': 'profond', 'it': 'profondo', 'pt': 'profundo', 'ru': 'глубокий', 'zh-CN': '深刻的'}
        },
        {
            'word': 'ambiguous',
            'definition': 'Open to more than one interpretation; not having one obvious meaning.',
            'example': 'The statement was intentionally ambiguous to avoid controversy.',
            'synonym': 'unclear',
            'translations': {'es': 'ambiguo', 'fr': 'ambigu', 'it': 'ambiguo', 'pt': 'ambíguo', 'ru': 'двусмысленный', 'zh-CN': '模棱两可的'}
        },
        {
            'word': 'quintessential',
            'definition': 'Representing the most perfect or typical example of a quality or class.',
            'example': 'He was the quintessential English gentleman.',
            'synonym': 'archetypal',
            'translations': {'es': 'esencial', 'fr': 'quintessence', 'it': 'quintessenziale', 'pt': 'quintessencial', 'ru': 'типичный', 'zh-CN': '典型的'}
        },
        {
            'word': 'altruistic',
            'definition': 'Showing a disinterested and selfless concern for the well-being of others.',
            'example': 'Her altruistic nature led her to volunteer at homeless shelters.',
            'synonym': 'selfless',
            'translations': {'es': 'altruista', 'fr': 'altruiste', 'it': 'altruistico', 'pt': 'altruísta', 'ru': 'альтруистический', 'zh-CN': '利他的'}
        },
        {
            'word': 'pragmatic',
            'definition': 'Dealing with things sensibly and realistically in a way that is based on practical considerations.',
            'example': 'We need a pragmatic approach to solving this issue.',
            'synonym': 'practical',
            'translations': {'es': 'pragmático', 'fr': 'pragmatique', 'it': 'pragmatico', 'pt': 'pragmático', 'ru': 'прагматичный', 'zh-CN': '务实的'}
        },
        {
            'word': 'paradoxical',
            'definition': 'Seemingly absurd or contradictory, but perhaps true.',
            'example': 'It\'s paradoxical that increasing one\'s leisure time can make one less happy.',
            'synonym': 'contradictory',
            'translations': {'es': 'paradójico', 'fr': 'paradoxal', 'it': 'paradossale', 'pt': 'paradoxal', 'ru': 'парадоксальный', 'zh-CN': '矛盾的'}
        }
    ]
}


def get_vocab_by_level(level='beginner'):
    """Get vocabulary words by difficulty level."""
    return VOCABULARY.get(level, VOCABULARY['beginner'])


def get_daily_word(level='beginner', selected_langs=None, force_ai=True):
    """
    Get a random word from the level for 'word of the day'.
    Now using Claude AI to generate a fresh word each time by default.
    
    Args:
        level (str): The difficulty level - beginner, intermediate, or advanced
        selected_langs (list): List of language codes the user has selected
        force_ai (bool): If True, will only use Claude AI and never fallback to static words
        
    Returns:
        dict: A word data dictionary with word, definition, example, etc.
    """
    from ai_services_simplified import query_claude, is_ai_service_available
    import json
    
    # If no languages specified, use default set
    if not selected_langs:
        selected_langs = ['es', 'fr', 'it', 'pt', 'ru', 'zh-CN']
    
    # Use Anthropic Claude to generate a random word
    if is_ai_service_available():
        try:
            # Create language list for prompt
            lang_list = ", ".join([f"{lang}" for lang in selected_langs])
            
            # Use current timestamp to ensure we get different words each time
            import time, random
            current_time = int(time.time())
            random_seed = random.randint(1000, 9999)
            
            prompt = f"""Please suggest a completely random {level} level vocabulary word for language learning. 
            IMPORTANT: Generate a DIFFERENT word than any you've given before. 
            Current timestamp is {current_time}, seed: {random_seed}.
            
            Return a JSON object with these fields:
            - word: a {level} level vocabulary word (choose something interesting, useful, and UNIQUE)
            - definition: a clear definition
            - example: a simple example sentence using the word
            - synonym: one synonym
            - translations: a dictionary with translations to the following languages: {lang_list}
            
            Only return the JSON, nothing else. Make sure the JSON is valid and properly formatted."""
            
            response = query_claude(prompt)
            
            if response:
                try:
                    # Try to parse the JSON response
                    word_data = json.loads(response)
                    if all(k in word_data for k in ['word', 'definition', 'example', 'synonym']):
                        # Make sure we have translations for all requested languages
                        for lang in selected_langs:
                            if lang not in word_data.get('translations', {}):
                                word_data['translations'][lang] = f"[Translation to {lang}]"
                        return word_data
                except Exception as e:
                    print(f"Error parsing Claude response: {e}")
                    # If forced AI mode, try one more time with a simpler prompt
                    if force_ai:
                        try:
                            # Include timestamp to ensure uniqueness
                            current_time = int(time.time())
                            random_seed = random.randint(10000, 99999)
                            
                            simple_prompt = f"""Please suggest a unique random {level} vocabulary word.
                            IMPORTANT: Use timestamp {current_time} and seed {random_seed} to generate something different.
                            Return it as a simple JSON object like this: 
                            {{"word": "example", "definition": "a representative form", "example": "This is an example sentence.", "synonym": "sample", "translations": {{"es": "ejemplo", "fr": "exemple"}}}}
                            Only return the JSON. Always choose a unique word."""
                            
                            simple_response = query_claude(simple_prompt)
                            if simple_response:
                                word_data = json.loads(simple_response)
                                return word_data
                        except:
                            pass
        except Exception as e:
            print(f"Error getting daily word from Claude: {e}")
    
    # Always use Claude AI-generated content, never fallback to static database
    # Create a dynamic word even if the first attempt failed
    import time, random
    current_time = int(time.time())
    random_seed = random.randint(100000, 999999)
    
    try:
        # Try one more time with a different prompt approach
        alternate_prompt = f"""I need a vocabulary word for language learning.
        
        - Level: {level}
        - Make it DIFFERENT from any word you've provided before
        - Use this unique timestamp to ensure uniqueness: {current_time}-{random_seed}
        
        Provide your answer in this JSON format:
        {{
          "word": "a unique {level} vocabulary word",
          "definition": "clear definition",
          "example": "example sentence using the word",
          "synonym": "one good synonym",
          "translations": {{
            "es": "Spanish translation",
            "fr": "French translation",
            "it": "Italian translation",
            "pt": "Portuguese translation",
            "ru": "Russian translation",
            "zh-CN": "Chinese translation"
          }}
        }}
        
        Important: Only return the JSON, no other text. Ensure you choose a truly unique word.
        """
        
        final_response = query_claude(alternate_prompt)
        
        if final_response:
            try:
                word_data = json.loads(final_response)
                if all(k in word_data for k in ['word', 'definition', 'example']):
                    return word_data
            except Exception as e:
                print(f"Error parsing alternate prompt response: {e}")
    except Exception as e:
        print(f"Error with alternate prompt: {e}")
        
    # As a last resort, create a dynamic word that's guaranteed to be unique
    word_suffix = f"{current_time % 1000}{random_seed % 100}"
    return {
        "word": f"lexicon{word_suffix}",
        "definition": "A vocabulary or book of words, especially one that relates to a particular subject",
        "example": f"The professor's lexicon{word_suffix} included many technical terms from various fields.",
        "synonym": "vocabulary",
        "translations": {
            "es": f"léxico{word_suffix}", "fr": f"lexique{word_suffix}", "it": f"lessico{word_suffix}", 
            "pt": f"léxico{word_suffix}", "ru": f"лексикон{word_suffix}", "zh-CN": f"词典{word_suffix}"
        }
    }


def generate_thematic_vocabulary(theme: str, lang_code: str, level: str = 'intermediate', count: int = 10) -> List[Dict[str, Any]]:
    """
    Generate thematic vocabulary list using AI.
    
    Args:
        theme (str): Theme for vocabulary (e.g., "travel", "food", "business")
        lang_code (str): Language code
        level (str): Learning level (beginner, intermediate, advanced)
        count (int): Number of words to generate
        
    Returns:
        list: List of vocabulary items with word, translation, and example
    """
    # Default empty list
    default_response = []
    
    # If the theme is general, return some predefined vocabulary for the matching level
    if theme.lower() == "general":
        if level in VOCABULARY:
            # Return a subset of the predefined vocabulary
            selected_items = random.sample(VOCABULARY[level], min(count, len(VOCABULARY[level])))
            return [
                {
                    "word": item['word'],
                    "translation": item.get('translations', {}).get(lang_code, ""),
                    "definition": item['definition'],
                    "example": item['example']
                } for item in selected_items
            ]
    
    # Use AI to generate vocabulary if available
    try:
        prompt = (
            f"Create a list of {count} {level}-level vocabulary words related to '{theme}' in {lang_code} language.\n"
            f"For each word, include the word itself, English translation, and a simple example sentence.\n"
            f"Format as a JSON array with objects containing 'word', 'translation', and 'example' keys."
        )
        
        response = query_claude(prompt)
        
        if response:
            # Extract the JSON from the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                logger.info(f"Used AI for generating thematic vocabulary for '{theme}'")
                return result
            
            # Try to load the entire response as JSON
            try:
                result = json.loads(response)
                if isinstance(result, list):
                    return result
                if isinstance(result, dict) and 'words' in result:
                    return result['words']
            except:
                pass
    except Exception as e:
        logger.error(f"Error generating thematic vocabulary: {e}")
    
    # Fallback: return an empty list or minimal default vocabulary
    if level in VOCABULARY:
        # Return a subset of the predefined vocabulary
        selected_items = random.sample(VOCABULARY[level], min(5, len(VOCABULARY[level])))
        return [
            {
                "word": item['word'],
                "translation": item.get('translations', {}).get(lang_code, ""),
                "definition": item['definition'],
                "example": item['example']
            } for item in selected_items
        ]
    
    return default_response

def get_word_options(correct_word, level='beginner', num_options=4):
    """Get multiple choice options for a word game - now generating AI-based options."""
    from ai_services_simplified import query_claude
    import json, time, random
    
    # Include the correct word
    options = [correct_word]
    
    # Try to get distractor options using Claude AI
    try:
        # Create timestamp and seed for uniqueness
        timestamp = int(time.time())
        random_seed = random.randint(1000, 9999)
        
        # Get alternative words of similar difficulty but different meanings
        prompt = f"""I need {num_options - 1} vocabulary words at the {level} level to use as distractors in a multiple-choice quiz.
        The correct answer is "{correct_word['word']}" meaning "{correct_word['definition']}".
        
        Please provide {num_options - 1} different words that are also {level} level but have DIFFERENT meanings.
        Include for each: word, definition, example sentence, and a synonym.
        
        Return as JSON array with each object containing: word, definition, example, synonym.
        Make each word DIFFERENT from "{correct_word['word']}" and from each other.
        
        Timestamp for uniqueness: {timestamp}-{random_seed}.
        Only return the JSON array, nothing else."""
        
        response = query_claude(prompt)
        
        if response:
            try:
                distractors = json.loads(response)
                if isinstance(distractors, list) and len(distractors) > 0:
                    # Add the distractors to our options
                    for word in distractors[:num_options-1]:  # Limit to needed number
                        if 'word' in word and 'definition' in word:
                            # Add placeholder translations
                            if 'translations' not in word:
                                word['translations'] = {}
                            options.append(word)
            except Exception as e:
                print(f"Error parsing AI-generated distractors: {e}")
    except Exception as e:
        print(f"Error generating word options with AI: {e}")
    
    # If we still don't have enough options, create some generic ones
    while len(options) < num_options:
        index = len(options)
        fake_word = f"option{index}_{int(time.time()) % 1000}"
        options.append({
            'word': fake_word,
            'definition': f"Alternative vocabulary option {index}",
            'example': f"This is an example using {fake_word}.",
            'synonym': f"alternate{index}",
            'translations': {}
        })
    
    # Shuffle the options
    random.shuffle(options)
    
    return options


# Translation challenge pairs for beginner level
TRANSLATION_CHALLENGES = {
    'beginner': [
        {'en': 'Good morning', 'es': 'Buenos días', 'fr': 'Bonjour', 'it': 'Buongiorno', 'pt': 'Bom dia', 'ru': 'Доброе утро', 'zh-CN': '早上好'},
        {'en': 'My name is John', 'es': 'Mi nombre es John', 'fr': 'Je m\'appelle John', 'it': 'Il mio nome è John', 'pt': 'Meu nome é John', 'ru': 'Меня зовут Джон', 'zh-CN': '我的名字是约翰'},
        {'en': 'How are you?', 'es': '¿Cómo estás?', 'fr': 'Comment allez-vous?', 'it': 'Come stai?', 'pt': 'Como vai você?', 'ru': 'Как дела?', 'zh-CN': '你好吗？'},
        {'en': 'I like to read books', 'es': 'Me gusta leer libros', 'fr': 'J\'aime lire des livres', 'it': 'Mi piace leggere libri', 'pt': 'Eu gosto de ler livros', 'ru': 'Я люблю читать книги', 'zh-CN': '我喜欢读书'},
        {'en': 'The weather is nice today', 'es': 'El clima está agradable hoy', 'fr': 'Le temps est agréable aujourd\'hui', 'it': 'Il tempo è bello oggi', 'pt': 'O tempo está bom hoje', 'ru': 'Сегодня хорошая погода', 'zh-CN': '今天天气很好'}
    ],
    'intermediate': [
        {'en': 'I would like to improve my language skills', 'es': 'Me gustaría mejorar mis habilidades lingüísticas', 'fr': 'Je voudrais améliorer mes compétences linguistiques', 'it': 'Vorrei migliorare le mie competenze linguistiche', 'pt': 'Eu gostaria de melhorar minhas habilidades linguísticas', 'ru': 'Я хотел бы улучшить свои языковые навыки', 'zh-CN': '我想提高我的语言技能'},
        {'en': 'The project requires careful planning', 'es': 'El proyecto requiere una planificación cuidadosa', 'fr': 'Le projet nécessite une planification minutieuse', 'it': 'Il progetto richiede un\'attenta pianificazione', 'pt': 'O projeto requer um planejamento cuidadoso', 'ru': 'Проект требует тщательного планирования', 'zh-CN': '该项目需要仔细规划'},
        {'en': 'We should consider all possible solutions', 'es': 'Deberíamos considerar todas las soluciones posibles', 'fr': 'Nous devrions considérer toutes les solutions possibles', 'it': 'Dovremmo considerare tutte le possibili soluzioni', 'pt': 'Devemos considerar todas as soluções possíveis', 'ru': 'Мы должны рассмотреть все возможные решения', 'zh-CN': '我们应该考虑所有可能的解决方案'},
        {'en': 'Learning a language takes time and practice', 'es': 'Aprender un idioma lleva tiempo y práctica', 'fr': 'Apprendre une langue prend du temps et de la pratique', 'it': 'Imparare una lingua richiede tempo e pratica', 'pt': 'Aprender um idioma leva tempo e prática', 'ru': 'Изучение языка требует времени и практики', 'zh-CN': '学习语言需要时间和练习'},
        {'en': 'The results show significant improvement', 'es': 'Los resultados muestran una mejora significativa', 'fr': 'Les résultats montrent une amélioration significative', 'it': 'I risultati mostrano un miglioramento significativo', 'pt': 'Os resultados mostram uma melhoria significativa', 'ru': 'Результаты показывают значительное улучшение', 'zh-CN': '结果显示显著改善'}
    ],
    'advanced': [
        {'en': 'The paradoxical nature of the situation left everyone perplexed', 'es': 'La naturaleza paradójica de la situación dejó a todos perplejos', 'fr': 'La nature paradoxale de la situation a laissé tout le monde perplexe', 'it': 'La natura paradossale della situazione ha lasciato tutti perplessi', 'pt': 'A natureza paradoxal da situação deixou todos perplexos', 'ru': 'Парадоксальная природа ситуации оставила всех в недоумении', 'zh-CN': '情况的矛盾性质让所有人都感到困惑'},
        {'en': 'The ephemeral beauty of the moment was not lost on the audience', 'es': 'La belleza efímera del momento no pasó desapercibida para el público', 'fr': 'La beauté éphémère du moment n\'a pas échappé au public', 'it': 'La bellezza effimera del momento non è passata inosservata al pubblico', 'pt': 'A beleza efêmera do momento não passou despercebida ao público', 'ru': 'Мимолетная красота момента не ускользнула от внимания публики', 'zh-CN': '观众没有错过瞬间的短暂美丽'},
        {'en': 'The meticulous attention to detail was evident in every aspect of the work', 'es': 'La meticulosa atención al detalle era evidente en todos los aspectos del trabajo', 'fr': 'L\'attention méticuleuse aux détails était évidente dans tous les aspects du travail', 'it': 'L\'attenzione meticolosa ai dettagli era evidente in ogni aspetto del lavoro', 'pt': 'A atenção meticulosa aos detalhes era evidente em todos os aspectos do trabalho', 'ru': 'Тщательное внимание к деталям было очевидно в каждом аспекте работы', 'zh-CN': '对细节的一丝不苟的关注在工作的各个方面都很明显'},
        {'en': 'Her eloquent speech captivated the audience and conveyed the message with remarkable clarity', 'es': 'Su elocuente discurso cautivó al público y transmitió el mensaje con notable claridad', 'fr': 'Son discours éloquent a captivé l\'auditoire et a transmis le message avec une clarté remarquable', 'it': 'Il suo discorso eloquente ha catturato l\'attenzione del pubblico e ha trasmesso il messaggio con notevole chiarezza', 'pt': 'Seu discurso eloquente cativou o público e transmitiu a mensagem com notável clareza', 'ru': 'Ее красноречивая речь захватила аудиторию и передала сообщение с замечательной ясностью', 'zh-CN': '她雄辩的演讲吸引了观众，并以非凡的清晰度传达了信息'},
        {'en': 'The ubiquitous nature of smartphones has transformed how we communicate and access information', 'es': 'La naturaleza ubicua de los teléfonos inteligentes ha transformado la forma en que nos comunicamos y accedemos a la información', 'fr': 'La nature omniprésente des smartphones a transformé notre façon de communiquer et d\'accéder à l\'information', 'it': 'La natura onnipresente degli smartphone ha trasformato il modo in cui comunichiamo e accediamo alle informazioni', 'pt': 'A natureza ubíqua dos smartphones transformou a forma como nos comunicamos e acessamos informações', 'ru': 'Вездесущий характер смартфонов изменил то, как мы общаемся и получаем доступ к информации', 'zh-CN': '智能手机的无处不在的特性已经改变了我们交流和获取信息的方式'}
    ]
}

def get_translation_challenge(level='beginner'):
    """Get a random translation challenge for the specified level using AI."""
    from ai_services_simplified import query_claude
    import json, time, random
    
    # Create timestamp to ensure uniqueness - define here so it's available throughout
    timestamp = int(time.time())
    random_seed = random.randint(10000, 99999)
    
    # Try to generate a dynamic translation challenge using Claude AI
    try:
        prompt = f"""Create a unique translation challenge at the {level} difficulty level.
        
        Please create a short phrase or sentence in English that would be appropriate for a {level} language learner. Then provide translations for this phrase into multiple languages.
        
        Return your response as a JSON object with language codes as keys and the translations as values. Include these languages:
        - 'en': [English original]
        - 'es': [Spanish translation]
        - 'fr': [French translation]
        - 'it': [Italian translation]
        - 'pt': [Portuguese translation]
        - 'ru': [Russian translation]
        - 'zh-CN': [Chinese Mandarin translation]
        
        Make sure all translations are accurate. Use this timestamp for uniqueness: {timestamp}-{random_seed}.
        Return only valid JSON, with no other text."""
        
        response = query_claude(prompt)
        if response:
            try:
                # Parse the JSON response
                challenge = json.loads(response)
                # Ensure it has at least English and a few other languages
                if challenge and 'en' in challenge and len(challenge) >= 3:
                    return challenge
            except Exception as e:
                print(f"Error parsing AI translation challenge: {e}")
    except Exception as e:
        print(f"Error generating AI translation challenge: {e}")
    
    # Fall back to predefined challenges only if AI fails
    # But first generate one more time with a simpler prompt
    try:
        # Simpler prompt
        simple_prompt = f"""Create a simple phrase in English and translate it to multiple languages.
        Format it as JSON like:
        {{
          "en": "Hello, how are you?",
          "es": "Hola, ¿cómo estás?",
          "fr": "Bonjour, comment ça va?",
          "it": "Ciao, come stai?",
          "pt": "Olá, como vai você?",
          "ru": "Привет, как дела?",
          "zh-CN": "你好，你好吗？"
        }}
        Use timestamp {timestamp} to ensure uniqueness. Only return valid JSON."""
        
        simple_response = query_claude(simple_prompt)
        if simple_response:
            try:
                challenge = json.loads(simple_response)
                if challenge and 'en' in challenge:
                    return challenge
            except Exception as e:
                print(f"Error parsing simple translation challenge: {e}")
    except Exception as e:
        print(f"Error with simple translation challenge: {e}")
    
    # Last resort: use the predefined challenges
    challenges = TRANSLATION_CHALLENGES.get(level, TRANSLATION_CHALLENGES['beginner'])
    if challenges:
        return random.choice(challenges)
    
    # If all else fails, create a minimal challenge with timestamp to ensure uniqueness
    time_suffix = f"{timestamp % 100}"
    return {
        'en': f"This is a simple phrase {time_suffix}.",
        'es': f"Esta es una frase simple {time_suffix}.",
        'fr': f"C'est une phrase simple {time_suffix}.",
        'it': f"Questa è una frase semplice {time_suffix}.",
        'pt': f"Esta é uma frase simples {time_suffix}.",
        'ru': f"Это простая фраза {time_suffix}.",
        'zh-CN': f"这是一个简单的短语 {time_suffix}。"
    }