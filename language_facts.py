"""
This module provides language facts and cultural trivia for the translation app's creative playground.
"""
import random

# Language facts for different languages
LANGUAGE_FACTS = {
    'en': [
        "English is the third most spoken native language worldwide, after Mandarin and Spanish.",
        "English has borrowed words from over 350 different languages.",
        "English is the official language of 67 different countries.",
        "The longest word in English without a vowel is 'rhythms'.",
        "The most common letter in English is 'e'.",
        "The oldest English dictionary was written in 1604.",
        "The English language contains the most words of any language, with over one million words.",
        "The phrase 'long time no see' is believed to be a literal translation from a Native American or Chinese phrase.",
        "The shortest complete sentence in English is 'I am.'",
        "English is the official language of the sky - all pilots have to identify themselves and communicate in English."
    ],
    'es': [
        "Spanish is the second most spoken native language in the world, after Mandarin Chinese.",
        "Spanish has two different words for 'to be': 'ser' and 'estar'.",
        "In Spanish, exclamatory sentences are preceded by an inverted exclamation mark (¡).",
        "Spanish is the official language of 21 countries.",
        "The Spanish alphabet used to have 30 letters, but now has 27.",
        "75% of Spanish words end in vowels.",
        "Spanish is one of the world's most phonetic languages - words are pronounced as they are spelled.",
        "There are over 400 million native Spanish speakers worldwide.",
        "The longest Spanish word is 'electroencefalografista' with 23 letters.",
        "Spanish was the diplomatic language of the world from the 17th to the 19th century."
    ],
    'fr': [
        "The French language has 13 vowels, compared to 5 in English.",
        "A single French word can have up to 5 accents (é, è, ê, ë, ç).",
        "French was the official language of England for over 300 years.",
        "French is the second most taught language worldwide after English.",
        "The French language has a special academic institution, the Académie Française, dedicated to preserving its purity.",
        "French has influenced up to a third of English vocabulary.",
        "In the 18th century, French was the language of diplomacy, culture, and art across Europe.",
        "There are approximately 274 million French speakers worldwide.",
        "French is an official language in 29 countries.",
        "The longest French word is 'anticonstitutionnellement' with 25 letters."
    ],
    'it': [
        "Italian is a direct descendant of Latin.",
        "Italian was standardized in the 14th century based on the Tuscan dialect.",
        "Every letter in Italian is pronounced - there are no silent letters.",
        "Italian is the fourth most studied language in the world.",
        "All Italian words end in vowels except for a few borrowed foreign words.",
        "Italian is the closest language to Latin among the Romance languages.",
        "Italian has the least number of letters in its alphabet among European languages, with just 21.",
        "Some English words borrowed from Italian include 'piano', 'pizza', 'opera', and 'cappuccino'.",
        "Italian has over 30 distinct dialects across the country.",
        "The oldest Italian document dates back to 960 AD."
    ],
    'pt': [
        "Portuguese is the sixth most spoken language in the world.",
        "Portuguese is the official language of 9 countries across four continents.",
        "Brazilian Portuguese and European Portuguese have significant differences in pronunciation, vocabulary, and grammar.",
        "Portuguese was spread worldwide during the Age of Discovery in the 15th-16th centuries.",
        "There are approximately 260 million native Portuguese speakers worldwide.",
        "Portuguese has a wealth of unique words without direct translations, like 'saudade' (a deep longing for someone or something absent).",
        "The oldest Portuguese-written document dates back to 1175.",
        "Portuguese uses diacritical marks to change pronunciation and meaning: á, à, â, ã, ç, é, ê, í, ó, ô, õ, ú.",
        "Portuguese was influenced by Arabic during the Moorish occupation of the Iberian Peninsula.",
        "Portuguese is considered one of the Romance languages most distant from Latin in terms of pronunciation."
    ],
    'ru': [
        "Russian has 33 letters in its alphabet, including 10 vowels, 21 consonants, and 2 signs without sound.",
        "The Russian language uses the Cyrillic alphabet.",
        "Russian doesn't have articles (no 'a', 'an', or 'the').",
        "Russian has three genders: masculine, feminine, and neuter.",
        "There are approximately 150 million native Russian speakers worldwide.",
        "Russian is the 8th most spoken language in the world by number of native speakers.",
        "Russian is one of the six official languages of the United Nations.",
        "Russian has a complex system of cases (6) that change the endings of nouns, pronouns, and adjectives.",
        "Russian is considered one of the more difficult languages for English speakers to learn.",
        "The Russian word for 'peace' and 'world' is the same: 'мир' (mir)."
    ],
    'zh-CN': [
        "Mandarin Chinese is the most spoken language in the world, with over 1.1 billion speakers.",
        "Chinese writing uses characters, not an alphabet, with over 50,000 characters (though only about 3,000 are needed for daily use).",
        "Chinese is a tonal language with four different tones in Mandarin, which can change the meaning of a word.",
        "The Chinese writing system has remained largely unchanged for thousands of years.",
        "Chinese characters are ideograms, representing ideas rather than sounds.",
        "Modern Chinese has about 7,000 characters in common use.",
        "Chinese has no verb conjugations or plural forms.",
        "The Chinese language has over 200 dialects.",
        "Each Chinese character fits into a square, regardless of complexity.",
        "Chinese is one of the oldest continuously used writing systems in the world, dating back over 3,000 years."
    ]
}

# Cultural trivia for different countries
CULTURAL_TRIVIA = {
    'en': [
        "In the UK, it's traditional to pull 'Christmas crackers' during Christmas dinner, which contain jokes, paper hats, and small gifts.",
        "The British have a unique fondness for discussing the weather, which serves as a common small talk topic.",
        "In the US, tipping 15-20% at restaurants is customary, unlike many other English-speaking countries.",
        "Afternoon tea is a British tradition dating back to the 1840s and typically includes tea, sandwiches, scones, and cakes.",
        "Americans celebrate Independence Day on July 4th with fireworks, barbecues, and parades.",
        "In England, the traditional Sunday roast dinner brings families together weekly, complete with roasted meat, potatoes, and Yorkshire pudding.",
        "Queuing (standing in line) is taken very seriously in the UK, and cutting in line is considered extremely rude.",
        "Australia Day on January 26th is celebrated with barbecues, concerts, and citizenship ceremonies.",
        "Halloween originated from Celtic harvest festivals, particularly Samhain, and was brought to America by Irish immigrants.",
        "Cricket is passionately followed in many English-speaking countries including England, Australia, and India."
    ],
    'es': [
        "Spain's siesta is a traditional afternoon nap, typically taken after the midday meal.",
        "The running of the bulls (encierro) in Pamplona, Spain is part of the San Fermín festival held each July.",
        "In Spain, Christmas gifts are traditionally exchanged on January 6th (Epiphany), when the Three Kings are said to visit children.",
        "Spain's La Tomatina is an annual festival in which participants throw tomatoes at each other for fun.",
        "Spanish meals are typically eaten later than in other European countries - lunch around 2 PM and dinner after 9 PM.",
        "In many Latin American countries, it's customary to celebrate a girl's 15th birthday (Quinceañera) as a significant coming-of-age event.",
        "The flamenco dance originated in the Andalusia region of Spain, with influences from various cultures.",
        "Mexican Day of the Dead (Día de los Muertos) celebrates and honors deceased loved ones with colorful altars and offerings.",
        "The Spanish tradition of 'tapas' refers to small portions of food served with drinks.",
        "In Spain and Latin America, people typically have two surnames: the first from their father and the second from their mother."
    ],
    'fr': [
        "French people traditionally greet each other with 'la bise' - kissing on the cheeks (the number varies by region from 1 to 4).",
        "Bastille Day (July 14th) commemorates the storming of the Bastille prison and is celebrated with fireworks and parades.",
        "A traditional French dinner consists of several courses: an appetizer, main dish, cheese plate, and dessert.",
        "In France, it's customary to say 'Bon appétit' before starting a meal.",
        "The baguette is so important in French culture that there are laws about its ingredients and baking method.",
        "French cafés are social institutions where people gather to discuss literature, politics, and philosophy.",
        "The Tour de France, one of the world's most famous cycling races, began in 1903.",
        "The Cannes Film Festival is one of the most prestigious film festivals in the world, held annually in May.",
        "In Quebec, Canada, French-speaking communities celebrate Saint-Jean-Baptiste Day on June 24th.",
        "French cuisine was added to UNESCO's list of the world's 'intangible cultural heritage' in 2010."
    ],
    'it': [
        "In Italy, cappuccino is traditionally only consumed in the morning, never after a meal or in the afternoon.",
        "The Italian tradition of 'passeggiata' is an evening walk where people socialize and show themselves in public spaces.",
        "Italy has the most UNESCO World Heritage Sites of any country, with 58 sites.",
        "Italian meals typically include multiple courses: antipasto (appetizer), primo (first course, usually pasta), secondo (main dish), and dolce (dessert).",
        "Opera originated in Italy during the late 16th century.",
        "Venice's Carnival tradition dates back to the 11th century and features elaborate masks and costumes.",
        "The traditional Italian Sunday lunch with family can last for hours, often from midday until late afternoon.",
        "The giving of gifts typically happens on January 6th (Epiphany) in Italy, when La Befana (a witch-like figure) brings presents to children.",
        "Soccer (football) is an integral part of Italian culture, with passionate supporters of teams like Juventus, AC Milan, and Inter Milan.",
        "The tradition of making pizza originated in Naples, Italy, in the late 18th century."
    ],
    'pt': [
        "In Portugal, dinner is typically eaten late, around 8-10 PM.",
        "Fado is a traditional Portuguese music genre characterized by mournful tunes and lyrics, often about the sea or the life of the poor.",
        "Brazilian Carnival is one of the world's most famous festivals, featuring colorful parades, music, and dance.",
        "In Brazil, football (soccer) is more than a sport—it's a national passion, with the country having won the FIFA World Cup five times.",
        "Portuguese pavement (calçada portuguesa) is a traditional style of stone mosaic paving found throughout Portugal and former colonies.",
        "Portuguese cuisine often features bacalhau (dried and salted cod), with claims that there are 365 different ways to prepare it.",
        "In Brazil, New Year's Eve celebrations often include people dressed in white jumping seven waves for good luck.",
        "The Portuguese tradition of 'petiscos' is similar to Spanish tapas - small portions of food to accompany drinks.",
        "The rooster of Barcelos is a popular symbol of Portugal, representing honesty, integrity, and good fortune.",
        "In Brazil, it's customary to serve brigadeiros (chocolate truffles) at birthday celebrations."
    ],
    'ru': [
        "Russians traditionally take off their shoes when entering a home and may offer guests slippers.",
        "The traditional Russian steam bath, banya, is an important social and health ritual.",
        "Russians celebrate New Year's Eve with greater fanfare than Christmas, featuring a visit from 'Ded Moroz' (Grandfather Frost) and his granddaughter 'Snegurochka'.",
        "In Russia, it's considered bad luck to shake hands or give something through a doorway without crossing the threshold.",
        "Maslenitsa is a Russian festival marking the end of winter, during which blini (pancakes) are eaten to symbolize the sun.",
        "Russian culture values hospitality highly, and hosts typically provide abundant food for guests.",
        "The kitchen table is the center of social life in Russian homes, where people gather to eat, drink tea, and engage in long conversations.",
        "May 9th, Victory Day, is one of Russia's most important holidays, commemorating the end of World War II.",
        "Russians don't smile at strangers as much as Westerners, reserving smiles for genuine emotions among friends and family.",
        "Giving an even number of flowers is considered bad luck in Russia, as even numbers are associated with funerals."
    ],
    'zh-CN': [
        "Chinese New Year celebrations last for 15 days and include traditions like giving red envelopes (hongbao) filled with money.",
        "In Chinese culture, the number 8 is considered lucky because its pronunciation sounds similar to the word for prosperity.",
        "Traditional Chinese medicine dates back over 2,500 years and includes practices like acupuncture, herbal medicine, and tai chi.",
        "Chinese calligraphy is considered one of the highest art forms, requiring harmony, rhythm, and balance.",
        "Tea culture is deeply ingrained in Chinese daily life, with numerous types of tea and elaborate tea ceremonies.",
        "The Mid-Autumn Festival celebrates family reunion, featuring mooncakes and lanterns during the full moon.",
        "Gift-giving in China follows specific etiquette, such as avoiding clocks (which symbolize death) and using both hands to present gifts.",
        "The practice of feng shui influences architecture and interior design to create harmony between people and their environment.",
        "Chinese cuisine varies greatly by region, with eight major culinary traditions including Sichuan, Cantonese, and Shanghainese.",
        "Red is the traditional color of good luck in Chinese culture, commonly used in celebrations and decorations."
    ]
}

def get_language_fact(lang_code):
    """Get a random language fact for the specified language."""
    facts = LANGUAGE_FACTS.get(lang_code, LANGUAGE_FACTS['en'])
    return random.choice(facts)

def get_cultural_trivia(lang_code):
    """Get a random cultural trivia fact for the specified language."""
    trivia = CULTURAL_TRIVIA.get(lang_code, CULTURAL_TRIVIA['en'])
    return random.choice(trivia)

def get_flag_emoji(lang_code):
    """Get the flag emoji for a language code."""
    flag_mapping = {
        'en': '🇬🇧',
        'es': '🇪🇸',
        'fr': '🇫🇷',
        'it': '🇮🇹',
        'pt': '🇵🇹',
        'ru': '🇷🇺',
        'zh-CN': '🇨🇳',
        'de': '🇩🇪',
        'ja': '🇯🇵',
        'ko': '🇰🇷',
        'ar': '🇸🇦',
        'hi': '🇮🇳',
        'tr': '🇹🇷',
        'nl': '🇳🇱',
        'pl': '🇵🇱',
        'sv': '🇸🇪',
        'vi': '🇻🇳',
        'th': '🇹🇭',
        'id': '🇮🇩',
        'ms': '🇲🇾',
        'he': '🇮🇱',
        'fa': '🇮🇷',
        'uk': '🇺🇦',
        'cs': '🇨🇿',
        'da': '🇩🇰',
        'fi': '🇫🇮',
        'el': '🇬🇷',
        'hu': '🇭🇺',
        'no': '🇳🇴',
        'ro': '🇷🇴'
    }
    return flag_mapping.get(lang_code, '')

# Language names in English
LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh-CN': 'Chinese',
    'de': 'German',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'tr': 'Turkish',
    'nl': 'Dutch',
    'pl': 'Polish',
    'sv': 'Swedish',
    'vi': 'Vietnamese',
    'th': 'Thai',
    'id': 'Indonesian',
    'ms': 'Malay',
    'he': 'Hebrew',
    'fa': 'Persian',
    'uk': 'Ukrainian',
    'cs': 'Czech',
    'da': 'Danish',
    'fi': 'Finnish',
    'el': 'Greek',
    'hu': 'Hungarian',
    'no': 'Norwegian',
    'ro': 'Romanian'
}

# List of all language codes
ALL_LANGUAGE_CODES = list(LANGUAGE_NAMES.keys())