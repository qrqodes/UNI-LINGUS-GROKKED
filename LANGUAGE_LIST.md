# Supported Languages

The Enhanced Language Learning and Translation Bot now supports 16 languages:

| Language Code | Language Name | Flag |
|---------------|---------------|------|
| en            | English       | 🇬🇧    |
| es            | Spanish       | 🇪🇸    |
| fr            | French        | 🇫🇷    |
| it            | Italian       | 🇮🇹    |
| pt            | Portuguese    | 🇵🇹    |
| ru            | Russian       | 🇷🇺    |
| zh-CN         | Chinese       | 🇨🇳    |
| de            | German        | 🇩🇪    |
| ja            | Japanese      | 🇯🇵    |
| ko            | Korean        | 🇰🇷    |
| ar            | Arabic        | 🇸🇦    |
| hi            | Hindi         | 🇮🇳    |
| tr            | Turkish       | 🇹🇷    |
| nl            | Dutch         | 🇳🇱    |
| pl            | Polish        | 🇵🇱    |
| sv            | Swedish       | 🇸🇪    |

## Language Pairs for Learning

The vocabulary learning games support the following language pairs:

| Language Pair | Description |
|---------------|-------------|
| en-es         | English - Spanish |
| en-fr         | English - French |
| en-it         | English - Italian |
| en-pt         | English - Portuguese |
| en-ru         | English - Russian |
| en-zh         | English - Chinese |
| en-de         | English - German |
| en-ja         | English - Japanese |
| en-ko         | English - Korean |

## Adding More Languages

To add support for additional languages:

1. Add the language code and name to the `LANGUAGE_NAMES` dictionary in `language_facts.py`
2. Add the corresponding flag emoji to the `get_flag_emoji()` function in `language_facts.py`
3. Optionally, add language facts and cultural trivia in the respective dictionaries

For example, to add Norwegian:

```python
# In LANGUAGE_NAMES dictionary
'no': 'Norwegian',

# In get_flag_emoji function
'no': '🇳🇴',
```

Note that all languages should be supported by the translator API. Some languages may have limited support for audio pronunciation.