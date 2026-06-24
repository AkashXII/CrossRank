import json
#note these are the translated 33 queries from the selected 50k documents which was used in the original mr tyDi dataset. as original is japanese->japanese.
queries = [
    {"query_id": "419", "text": "When was Odakyu Electric Railway established?"},
    {"query_id": "534", "text": "What was Nobuhiro Watsuki's debut work?"},
    {"query_id": "648", "text": "How many countries are there on the European continent?"},
    {"query_id": "741", "text": "What period did Mozart live in?"},
    {"query_id": "842", "text": "How many countries are there in Europe?"},
    {"query_id": "911", "text": "Which company was the first to release a digital camera?"},
    {"query_id": "1015", "text": "What is the capital of Switzerland?"},
    {"query_id": "1054", "text": "What academic field studies life history?"},
    {"query_id": "1059", "text": "Is Kyoto Imperial University the predecessor of Kyoto University?"},
    {"query_id": "1205", "text": "Where was Mencius born?"},
    {"query_id": "1758", "text": "What is the etymology of the word occult?"},
    {"query_id": "1802", "text": "What was Masahiro Shibata's debut work?"},
    {"query_id": "1822", "text": "What was Masami Kurumada's debut work?"},
    {"query_id": "2017", "text": "What is the gender of Kaoru Shintani?"},
    {"query_id": "2123", "text": "When did plastic models originate?"},
    {"query_id": "2210", "text": "What is the operating system of Apple Computer?"},
    {"query_id": "2256", "text": "How many countries are members of the European Parliament?"},
    {"query_id": "2280", "text": "What is the capital of Paraguay?"},
    {"query_id": "2452", "text": "Who is the founder of GNU?"},
    {"query_id": "2604", "text": "What language did English derive from?"},
    {"query_id": "2713", "text": "When did the World Rally Championship start?"},
    {"query_id": "2898", "text": "What was Kaoru Shintani's debut work?"},
    {"query_id": "2943", "text": "When did rock music originate?"},
    {"query_id": "3451", "text": "Where is Hiroshima?"},
    {"query_id": "3466", "text": "Who is the author of the robot anime Astro Boy?"},
    {"query_id": "3488", "text": "Who created the Julian calendar?"},
    {"query_id": "3571", "text": "What is the area of Italy?"},
    {"query_id": "3818", "text": "How many countries are in the European Union?"},
    {"query_id": "3841", "text": "The Emmy Award is for TV programs, what is the biggest film award in America?"},
    {"query_id": "3896", "text": "What was the first TV period drama produced in Japan?"},
    {"query_id": "4061", "text": "When was the digital camera developed?"},
    {"query_id": "4165", "text": "What was Rumiko Takahashi's debut work?"},
    {"query_id": "4589", "text": "When did musical films originate?"},
]

with open("index/english_queries.json", "w") as f:
    json.dump(queries, f, indent=2)

print(f"Saved {len(queries)} queries.")