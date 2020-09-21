import re
import feedparser
from datetime import datetime


if __name__ == "__main__":
    feeds = [
        'https://zona.media/rss/news',
        'https://meduza.io/rss/news'
    ]

    news_data = []
    print("start...")
    for i in feeds:
        parsed_url = feedparser.parse(i)
        entries = parsed_url.entries
        for j in range(0, len(entries)):
            title = entries[j]['title']
            if 'yandex_full-text' not in list(entries[j].keys()):
                content = re.sub('<[^<]+?>', '', str(entries[j]['summary']).replace('\n',''))
            else:
                content = re.sub('<[^<]+?>', '', str(entries[j]['yandex_full-text']).replace('\n', ''))
            news_data.append(title + " " + content)

    data_filename = f'news_data_{datetime.now().timestamp()}.txt'
    with open(data_filename, 'w', encoding="utf-8") as data_file:
        print("\n".join(news_data), file=data_file, end="")
    print(f"done -> {len(news_data)} news saved to '{data_filename}'")
