import requests
import re
from bs4 import BeautifulSoup

# URLを設定
BASE_URL = 'https://faq01.bk.mufg.jp'

def get_question_links(category_url):
    # Get the HTML content of the category page
    response = requests.get(category_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all question links
    category_lists = soup.find_all(class_='cate_small clearfix')
    category_paths = []
    for category_list in category_lists:
        for category in category_list.find_all('a', href=True):
            category_paths.append(re.sub(r'\?site_domain=default', '', category['href']))
    return category_paths

def get_qa(question_url):
    # URLにGETリクエストを送信
    response = requests.get(BASE_URL + question_url)
    qa_soup = BeautifulSoup(response.content, 'html.parser')

    # 回答のタイトルの直後にある要素を調べる
    answer_title = qa_soup.find('h2', class_='faq_ans_bor faq_ans_ttl')
    answer_content = answer_title.find_next_sibling()

    # 質問を抽出
    question = qa_soup.find('h2', class_='faq_qstCont_ttl').get_text(strip=True)

    # 回答の内容を抽出（重複を避ける）
    answer_parts = []
    last_text = ""
    for part in answer_content.descendants:
        if part.name == 'a':
            # リンクの場合は、テキストとURLを取得し、フォーマットを適用
            link_text = part.get_text(strip=True)
            link_url = part['href']
            link_with_url = f"{link_text}({link_url})"
            answer_parts.append(link_with_url)
            last_text = link_text
        elif part.string and part.string.strip():
            current_text = part.string.strip()
            if current_text != last_text:
                # 重複していないテキストのみを追加
                answer_parts.append(current_text)
                last_text = current_text

    # 回答の内容を抽出
    answer_content_text = ' '.join(answer_parts)

    def has_border_left(style):
        return style and "border-left: 3px solid" in style
    border_left_headings = qa_soup.find_all(style=has_border_left)
    for heading in border_left_headings:
        heading_text = heading.get_text(strip=True)
        # 見出しテキストを "##見出しテキスト" の形式に変換
        answer_content_text = answer_content_text.replace(heading_text, f"# {heading_text}")

    accordion_titles = qa_soup.find_all(class_='accordion-title')
    for title in accordion_titles:
        title_text = title.get_text(strip=True)
        # 見出しテキストを "## 見出しテキスト" の形式に変換
        answer_content_text = answer_content_text.replace(title_text, f"## {title_text}")

    print("質問:" + question +" 回答:" +answer_content_text)
    return "質問:" + question +" 回答:" +answer_content_text


category_paths = get_question_links(BASE_URL)

with open('output.txt', 'w') as file:
    for category_path in category_paths:
        print("processing: " + category_path)
        CATEGORY_PATH_URL = "{category_path}?page={page}&site_domain=default&sort=sort_access&sort_order=desc"
        for i in range(1, 100):
            Q_LIST_URL = BASE_URL + CATEGORY_PATH_URL.format(category_path=category_path, page=i)
            response = requests.get(Q_LIST_URL)
            soup = BeautifulSoup(response.content, 'html.parser')
            if not soup.find_all(class_="pagination"):
                print("Done: "+ Q_LIST_URL)
                break

            print("processing: " + CATEGORY_PATH_URL.format(category_path=category_path, page=i))
            soup = BeautifulSoup(response.content, 'html.parser')
            questions = soup.find_all(class_='search_qattl icoQ_sml')
            for question in questions:
                question_path = question.find('a', href=True)['href']
                print("processing: " + question_path)
                try:
                    file.write(get_qa(question_path) + '\n')
                except Exception as e: 
                    print("Error: " + question_path + " " + str(e))
                    continue
            print("Done: "+ Q_LIST_URL)
        print("Done: "+ category_path)
