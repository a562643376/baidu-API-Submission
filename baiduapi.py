import json
import requests
import os

config_file = 'baiduapi-config.json'

def save_config(api_url):
    try:
        site = api_url.split('site=')[1].split('&')[0]
        token = api_url.split('token=')[1]
        config = {
            'site': site,
            'token': token
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f)
        print(f"配置已保存到 {config_file}")
    except Exception as e:
        print(f"解析API URL时出错: {e}")

def load_config():
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def submit_urls(urls, site, token):
    api_url = f"http://data.zz.baidu.com/urls?site={site}&token={token}"
    headers = {
        'Content-Type': 'text/plain'
    }
    response = requests.post(api_url, headers=headers, data='\n'.join(urls))
    if response.status_code == 200:
        print("提交成功")
    else:
        print(f"提交失败: {response.text}")

def main():
    config = load_config()
    if not config:
        api_url = input("请输入你的百度API链接: ")
        save_config(api_url)
        config = load_config()
    
    if config:
        site = config.get('site')
        token = config.get('token')

        while True:
            choice = input("请选择手动输入(1)还是读取sitemap.txt(2): ")
            if choice == '1':
                url = input("请输入你需要提交的链接（需要带http(s)协议头）: ")
                submit_urls([url], site, token)
            elif choice == '2':
                file_path = 'sitemap.txt'
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        urls = f.read().splitlines()
                    print(f"在{file_path}中一共找到{len(urls)}条连接")
                    for idx, url in enumerate(urls, start=1):
                        print(f"正在提交第{idx}条：{url}")
                        submit_urls([url], site, token)
                else:
                    print(f"文件{file_path}不存在")
            else:
                print("无效的选择")
            
            continue_choice = input("是否继续提交（y/n）: ")
            if continue_choice.lower() != 'y':
                print("正在退出，更多seo服务请咨询微信号：liangding562643376")
                break
    else:
        print("配置加载失败，请检查API URL格式")

if __name__ == "__main__":
    main()
