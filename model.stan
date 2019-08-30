python -m venv scrapy
cd scrapy
activate.bat
pip install Twisted-18.9.0-cp37-cp37m-win32.whl
pip install pywin32
pip install scrapy
.\scrapy startproject u5 
.\u5>scrapy crawl u5 -o 1.xml

.\u5>del 3.xml && scrapy crawl rbt -o 3.xml   >>перезаписать файл вывода

При проблеме с кодировкой установите настройку FEED_EXPORT_ENCODING в settings.py:
FEED_EXPORT_ENCODING = 'utf-8'


pip install openpyxl
https://openpyxl.readthedocs.io/en/stable/usage.html
___________________________________
del 3.xml && del 1.txt && scrapy crawl rbt -o 3.xml -a urlx=http://shop.rosbt.ru/product/krovat-detskaya-zhestkaya-standart-k-yaroslavl --logfile 1.txt --loglevel INFO

del 3.xml && del 1.txt && scrapy crawl rbt -o 3.xml --logfile 1.txt --loglevel INFO
scrapy runspider rbt_spider.py -o 4.xml

pip freeze --local - список зависимостей проекта
pip install -r requirements.txt

git pull
git add .
git status
pause
git commit . -m "auto commit"
git push
