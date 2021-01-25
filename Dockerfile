FROM tiangolo/uvicorn-gunicorn:python3.8
RUN pip install --upgrade pip
RUN pip install --no-cache-dir fastapi \
				scikit-learn==0.23.2 \
				nltk \
				spacy==2.3.2 \
				pandas \
				sklearn \
				bs4 \
				spacy-udpipe \
				transformers==3.4.0 \
				tensorflow==2.3.1 \
				keras==2.4.3 \
				torch \ 
				spacy-udpipe

RUN python -m nltk.downloader stopwords
RUN python -m spacy download 'nl_core_news_lg'
RUN python -m spacy download 'de_core_news_lg'
RUN python -m spacy download 'en_core_web_lg'
RUN python -m spacy download 'fr_core_news_lg'
RUN python -m spacy download 'it_core_news_lg'
RUN python -m spacy download 'nb_core_news_lg'



COPY ./app /app