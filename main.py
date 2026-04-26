from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exception_handlers import http_exception_handler
import pickle
import string

app = FastAPI()

# Mount static files and templates
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')


class PorterStemmer:
    def stem(self, word: str) -> str:
        suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                return word[:-len(suffix)]
        return word


stopwords_set = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
    'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they',
    'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
    'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
    "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

ps = PorterStemmer()


def transform_text(text: str) -> str:
    text = text.lower()
    words = [
        ps.stem(word.strip(string.punctuation))
        for word in text.split()
        if word.strip(string.punctuation).isalnum() and word.lower() not in stopwords_set
    ]
    return ' '.join(words)


try:
    with open('modelmnb.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except Exception as e:
    print('Error loading model/vectorizer:', e)
    model = None
    vectorizer = None


@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context={
            'request': request,
            'prediction': None,
            'message_text': '',
        },
    )


@app.post('/predict', response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    if not model or not vectorizer:
        return templates.TemplateResponse(
            request=request,
            name='index.html',
            context={
                'request': request,
                'result': 'Model or vectorizer not loaded.',
                'prediction': None,
                'message_text': text,
            },
        )

    transformed_sms = transform_text(text)
    vector_input = vectorizer.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context={
            'request': request,
            'prediction': result,
            'message_text': text,
        },
    )


@app.get('/download-model', response_class=HTMLResponse)
async def download_model(request: Request):
    return templates.TemplateResponse(
        request=request,
        name='model.html',
        context={'request': request},
    )


@app.get('/about', response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse(
        request=request,
        name='about.html',
        context={'request': request},
    )


@app.get('/contact', response_class=HTMLResponse)
async def contact(request: Request):
    return templates.TemplateResponse(
        request=request,
        name='contact.html',
        context={'request': request},
    )


@app.get('/dataPrivacy', response_class=HTMLResponse)
async def data_privacy(request: Request):
    return templates.TemplateResponse(
        request=request,
        name='dataprivacy.html',
        context={'request': request},
    )


@app.get('/model', response_class=HTMLResponse)
async def model_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name='model.html',
        context={'request': request},
    )


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return templates.TemplateResponse(
            request=request,
            name='404.html',
            context={'request': request},
            status_code=404,
        )
    return await http_exception_handler(request, exc)
