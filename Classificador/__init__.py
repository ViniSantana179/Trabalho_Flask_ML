from flask import Flask, render_template, url_for
app = Flask(__name__)

from Classificador import routes