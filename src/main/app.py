from flask import Flask, render_template, Response

from src.main import DDController

app = Flask(__name__, static_folder='./public/out', template_folder='./public')
app.register_blueprint(DDController.controller, url_prefix='/api/drugDiscoveryAgent')


@app.route("/")
def hello():
    return render_template('index.html')


@app.route('/api/')
def api():
    return Response("Welcome to Pipeline Manager", status=200)


if __name__ == "__main__":
    app.run()
