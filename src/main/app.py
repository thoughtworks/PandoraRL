from flask import Flask, render_template, Response

from src.main import DDController, LogController
from src.main import ArtifactController

app = Flask(__name__, static_folder='./public/out', template_folder='./public')
app.register_blueprint(DDController.controller, url_prefix='/api/drugDiscoveryAgent')
app.register_blueprint(ArtifactController.controller, url_prefix='/artifacts/')
app.register_blueprint(LogController.controller, url_prefix='/logs/')


@app.route("/")
def hello():
    return render_template('index.html')


@app.errorhandler(404)
def not_found(e):

    # defining function
    return render_template("404.html")


@app.route('/api/')
def api():
    return Response("Welcome to Pipeline Manager", status=200)


if __name__ == "__main__":
    app.run()
